"""Real network TCP transport built on ``multiprocessing.connection`` framing.

Enables multi-node distributed training over real TCP sockets (for example a
parameter server on a cloud VM and workers on separate Colab runtimes) without
external dependencies like pyzmq or gRPC.

Security model: every connection must pass the standard HMAC challenge/response
handshake keyed by ``authkey`` before any payload is unpickled. Use a long random
secret (``ASGC_AUTHKEY``) whenever the server listens on a public interface.
Traffic is authenticated but not encrypted.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import queue
import socket
import threading
import time
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
from typing import Any

import torch

from torch_asgc.transport.base import (
    FetchRequest,
    FetchResponse,
    PushAck,
    PushUpdate,
    TransportClient,
    TransportServer,
)

logger = logging.getLogger(__name__)

DEFAULT_AUTHKEY = b"torch_asgc"
AUTHKEY_ENV_VAR = "ASGC_AUTHKEY"


def resolve_authkey(authkey: bytes | str | None) -> bytes:
    """Explicit key wins, then the ``ASGC_AUTHKEY`` environment variable, then the default."""
    if authkey is None:
        authkey = os.environ.get(AUTHKEY_ENV_VAR) or DEFAULT_AUTHKEY
    if isinstance(authkey, str):
        authkey = authkey.encode("utf-8")
    return authkey


def encode_message(message: Any) -> bytes:
    """Serialize a message with *standard* pickle so tensor data travels by value.

    ``Connection.send`` uses ``ForkingPickler``, and ``import torch`` registers
    shared-memory reducers on it: tensors would be sent as file-descriptor handles
    that only resolve on the sender's machine. That works on localhost and fails
    with ``FileNotFoundError`` across machines, so this transport always sends bytes.
    """
    return pickle.dumps(_to_cpu(message), protocol=pickle.HIGHEST_PROTOCOL)


def decode_message(data: bytes) -> Any:
    return pickle.loads(data)


def _to_cpu(obj: Any) -> Any:
    """Move every tensor in a message to CPU before pickling.

    A GPU worker's compressed payloads live on CUDA; unpickling CUDA tensors on a
    CPU-only parameter server (or vice versa) fails, so the wire format is CPU-only.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {f.name: _to_cpu(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init}
        return dataclasses.replace(obj, **changes)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def _tune_socket(sock: socket.socket) -> None:
    """Low-latency, dead-peer-detecting settings for long-lived WAN connections.

    ``multiprocessing.connection`` writes a length header and a payload as separate
    sends, so Nagle's algorithm would add a delayed-ACK stall to every message.
    """
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)):
        if hasattr(socket, opt):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), value)


class TCPTransportClient(TransportClient):
    """Network TCP client connecting to a remote ParameterServer.

    Args:
        address: ``(host, port)`` of the parameter server.
        authkey: Shared secret; falls back to ``ASGC_AUTHKEY`` then the default.
        connect_timeout: Seconds allowed for one TCP connect plus handshake.
        connect_retries: Extra connection attempts (useful while the server boots).
        retry_delay: Seconds between connection attempts.
        reply_timeout: Seconds to wait for a server reply before raising
            ``TimeoutError``; ``None`` waits forever.
    """

    def __init__(
        self,
        address: tuple[str, int],
        authkey: bytes | str | None = None,
        connect_timeout: float = 30.0,
        connect_retries: int = 0,
        retry_delay: float = 5.0,
        reply_timeout: float | None = 600.0,
    ) -> None:
        self.address = address
        self.authkey = resolve_authkey(authkey)
        self.reply_timeout = reply_timeout
        self._lock = threading.Lock()
        self._conn = self._connect_with_retries(connect_timeout, connect_retries, retry_delay)

    def _connect_with_retries(self, timeout: float, retries: int, delay: float) -> Connection:
        for attempt in range(retries + 1):
            try:
                return self._open_connection(timeout)
            except (OSError, EOFError) as exc:
                if attempt == retries:
                    raise ConnectionError(
                        f"could not reach parameter server at {self.address[0]}:{self.address[1]}: {exc}"
                    ) from exc
                logger.warning("connect attempt %d failed (%s); retrying in %.0fs", attempt + 1, exc, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _open_connection(self, timeout: float) -> Connection:
        sock = socket.create_connection(self.address, timeout=timeout)
        _tune_socket(sock)
        sock.settimeout(None)  # Connection performs blocking fd I/O
        conn = Connection(sock.detach())
        try:
            # Client side of the handshake mirrors multiprocessing.connection.Client
            answer_challenge(conn, self.authkey)
            deliver_challenge(conn, self.authkey)
        except Exception:
            conn.close()
            raise
        return conn

    def _request(self, message: tuple[str, Any]) -> Any:
        data = encode_message(message)
        with self._lock:
            self._conn.send_bytes(data)
            if self.reply_timeout is not None and not self._conn.poll(self.reply_timeout):
                raise TimeoutError(f"no reply from parameter server within {self.reply_timeout:.0f}s")
            return decode_message(self._conn.recv_bytes())

    def push(self, update: PushUpdate) -> PushAck:
        return self._request(("push", update))

    def fetch(self, request: FetchRequest) -> FetchResponse:
        return self._request(("fetch", request))

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.send_bytes(encode_message(("close", None)))
                self._conn.close()
        except Exception:
            pass


_STOP = object()


class _ConnectionWriter:
    """Sends one connection's outgoing messages, in order, on a dedicated thread.

    Writing a multi-megabyte parameter delta over a high-latency link takes seconds.
    Doing it on the parameter server's polling thread would serialize every
    worker's download behind the slowest one; a writer per connection lets the
    downloads overlap and keeps the server applying pushes meanwhile.
    """

    def __init__(self, conn: Connection, peer: Any) -> None:
        self._conn = conn
        self._peer = peer
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def send(self, message: Any) -> None:
        self._queue.put(message)

    def close(self) -> None:
        self._queue.put(_STOP)

    def _run(self) -> None:
        while (message := self._queue.get()) is not _STOP:
            try:
                self._conn.send_bytes(encode_message(message))
            except Exception as exc:
                logger.warning("send to %s failed: %s", self._peer, exc)
                return


class TCPTransportServer(TransportServer):
    """Network TCP server listening for worker connections.

    The accept loop only accepts sockets; each connection authenticates on its own
    thread, so a failed or stalled handshake (wrong key, port scanner) never blocks
    or stops other workers from connecting. Replies are queued to a per-connection
    writer thread, so ``send_*`` calls return immediately and downloads to
    different workers proceed in parallel.
    """

    def __init__(self, address: tuple[str, int], authkey: bytes | str | None = None) -> None:
        self.authkey = resolve_authkey(authkey)
        self._sock = socket.create_server(address, backlog=64)
        self.address = self._sock.getsockname()[:2]
        self.running = True
        self._connections: dict[str, Any] = {}
        self._push_queue: list[tuple[Any, PushUpdate]] = []
        self._fetch_queue: list[tuple[Any, FetchRequest]] = []
        self._lock = threading.Lock()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    @property
    def connected_workers(self) -> list[str]:
        with self._lock:
            return sorted(self._connections)

    def _accept_loop(self) -> None:
        while self.running:
            try:
                sock, peer = self._sock.accept()
            except OSError:
                if not self.running:
                    break
                time.sleep(0.1)  # transient accept failure (e.g. EMFILE); keep serving
                continue
            threading.Thread(target=self._client_handler, args=(sock, peer), daemon=True).start()

    def _authenticate(self, sock: socket.socket, peer: Any) -> Connection | None:
        _tune_socket(sock)
        conn = Connection(sock.detach())
        try:
            # Server side mirrors multiprocessing.connection.Listener.accept
            deliver_challenge(conn, self.authkey)
            answer_challenge(conn, self.authkey)
        except Exception as exc:
            logger.warning("rejected connection from %s: %s", peer, exc)
            conn.close()
            return None
        return conn

    def _client_handler(self, sock: socket.socket, peer: Any) -> None:
        conn = self._authenticate(sock, peer)
        if conn is None:
            return
        logger.info("authenticated connection from %s", peer)
        writer = _ConnectionWriter(conn, peer)
        worker_ids: set[str] = set()
        while self.running:
            try:
                msg_type, payload = decode_message(conn.recv_bytes())
            except (EOFError, OSError):
                break
            if msg_type == "close":
                break
            if msg_type not in ("push", "fetch"):
                continue
            with self._lock:
                if payload.worker_id not in self._connections:
                    logger.info("worker %s joined from %s", payload.worker_id, peer)
                self._connections[payload.worker_id] = writer
                worker_ids.add(payload.worker_id)
                pending = self._push_queue if msg_type == "push" else self._fetch_queue
                pending.append((conn, payload))
        writer.close()
        self._drop_connection(conn, writer, worker_ids, peer)

    def _drop_connection(self, conn: Any, writer: _ConnectionWriter, worker_ids: set[str], peer: Any) -> None:
        with self._lock:
            for worker_id in worker_ids:
                # A reconnecting worker may already own a newer connection
                if self._connections.get(worker_id) is writer:
                    del self._connections[worker_id]
                    logger.info("worker %s disconnected (%s)", worker_id, peer)
        try:
            conn.close()
        except Exception:
            pass

    def poll_push(self, timeout: float = 0.0) -> PushUpdate | None:
        with self._lock:
            if self._push_queue:
                _, update = self._push_queue.pop(0)
                return update
        return None

    def send_push_ack(self, worker_id: str, ack: PushAck) -> None:
        self._send(worker_id, ack)

    def poll_fetch(self, timeout: float = 0.0) -> FetchRequest | None:
        with self._lock:
            if self._fetch_queue:
                _, req = self._fetch_queue.pop(0)
                return req
        return None

    def send_fetch_response(self, worker_id: str, response: FetchResponse) -> None:
        self._send(worker_id, response)

    def _send(self, worker_id: str, message: Any) -> None:
        with self._lock:
            writer = self._connections.get(worker_id)
        if writer is not None:
            writer.send(message)

    def close(self) -> None:
        self.running = False
        try:
            self._sock.shutdown(socket.SHUT_RDWR)  # wakes a blocked accept() on Linux
        except OSError:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
