"""Real network TCP transport using Python standard library multiprocessing.connection.

Enables multi-node distributed training over real TCP sockets without requiring
external dependencies like pyzmq or gRPC.
"""
from __future__ import annotations

from multiprocessing.connection import Client, Listener
import threading
from typing import Any

from torch_asgc.transport.base import (
    FetchRequest,
    FetchResponse,
    PushAck,
    PushUpdate,
    TransportClient,
    TransportServer,
)


class TCPTransportClient(TransportClient):
    """Network TCP client connecting to a remote ParameterServer."""

    def __init__(self, address: tuple[str, int], authkey: bytes = b"torch_asgc") -> None:
        self.address = address
        self.authkey = authkey
        self._lock = threading.Lock()
        self._conn = Client(address, authkey=authkey)

    def push(self, update: PushUpdate) -> PushAck:
        with self._lock:
            self._conn.send(("push", update))
            ack: PushAck = self._conn.recv()
            return ack

    def fetch(self, request: FetchRequest) -> FetchResponse:
        with self._lock:
            self._conn.send(("fetch", request))
            response: FetchResponse = self._conn.recv()
            return response

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.send(("close", None))
                self._conn.close()
        except Exception:
            pass


class TCPTransportServer(TransportServer):
    """Network TCP server listening for worker connections."""

    def __init__(self, address: tuple[str, int], authkey: bytes = b"torch_asgc") -> None:
        self.listener = Listener(address, authkey=authkey)
        self.running = True
        self._connections: dict[str, Any] = {}
        self._push_queue: list[tuple[Any, PushUpdate]] = []
        self._fetch_queue: list[tuple[Any, FetchRequest]] = []
        self._lock = threading.Lock()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while self.running:
            try:
                conn = self.listener.accept()
                t = threading.Thread(target=self._client_handler, args=(conn,), daemon=True)
                t.start()
            except Exception:
                break

    def _client_handler(self, conn: Any) -> None:
        while self.running:
            try:
                msg_type, payload = conn.recv()
                if msg_type == "push":
                    with self._lock:
                        self._connections[payload.worker_id] = conn
                        self._push_queue.append((conn, payload))
                elif msg_type == "fetch":
                    with self._lock:
                        self._connections[payload.worker_id] = conn
                        self._fetch_queue.append((conn, payload))
                elif msg_type == "close":
                    break
            except (EOFError, ConnectionResetError):
                break
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
        with self._lock:
            conn = self._connections.get(worker_id)
        if conn is not None:
            try:
                conn.send(ack)
            except Exception:
                pass

    def poll_fetch(self, timeout: float = 0.0) -> FetchRequest | None:
        with self._lock:
            if self._fetch_queue:
                _, req = self._fetch_queue.pop(0)
                return req
        return None

    def send_fetch_response(self, worker_id: str, response: FetchResponse) -> None:
        with self._lock:
            conn = self._connections.get(worker_id)
        if conn is not None:
            try:
                conn.send(response)
            except Exception:
                pass

    def close(self) -> None:
        self.running = False
        try:
            self.listener.close()
        except Exception:
            pass
