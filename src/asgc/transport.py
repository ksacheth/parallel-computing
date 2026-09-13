"""Process-boundary messages and the queue-based worker/server transport.

Byte accounting is attached to the payload at encode time; the transport only
carries what it is given, so communication metrics cannot drift from the bytes
actually moved between processes.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

import torch

from asgc.config import CompressionConfig

STOP_VERSION = -1


@dataclass
class FetchRequest:
    """A worker's request for the current model; carries the version it already
    holds so the server can skip shipping an unchanged snapshot."""

    worker_id: int
    known_version: int


@dataclass
class FetchResponse:
    """Reply to a worker fetch. ``params`` is None either for a stop signal or
    when the worker's snapshot is already current (keep the local copy)."""

    params: dict[str, torch.Tensor] | None
    version: int
    stop: bool = False
    compression: CompressionConfig | None = None


@dataclass
class PushUpdate:
    """A compressed gradient push tagged with the version it was computed from."""

    worker_id: int
    version: int
    payload: list
    payload_bytes: int
    fetch_s: float = 0.0
    compute_s: float = 0.0
    residual_norm: float = 0.0  # error-feedback residual L2 norm; 0.0 when EF disabled


class Transport:
    """Queue-based transport for one server and N workers on a single machine.

    Queues must be created in the parent process so children inherit them,
    which is required under the Windows spawn start method.
    """

    def __init__(self, ctx: mp.context.BaseContext, comm_delay_s: float = 0.0) -> None:
        self.comm_delay_s = comm_delay_s
        self._ctx = ctx
        self.fetch_requests: mp.Queue = ctx.Queue()
        self.pushes: mp.Queue = ctx.Queue()
        self.replies: dict[int, mp.Queue] = {}

    def register_worker(self, worker_id: int) -> None:
        self.replies[worker_id] = self._ctx.Queue()

    # ---- worker side ----

    def fetch(self, worker_id: int, known_version: int) -> FetchResponse:
        self.fetch_requests.put(FetchRequest(worker_id=worker_id, known_version=known_version))
        return self.replies[worker_id].get()

    def push(self, update: PushUpdate) -> None:
        if self.comm_delay_s > 0:
            time.sleep(self.comm_delay_s)
        self.pushes.put(update)

    # ---- server side ----

    def next_fetch_request(self, timeout: float) -> FetchRequest | None:
        try:
            return self.fetch_requests.get(timeout=timeout)
        except queue.Empty:
            return None

    def reply(self, worker_id: int, response: FetchResponse) -> None:
        if self.comm_delay_s > 0:
            time.sleep(self.comm_delay_s)
        self.replies[worker_id].put(response)

    def next_push(self, timeout: float) -> PushUpdate | None:
        try:
            return self.pushes.get(timeout=timeout)
        except queue.Empty:
            return None
