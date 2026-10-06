"""Local queue-based transport for in-process testing and single-node multiprocessing."""
from __future__ import annotations

import queue
from typing import Any

from torch_asgc.transport.base import (
    FetchRequest,
    FetchResponse,
    PushAck,
    PushUpdate,
    TransportClient,
    TransportServer,
)


class LocalTransportHub:
    """Shared hub containing message queues for local communication."""

    def __init__(self, queue_factory: Any = queue.Queue) -> None:
        self.push_queue = queue_factory()
        self.fetch_queue = queue_factory()
        self.push_acks: dict[str, Any] = {}
        self.fetch_replies: dict[str, Any] = {}
        self._q_factory = queue_factory

    def get_or_create_ack_queue(self, worker_id: str) -> Any:
        if worker_id not in self.push_acks:
            self.push_acks[worker_id] = self._q_factory()
        return self.push_acks[worker_id]

    def get_or_create_reply_queue(self, worker_id: str) -> Any:
        if worker_id not in self.fetch_replies:
            self.fetch_replies[worker_id] = self._q_factory()
        return self.fetch_replies[worker_id]


class LocalTransportClient(TransportClient):
    """Worker-side client communicating through local queues."""

    def __init__(self, hub: LocalTransportHub, worker_id: str) -> None:
        self.hub = hub
        self.worker_id = worker_id
        self.ack_queue = hub.get_or_create_ack_queue(worker_id)
        self.reply_queue = hub.get_or_create_reply_queue(worker_id)

    def push(self, update: PushUpdate) -> PushAck:
        self.hub.push_queue.put(update)
        return self.ack_queue.get()

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.hub.fetch_queue.put(request)
        return self.reply_queue.get()

    def close(self) -> None:
        pass


class LocalTransportServer(TransportServer):
    """Server-side transport communicating through local queues."""

    def __init__(self, hub: LocalTransportHub) -> None:
        self.hub = hub

    def poll_push(self, timeout: float = 0.0) -> PushUpdate | None:
        try:
            return self.hub.push_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_push_ack(self, worker_id: str, ack: PushAck) -> None:
        q = self.hub.get_or_create_ack_queue(worker_id)
        q.put(ack)

    def poll_fetch(self, timeout: float = 0.0) -> FetchRequest | None:
        try:
            return self.hub.fetch_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_fetch_response(self, worker_id: str, response: FetchResponse) -> None:
        q = self.hub.get_or_create_reply_queue(worker_id)
        q.put(response)

    def close(self) -> None:
        pass
