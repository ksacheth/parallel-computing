"""Transport abstractions and message schemas for ASGC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass
class PushUpdate:
    """A compressed gradient bucket push from a worker."""

    worker_id: str
    bucket_index: int
    version: int
    payload: Any
    payload_bytes: int
    loss_val: float | None = None
    residual_norm: float = 0.0


@dataclass
class PushAck:
    """Server acknowledgment for a gradient push."""

    status: str  # "accepted", "downweighted", "rejected"
    weight: float
    current_version: int
    compression_setting: float | None = None  # updated rho or bits if changed
    s_max: int | None = None


@dataclass
class FetchRequest:
    """Worker request for model parameters."""

    worker_id: str
    known_version: int


@dataclass
class FetchResponse:
    """Server response containing model weights or deltas."""

    version: int
    params: dict[str, torch.Tensor] | None  # None if worker already has latest
    stop: bool = False
    compression_setting: float | None = None
    s_max: int | None = None


class TransportClient(Protocol):
    """Client-side transport interface used by workers."""

    def push(self, update: PushUpdate) -> PushAck: ...

    def fetch(self, request: FetchRequest) -> FetchResponse: ...

    def close(self) -> None: ...


class TransportServer(Protocol):
    """Server-side transport interface used by the parameter server."""

    def poll_push(self, timeout: float = 0.0) -> PushUpdate | None: ...

    def send_push_ack(self, worker_id: str, ack: PushAck) -> None: ...

    def poll_fetch(self, timeout: float = 0.0) -> FetchRequest | None: ...

    def send_fetch_response(self, worker_id: str, response: FetchResponse) -> None: ...

    def close(self) -> None: ...
