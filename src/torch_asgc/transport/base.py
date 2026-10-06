"""Transport abstractions and message schemas for ASGC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass
class BucketPayload:
    """Compressed gradient payload for one gradient bucket."""

    bucket_index: int
    payload: Any
    payload_bytes: int
    residual_norm: float = 0.0


class PushUpdate:
    """A unified logical step push from a worker containing one or more buckets."""

    def __init__(
        self,
        worker_id: str,
        version: int,
        buckets: list[BucketPayload] | None = None,
        loss_val: float | None = None,
        # Backward-compatibility arguments for single-bucket pushes:
        bucket_index: int | None = None,
        payload: Any = None,
        payload_bytes: int = 0,
        residual_norm: float = 0.0,
    ) -> None:
        self.worker_id = worker_id
        self.version = version
        self.loss_val = loss_val
        if buckets is not None:
            self.buckets = buckets
        elif bucket_index is not None:
            self.buckets = [
                BucketPayload(
                    bucket_index=bucket_index,
                    payload=payload,
                    payload_bytes=payload_bytes,
                    residual_norm=residual_norm,
                )
            ]
        else:
            self.buckets = []

    @property
    def total_bytes(self) -> int:
        return sum(b.payload_bytes for b in self.buckets)

    # Backward-compatible convenience properties
    @property
    def bucket_index(self) -> int:
        return self.buckets[0].bucket_index if self.buckets else 0

    @property
    def payload(self) -> Any:
        return self.buckets[0].payload if self.buckets else None

    @property
    def payload_bytes(self) -> int:
        return self.total_bytes

    @property
    def residual_norm(self) -> float:
        return self.buckets[0].residual_norm if self.buckets else 0.0


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
    params: dict[str, torch.Tensor] | None = None  # Full snapshot fallback
    delta: dict[str, torch.Tensor] | None = None  # Accumulated parameter delta dictionary
    is_delta: bool = False
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
