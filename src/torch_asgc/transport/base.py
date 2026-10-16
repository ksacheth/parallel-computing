"""Transport abstractions, message schemas, and address helpers for ASGC."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class QuantizedDelta:
    """INT8-quantized parameter delta for 4x downlink bandwidth reduction.

    Stores signed int8 codes in [-127, 127] and a float32 unit scale (max_abs / 127.0).
    """

    codes: torch.Tensor  # int8
    scale: torch.Tensor  # float32 scalar
    shape: torch.Size
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> tuple[QuantizedDelta, torch.Tensor]:
        """Quantize a tensor to INT8 and return (QuantizedDelta, reconstructed_fp32)."""
        flat = tensor.detach().to(torch.float32)
        if flat.numel() == 0:
            empty_codes = torch.empty(0, dtype=torch.int8)
            zero_scale = torch.tensor(0.0, dtype=torch.float32)
            return (
                cls(codes=empty_codes, scale=zero_scale, shape=tensor.shape, dtype=tensor.dtype),
                torch.zeros_like(flat),
            )

        max_abs = flat.abs().max()
        if max_abs.item() == 0.0 or not torch.isfinite(max_abs):
            zero_codes = torch.zeros(flat.shape, dtype=torch.int8, device="cpu")
            zero_scale = torch.tensor(0.0, dtype=torch.float32)
            return (
                cls(codes=zero_codes, scale=zero_scale, shape=tensor.shape, dtype=tensor.dtype),
                torch.zeros_like(flat),
            )

        unit_scale = (max_abs / 127.0).detach().cpu().to(torch.float32)
        scale_dev = unit_scale.to(device=flat.device)
        codes_dev = (flat / scale_dev).round().clamp(-127, 127).to(torch.int8)
        reconstructed = codes_dev.to(torch.float32) * scale_dev
        return (
            cls(
                codes=codes_dev.cpu(),
                scale=unit_scale,
                shape=tensor.shape,
                dtype=tensor.dtype,
            ),
            reconstructed,
        )

    def decode(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Reconstruct the delta tensor on the requested device and dtype."""
        target_device = device if device is not None else self.codes.device
        target_dtype = dtype if dtype is not None else self.dtype
        if self.codes.numel() == 0 or self.scale.item() == 0.0:
            return torch.zeros(self.shape, device=target_device, dtype=target_dtype)
        out = self.codes.to(device=target_device, dtype=torch.float32) * self.scale.to(
            device=target_device, dtype=torch.float32
        )
        return out.reshape(self.shape).to(dtype=target_dtype)

    def encoded_bytes(self) -> int:
        return self.codes.numel() * 1 + self.scale.element_size()


@dataclass
class BucketPayload:
    """Compressed gradient payload for one gradient bucket."""

    bucket_index: int
    payload: Any
    payload_bytes: int
    residual_norm: float = 0.0


@dataclass
class PushUpdate:
    """A unified logical step push from a worker containing all gradient buckets."""

    worker_id: str
    version: int
    buckets: list[BucketPayload] = field(default_factory=list)
    loss_val: float | None = None

    @property
    def total_bytes(self) -> int:
        return sum(b.payload_bytes for b in self.buckets)


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
    """Server response containing model weights or quantized parameter deltas."""

    version: int
    params: dict[str, torch.Tensor] | None = None  # Full snapshot fallback
    delta: dict[str, QuantizedDelta | torch.Tensor] | None = None  # Quantized parameter deltas
    buffers: dict[str, torch.Tensor] | None = None  # Non-trainable buffers (e.g. BatchNorm stats)
    is_delta: bool = False
    stop: bool = False
    compression_setting: float | None = None
    s_max: int | None = None

    def encoded_bytes(self) -> int:
        """Calculate wire bytes of the downlink payload."""
        total = 0
        if self.params is not None:
            total += sum(t.numel() * t.element_size() for t in self.params.values())
        if self.delta is not None:
            for d in self.delta.values():
                if isinstance(d, QuantizedDelta):
                    total += d.encoded_bytes()
                else:
                    total += d.numel() * d.element_size()
        if self.buffers is not None:
            total += sum(b.numel() * b.element_size() for b in self.buffers.values())
        return total


def parse_tcp_address(address: str | tuple[str, int]) -> tuple[str, int]:
    """Parse a TCP address from a ('host', port) tuple or 'tcp://host:port' string."""
    if isinstance(address, tuple):
        host, port = address
        return str(host), int(port)
    if isinstance(address, str):
        cleaned = address.replace("tcp://", "").strip()
        if ":" not in cleaned:
            raise ValueError(f"expected 'host:port' or 'tcp://host:port', got {address!r}")
        host, port_str = cleaned.rsplit(":", 1)
        return host, int(port_str)
    raise ValueError(f"unsupported address format: {address!r}")


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
