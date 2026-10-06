"""1D Bucket-aware Gradient Compression.

Supports:
- Identity: Uncompressed FP32 baseline.
- TopK: Magnitude sparsification with flat int32 indices and float32 values.
- Quantize: Symmetric stochastic/deterministic QSGD quantization on a float32 L2 scale.

All compressors operate on 1D flat tensors (produced by GradientBucket)
and reconstruct flat 1D tensors with identical shape and device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class DensePiece:
    """Uncompressed 1D flat tensor."""

    values: torch.Tensor

    def encoded_bytes(self) -> int:
        return self.values.numel() * self.values.element_size()


@dataclass
class SparsePiece:
    """Top-K 1D sparse tensor: int32 flat indices and float32 values."""

    indices: torch.Tensor  # int32
    values: torch.Tensor  # float32
    total_numel: int

    def encoded_bytes(self) -> int:
        # 4 bytes for int32 index + 4 bytes for float32 value + 4 bytes metadata
        return self.indices.numel() * 4 + self.values.numel() * 4 + 4


@dataclass
class QuantizedPiece:
    """Quantized 1D tensor: int8 codes and float32 scale."""

    codes: torch.Tensor  # int8
    scale: torch.Tensor  # float32 scalar (L2 norm)
    total_numel: int

    def encoded_bytes(self) -> int:
        # 1 byte per int8 code + 4 bytes for float32 scale + 4 bytes metadata
        return self.codes.numel() * 1 + 4 + 4


CompressedPayload = DensePiece | SparsePiece | QuantizedPiece


class Compressor(Protocol):
    """Protocol for 1D bucket compressors."""

    def encode(self, tensor: torch.Tensor) -> CompressedPayload: ...

    def decode(self, payload: CompressedPayload, target_device: torch.device) -> torch.Tensor: ...

    def encoded_bytes(self, payload: CompressedPayload) -> int: ...


class IdentityCompressor:
    """Passes gradients through uncompressed."""

    def encode(self, tensor: torch.Tensor) -> DensePiece:
        return DensePiece(values=tensor.detach().clone())

    def decode(self, payload: CompressedPayload, target_device: torch.device) -> torch.Tensor:
        if not isinstance(payload, DensePiece):
            raise TypeError(f"expected DensePiece, got {type(payload)}")
        return payload.values.to(device=target_device).clone()

    def encoded_bytes(self, payload: CompressedPayload) -> int:
        return payload.encoded_bytes()


class TopKCompressor:
    """Top-K sparsification: keep the k = max(1, round(rho * d)) largest magnitude entries."""

    def __init__(self, rho: float) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        self.rho = rho

    def encode(self, tensor: torch.Tensor) -> SparsePiece:
        numel = tensor.numel()
        if numel == 0:
            return SparsePiece(
                indices=torch.empty(0, dtype=torch.int32, device=tensor.device),
                values=torch.empty(0, dtype=torch.float32, device=tensor.device),
                total_numel=0,
            )

        k = max(1, round(self.rho * numel))
        flat = tensor.detach().reshape(-1)
        top = torch.topk(flat.abs(), k).indices
        return SparsePiece(
            indices=top.to(torch.int32),
            values=flat[top].to(torch.float32).clone(),
            total_numel=numel,
        )

    def decode(self, payload: CompressedPayload, target_device: torch.device) -> torch.Tensor:
        if not isinstance(payload, SparsePiece):
            raise TypeError(f"expected SparsePiece, got {type(payload)}")

        out = torch.zeros(payload.total_numel, dtype=torch.float32, device=target_device)
        if payload.indices.numel() > 0:
            indices = payload.indices.to(device=target_device, dtype=torch.int64)
            values = payload.values.to(device=target_device, dtype=torch.float32)
            out.scatter_(0, indices, values)
        return out

    def encoded_bytes(self, payload: CompressedPayload) -> int:
        return payload.encoded_bytes()


class QuantizeCompressor:
    """QSGD symmetric stochastic quantization to int8 codes."""

    def __init__(self, bits: int = 8, stochastic: bool = True) -> None:
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.bits = bits
        self.levels = (1 << (bits - 1)) - 1  # e.g., 127 for 8-bit signed
        self.stochastic = stochastic

    def encode(self, tensor: torch.Tensor) -> QuantizedPiece:
        numel = tensor.numel()
        if numel == 0:
            return QuantizedPiece(
                codes=torch.empty(0, dtype=torch.int8, device=tensor.device),
                scale=torch.tensor(0.0, dtype=torch.float32, device=tensor.device),
                total_numel=0,
            )

        flat = tensor.detach().reshape(-1).to(torch.float32)
        norm = torch.linalg.vector_norm(flat)
        if norm.item() == 0.0 or not torch.isfinite(norm):
            return QuantizedPiece(
                codes=torch.zeros(numel, dtype=torch.int8, device=tensor.device),
                scale=torch.tensor(0.0, dtype=torch.float32, device=tensor.device),
                total_numel=numel,
            )

        scaled = flat.abs() / norm * self.levels
        floor_val = scaled.floor()
        if self.stochastic:
            prob = scaled - floor_val
            rounded = floor_val + torch.bernoulli(prob)
        else:
            rounded = scaled.round()

        clamped = rounded.clamp(0, self.levels)
        signed_codes = (clamped * flat.sign()).to(torch.int8)
        return QuantizedPiece(
            codes=signed_codes,
            scale=norm.cpu(),
            total_numel=numel,
        )

    def decode(self, payload: CompressedPayload, target_device: torch.device) -> torch.Tensor:
        if not isinstance(payload, QuantizedPiece):
            raise TypeError(f"expected QuantizedPiece, got {type(payload)}")

        if payload.total_numel == 0:
            return torch.empty(0, dtype=torch.float32, device=target_device)

        scale = payload.scale.to(target_device)
        if scale.item() == 0.0 or self.levels == 0:
            return torch.zeros(payload.total_numel, dtype=torch.float32, device=target_device)

        codes = payload.codes.to(device=target_device, dtype=torch.float32)
        return (codes / self.levels) * scale

    def encoded_bytes(self, payload: CompressedPayload) -> int:
        return payload.encoded_bytes()


def make_compressor(mode: str, **kwargs: float | int | bool) -> Compressor:
    """Factory helper to build a compressor instance."""
    mode = mode.lower()
    if mode == "identity":
        return IdentityCompressor()
    if mode == "topk":
        rho = float(kwargs.get("rho", 0.1))
        return TopKCompressor(rho=rho)
    if mode in ("quantize", "qsgd"):
        bits = int(kwargs.get("bits", 8))
        stochastic = bool(kwargs.get("stochastic", True))
        return QuantizeCompressor(bits=bits, stochastic=stochastic)
    raise ValueError(f"unknown compression mode: {mode!r}")
