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


@dataclass(frozen=True)
class CompressionConfig:
    """Configuration value object for gradient compression."""

    mode: str = "topk"
    rho: float = 0.1
    bits: int = 8
    stochastic: bool = True

    def __post_init__(self) -> None:
        mode_lower = self.mode.lower()
        normalized_mode = "quantize" if mode_lower == "qsgd" else mode_lower
        object.__setattr__(self, "mode", normalized_mode)
        if self.mode not in ("identity", "topk", "quantize"):
            raise ValueError(f"unknown compression mode: {self.mode!r}")
        if self.mode == "topk" and not (0.0 < self.rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {self.rho}")
        if self.mode == "quantize" and not (2 <= self.bits <= 8):
            raise ValueError(f"bits must be in [2, 8], got {self.bits}")

    @property
    def strength(self) -> float:
        """Active numeric compression strength (rho for topk, bits for quantize)."""
        if self.mode == "topk":
            return self.rho
        if self.mode == "quantize":
            return float(self.bits)
        return 1.0

    def with_strength(self, value: float) -> CompressionConfig:
        """Return a new CompressionConfig with updated compression strength."""
        if self.mode == "topk":
            return CompressionConfig(
                mode=self.mode,
                rho=float(value),
                bits=self.bits,
                stochastic=self.stochastic,
            )
        if self.mode == "quantize":
            return CompressionConfig(
                mode=self.mode,
                rho=self.rho,
                bits=int(round(value)),
                stochastic=self.stochastic,
            )
        return self


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
        # 4 bytes per int32 index + 4 bytes per float32 value
        return self.indices.numel() * 4 + self.values.numel() * self.values.element_size()


@dataclass
class QuantizedPiece:
    """Quantized 1D tensor: int8 codes and float32 pre-divided scale (norm / levels)."""

    codes: torch.Tensor  # int8 in [-levels, levels]
    scale: torch.Tensor  # float32 scalar (||u||_2 / levels)
    total_numel: int

    def encoded_bytes(self) -> int:
        # 1 byte per int8 code + 4 bytes for float32 scale
        return self.codes.numel() * 1 + self.scale.element_size()


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
    """QSGD symmetric quantization to s = 2^(bits-1) - 1 levels per side.

    Folds 1/s into ``scale`` so ``decode`` is a pure function of the payload and
    cannot race when the adaptive controller adjusts ``bits`` while updates are in flight.
    """

    def __init__(self, bits: int = 8, stochastic: bool = True) -> None:
        if not 2 <= bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {bits}")
        self.bits = bits
        self.levels = (1 << (bits - 1)) - 1  # e.g., 127 for 8-bit signed
        self.stochastic = stochastic

    def encode(self, tensor: torch.Tensor) -> QuantizedPiece:
        numel = tensor.numel()
        if numel == 0:
            return QuantizedPiece(
                codes=torch.empty(0, dtype=torch.int8, device=tensor.device),
                scale=torch.tensor(0.0, dtype=torch.float32),
                total_numel=0,
            )

        flat = tensor.detach().reshape(-1).to(torch.float32)
        norm = torch.linalg.vector_norm(flat)
        if norm.item() == 0.0 or not torch.isfinite(norm):
            return QuantizedPiece(
                codes=torch.zeros(numel, dtype=torch.int8, device=tensor.device),
                scale=torch.tensor(0.0, dtype=torch.float32),
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
        unit_scale = (norm / self.levels).detach().cpu().to(torch.float32)
        return QuantizedPiece(
            codes=signed_codes,
            scale=unit_scale,
            total_numel=numel,
        )

    def decode(self, payload: CompressedPayload, target_device: torch.device) -> torch.Tensor:
        if not isinstance(payload, QuantizedPiece):
            raise TypeError(f"expected QuantizedPiece, got {type(payload)}")

        if payload.total_numel == 0:
            return torch.empty(0, dtype=torch.float32, device=target_device)

        scale = payload.scale.to(device=target_device, dtype=torch.float32)
        if scale.item() == 0.0:
            return torch.zeros(payload.total_numel, dtype=torch.float32, device=target_device)

        codes = payload.codes.to(device=target_device, dtype=torch.float32)
        return codes * scale

    def encoded_bytes(self, payload: CompressedPayload) -> int:
        return payload.encoded_bytes()


def make_compressor(
    config_or_mode: CompressionConfig | str = "topk",
    **kwargs: float | int | bool,
) -> Compressor:
    """Factory helper to build a compressor instance from a CompressionConfig or mode string."""
    if isinstance(config_or_mode, CompressionConfig):
        cfg = config_or_mode
    else:
        cfg = CompressionConfig(
            mode=config_or_mode,
            rho=float(kwargs.get("rho", 0.1)),
            bits=int(kwargs.get("bits", 8)),
            stochastic=bool(kwargs.get("stochastic", True)),
        )

    if cfg.mode == "identity":
        return IdentityCompressor()
    if cfg.mode == "topk":
        return TopKCompressor(rho=cfg.rho)
    if cfg.mode == "quantize":
        return QuantizeCompressor(bits=cfg.bits, stochastic=cfg.stochastic)
    raise ValueError(f"unknown compression mode: {cfg.mode!r}")
