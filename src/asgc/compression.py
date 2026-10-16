"""Gradient compression: encode worker gradients into wire payloads.

A payload is one piece per parameter tensor, aligned with ``model.parameters()``
order. Wire formats are counted exactly by ``encoded_bytes`` (float32 dense
values, float32 values + int32 sparse indices, int8 codes + float32 scale), so
communication metrics cannot drift from what is actually moved between
processes. Sub-byte packing for quantization is future work; the accounting
honestly counts the unpacked int8 storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from asgc.config import CompressionConfig


@dataclass
class DensePiece:
    """Uncompressed piece: the raw gradient tensor."""

    values: torch.Tensor


@dataclass
class SparsePiece:
    """Top-K piece: signed values at flat indices into the original tensor."""

    indices: torch.Tensor  # int32 flat indices
    values: torch.Tensor  # float32


@dataclass
class QuantizedPiece:
    """QSGD-style piece: int8 codes on a per-tensor float32 L2 scale."""

    codes: torch.Tensor  # int8 in [-s, s]
    scale: torch.Tensor  # float32 scalar


Piece = DensePiece | SparsePiece | QuantizedPiece


class Compressor(Protocol):
    """Implementations must keep decode a pure function of (payload, like):
    the worker decodes its own payload to maintain its error-feedback
    residual and must see exactly what the server reconstructs."""

    def encode(self, grads: list[torch.Tensor]) -> list[Piece]: ...

    def decode(self, payload: list[Piece], like: Sequence[torch.Tensor]) -> list[torch.Tensor]: ...

    def encoded_bytes(self, payload: list[Piece]) -> int: ...


class IdentityCompressor:
    """No compression: gradients pass through unchanged."""

    def encode(self, grads: list[torch.Tensor]) -> list[Piece]:
        return [DensePiece(g.detach().clone()) for g in grads]

    def decode(self, payload: list[Piece], like: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [p.values.to(ref).clone() for p, ref in zip(payload, like)]

    def encoded_bytes(self, payload: list[Piece]) -> int:
        return sum(p.values.numel() * p.values.element_size() for p in payload)


class TopKCompressor:
    """Keep the k = rho*d largest-magnitude entries per tensor.

    Wire format: float32 values + int32 flat indices, 8 bytes per kept entry.
    Indices are stored int32 (models here stay far below 2^31 parameters) and
    upcast to int64 at decode time; accounting counts the int32 wire format.
    """

    def __init__(self, rho: float) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        self.rho = rho

    def _k(self, numel: int) -> int:
        return max(1, round(self.rho * numel))

    def encode(self, grads: list[torch.Tensor]) -> list[Piece]:
        pieces: list[Piece] = []
        for g in grads:
            flat = g.detach().reshape(-1)
            k = self._k(flat.numel())
            top = torch.topk(flat.abs(), k).indices
            pieces.append(SparsePiece(indices=top.to(torch.int32), values=flat[top].clone()))
        return pieces

    def decode(self, payload: list[Piece], like: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for piece, ref in zip(payload, like):
            # payloads may arrive on CPU from the transport; move to the
            # reference parameter's device before scattering
            dense = torch.zeros_like(ref)
            idx = piece.indices.to(torch.int64).to(ref.device)
            dense.reshape(-1)[idx] = piece.values.to(ref)
            out.append(dense)
        return out

    def encoded_bytes(self, payload: list[Piece]) -> int:
        return sum(p.indices.numel() * 4 + p.values.numel() * p.values.element_size() for p in payload)


class QuantizeCompressor:
    """QSGD-style stochastic quantization to s = 2^(bits-1) - 1 levels per side.

    Each tensor transmits its L2 norm (float32) plus one int8 code per element:
    code = stochastically_round(s * u / ||u||_2). Stochastic rounding keeps the
    estimator unbiased; the worker's per-worker seeding makes runs reproducible.
    """

    def __init__(self, bits: int) -> None:
        if not 2 <= bits <= 8:
            raise ValueError(f"bits must be in [2, 8], got {bits}")
        self.levels = 2 ** (bits - 1) - 1

    def encode(self, grads: list[torch.Tensor]) -> list[Piece]:
        pieces: list[Piece] = []
        for g in grads:
            flat = g.detach().reshape(-1).to(torch.float32)
            norm = flat.norm()
            if norm == 0:
                pieces.append(QuantizedPiece(
                    codes=torch.zeros(flat.numel(), dtype=torch.int8),
                    scale=torch.zeros(()),
                ))
                continue
            scaled = self.levels * flat / norm
            floor = scaled.floor()
            # round up with probability equal to the fractional part
            frac = scaled - floor
            codes = (floor + (torch.rand_like(frac) < frac).to(torch.float32)).to(torch.int8)
            # fold 1/s into the scale so decode never depends on the encoder's
            # bit width — adapted settings cannot race with in-flight payloads
            pieces.append(QuantizedPiece(codes=codes, scale=(norm / self.levels).detach().clone()))
        return pieces

    def decode(self, payload: list[Piece], like: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for piece, ref in zip(payload, like):
            restored = piece.codes.to(torch.float32).to(ref.device) * piece.scale.to(ref.device)
            out.append(restored.reshape(ref.shape).to(ref))
        return out

    def encoded_bytes(self, payload: list[Piece]) -> int:
        return sum(p.codes.numel() * 1 + p.scale.element_size() for p in payload)


def make_compressor(config: CompressionConfig) -> Compressor:
    if config.mode == "identity":
        return IdentityCompressor()
    if config.mode == "topk":
        return TopKCompressor(config.rho)
    if config.mode == "quantize":
        return QuantizeCompressor(config.bits)
    raise ValueError(f"unknown compression mode: {config.mode!r}")
