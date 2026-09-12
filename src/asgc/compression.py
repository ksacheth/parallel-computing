"""Gradient compression: encode worker gradients into wire payloads.

A payload is a list of per-parameter tensors aligned with ``model.parameters()``
order. Top-K and quantization land here in stage 3; the protocol is designed so
that adding them requires no changes to the worker or server.
"""
from __future__ import annotations

from typing import Protocol

import torch


class Compressor(Protocol):
    def encode(self, grads: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def decode(self, payload: list[torch.Tensor]) -> list[torch.Tensor]: ...

    def encoded_bytes(self, payload: list[torch.Tensor]) -> int: ...


class IdentityCompressor:
    """No compression: gradients pass through unchanged."""

    def encode(self, grads: list[torch.Tensor]) -> list[torch.Tensor]:
        return [g.detach().clone() for g in grads]

    def decode(self, payload: list[torch.Tensor]) -> list[torch.Tensor]:
        return payload

    def encoded_bytes(self, payload: list[torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in payload)


def make_compressor(mode: str) -> Compressor:
    if mode == "identity":
        return IdentityCompressor()
    raise ValueError(f"unknown compression mode: {mode!r}")
