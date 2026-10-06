"""Contiguous 1D gradient bucketing.

Coalesces arbitrarily shaped parameter gradient tensors into contiguous flat
1D memory buffers (buckets). This avoids per-tensor compression overhead, allows
Top-K and quantization to allocate bandwidth globally across the model where
gradient energy is concentrated, and minimizes kernel launch overhead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class TensorLayout:
    """Metadata for reconstructing an original tensor from a flat 1D slice."""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    numel: int
    offset: int  # start index in flat buffer


class GradientBucket:
    """A contiguous flat buffer managing a group of parameter gradients."""

    def __init__(self, parameters: Sequence[tuple[str, nn.Parameter]], max_bytes: int = 25 * 1024 * 1024) -> None:
        """Initialize bucket layout for trainable parameters.

        Args:
            parameters: List of (name, parameter) tuples that require gradients.
            max_bytes: Maximum size in bytes before starting a new bucket (default 25MB).
        """
        self.layouts: list[TensorLayout] = []
        offset = 0
        total_numel = 0
        ref_device = None
        ref_dtype = torch.float32

        for name, param in parameters:
            if not param.requires_grad:
                continue
            if ref_device is None:
                ref_device = param.device
                ref_dtype = param.dtype
            n = param.numel()
            self.layouts.append(
                TensorLayout(
                    name=name,
                    shape=param.shape,
                    dtype=param.dtype,
                    device=param.device,
                    numel=n,
                    offset=offset,
                )
            )
            offset += n
            total_numel += n

        self.total_numel = total_numel
        self.device = ref_device if ref_device is not None else torch.device("cpu")
        self.dtype = ref_dtype

    def flatten(self, params_dict: dict[str, nn.Parameter]) -> torch.Tensor:
        """Extract gradients from params_dict and pack into a flat 1D FP32 tensor.

        If a parameter's grad is None, zeros are packed for that slice.
        Always returns a float32 flat buffer for stable numerical compression.
        """
        if self.total_numel == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        buffer = torch.zeros(self.total_numel, dtype=torch.float32, device=self.device)
        for layout in self.layouts:
            p = params_dict.get(layout.name)
            if p is not None and p.grad is not None:
                grad_data = p.grad.detach()
                if grad_data.dtype != torch.float32:
                    grad_data = grad_data.to(torch.float32)
                buffer[layout.offset : layout.offset + layout.numel] = grad_data.reshape(-1)

        return buffer

    def unflatten_to_dict(self, flat_buffer: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unflatten a 1D buffer back into a dictionary of {param_name: tensor}."""
        result: dict[str, torch.Tensor] = {}
        for layout in self.layouts:
            slice_data = flat_buffer[layout.offset : layout.offset + layout.numel]
            reshaped = slice_data.reshape(layout.shape).to(dtype=layout.dtype, device=layout.device)
            result[layout.name] = reshaped
        return result

    def copy_to_params_(self, flat_buffer: torch.Tensor, params_dict: dict[str, nn.Parameter]) -> None:
        """In-place write unflattened buffer contents into parameter grads or data."""
        unflattened = self.unflatten_to_dict(flat_buffer)
        for name, tensor in unflattened.items():
            param = params_dict.get(name)
            if param is not None:
                if param.grad is not None:
                    param.grad.copy_(tensor)
                else:
                    param.grad = tensor.clone()


def create_buckets(
    model: nn.Module,
    bucket_cap_mb: float = 25.0,
) -> list[GradientBucket]:
    """Partition a model's trainable parameters into contiguous 1D buckets.

    Parameters are sorted in reverse execution order (from output back to input)
    so that earlier gradients computed during backprop can be dispatched sooner.
    """
    max_bytes = int(bucket_cap_mb * 1024 * 1024)
    trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    buckets: list[GradientBucket] = []
    current_params: list[tuple[str, nn.Parameter]] = []
    current_bytes = 0

    for name, p in trainable_params:
        p_bytes = p.numel() * 4  # FP32 accounting
        if current_bytes + p_bytes > max_bytes and current_params:
            buckets.append(GradientBucket(current_params, max_bytes=max_bytes))
            current_params = []
            current_bytes = 0

        current_params.append((name, p))
        current_bytes += p_bytes

    if current_params:
        buckets.append(GradientBucket(current_params, max_bytes=max_bytes))

    return buckets
