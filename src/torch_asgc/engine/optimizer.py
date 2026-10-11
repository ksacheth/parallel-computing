"""Server-side stateful optimizer engine for staleness-weighted updates.

Wraps standard PyTorch optimizers (SGD, AdamW, Adam) and applies staleness
attenuation w(tau) by scaling the effective step size (learning rate) during
optimizer.step(). Passing unscaled gradients into param.grad preserves true
first- and second-moment statistics in adaptive optimizers (preventing AdamW's
m_t / sqrt(v_t) normalization from cancelling out w(tau)).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer


class StatefulServerOptimizer:
    """Applies staleness-weighted updates to server model parameters."""

    def __init__(
        self,
        optimizer: Optimizer,
        parameters: Sequence[tuple[str, nn.Parameter]],
        clip_grad_norm: float | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.params_dict = {name: p for name, p in parameters if p.requires_grad}
        self.clip_grad_norm = clip_grad_norm

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step_with_gradients(
        self,
        unflattened_grads: dict[str, torch.Tensor],
        weight: float,
    ) -> None:
        """Apply staleness-weighted gradient dictionary to the model parameters.

        Args:
            unflattened_grads: Dictionary of {param_name: gradient_tensor}.
            weight: Staleness attenuation weight w in (0, 1].
        """
        if weight <= 0.0:
            return

        for name, param in self.params_dict.items():
            g = unflattened_grads.get(name)
            if g is None:
                continue

            grad_tensor = g.to(device=param.device, dtype=param.dtype)
            if param.grad is None:
                param.grad = grad_tensor.clone()
            else:
                param.grad.copy_(grad_tensor)

        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.params_dict.values()),
                max_norm=self.clip_grad_norm,
            )

        orig_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        try:
            for group, base_lr in zip(self.optimizer.param_groups, orig_lrs):
                group["lr"] = base_lr * float(weight)
            self.optimizer.step()
        finally:
            for group, base_lr in zip(self.optimizer.param_groups, orig_lrs):
                group["lr"] = base_lr

        self.optimizer.zero_grad(set_to_none=True)
