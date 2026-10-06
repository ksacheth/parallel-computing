"""Server-side stateful optimizer engine for staleness-weighted updates.

Wraps standard PyTorch optimizers (SGD, AdamW, Adam) or applies staleness-weighted
gradients with Delay-Compensated Momentum and decoupled weight decay.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer


class StatefulServerOptimizer:
    """Applies staleness-weighted updates to server model parameters.

    Integrates with standard PyTorch Optimizers while properly scaling gradients
    by staleness weight w(tau) before momentum and second-moment updates.
    """

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

            # Scale gradient by staleness attenuation weight w(tau)
            scaled_g = (g * weight).to(device=param.device, dtype=param.dtype)

            if param.grad is None:
                param.grad = scaled_g
            else:
                param.grad.copy_(scaled_g)

        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.params_dict.values()),
                max_norm=self.clip_grad_norm,
            )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
