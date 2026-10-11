"""Distributed optimizer wrapper for ASGC.

Intercepts standard PyTorch optimizer.step() calls to handle:
- 1D contiguous gradient bucketing
- Micro-local step gradient accumulation (H >= 1)
- Worker-side pre-residual gradient clipping
- Error-feedback residual tracking (u = g + e)
- Bucket compression (Top-K / QSGD)
- Asynchronous push to parameter server
- Transactional residual rollback if update is rejected
- Parameter synchronization with the parameter server (quantized deltas or snapshots)
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torch_asgc.client import ASGCClient
from torch_asgc.compression import CompressionConfig, make_compressor
from torch_asgc.engine.bucketing import create_buckets
from torch_asgc.engine.residual import ErrorFeedbackManager
from torch_asgc.transport.base import BucketPayload


class DistributedOptimizer(Optimizer):
    """Wraps an arbitrary PyTorch optimizer to enable ASGC distributed training."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        model: nn.Module,
        client: ASGCClient,
        compression_mode: str = "topk",
        rho: float = 0.1,
        bits: int = 8,
        error_feedback: bool = True,
        bucket_cap_mb: float = 25.0,
        local_steps: int = 1,
        max_grad_norm: float | None = None,
        compression: CompressionConfig | None = None,
    ) -> None:
        self.base_optimizer = base_optimizer
        self.model = model
        self.client = client
        self.error_feedback_enabled = error_feedback
        self.max_grad_norm = max_grad_norm
        self._step_count = 0
        self.pending_loss: float | None = None

        # Active compression configuration
        self.compression_config = (
            compression
            if compression is not None
            else CompressionConfig(mode=compression_mode, rho=rho, bits=bits)
        )
        self.compressor = make_compressor(self.compression_config)

        # 1D Buckets and Error Feedback Managers
        self.buckets = create_buckets(model, bucket_cap_mb=bucket_cap_mb)
        self.ef_managers: list[ErrorFeedbackManager] = [
            ErrorFeedbackManager(
                size=b.total_numel,
                device=b.device,
                enabled=error_feedback,
            )
            for b in self.buckets
        ]

        self.params_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.last_sync_version = max(0, client.known_version)

        # Micro-Local Steps (H >= 1) gradient accumulation buffers
        self.local_steps = max(1, local_steps)
        self._local_step_counter = 0
        self._local_grad_accum: list[torch.Tensor] = [
            torch.zeros(b.total_numel, dtype=torch.float32, device=b.device)
            for b in self.buckets
        ]

    @property
    def compression_mode(self) -> str:
        return self.compression_config.mode

    @property
    def current_rho(self) -> float:
        return self.compression_config.rho

    @property
    def current_bits(self) -> int:
        return self.compression_config.bits

    @property
    def defaults(self) -> dict[str, Any]:
        return self.base_optimizer.defaults

    @defaults.setter
    def defaults(self, value: dict[str, Any]) -> None:
        self.base_optimizer.defaults = value

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.base_optimizer.param_groups = value

    @property
    def state(self) -> dict[Any, Any]:
        return self.base_optimizer.state

    def state_dict(self) -> dict[str, Any]:
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        self.base_optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        closure: Callable[[], float] | None = None,
        loss_val: float | None = None,
        scaler: Any | None = None,
    ) -> float | None:
        """Perform a distributed optimization step.

        Args:
            closure: Optional closure function for reevaluating the model.
            loss_val: Optional loss scalar value for adaptive telemetry.
            scaler: Optional PyTorch GradScaler instance. If provided, gradients
                will be unscaled before compression and checked for Inf/NaN.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                if loss_val is None and isinstance(loss, (torch.Tensor, float)):
                    loss_val = float(loss)

        if loss_val is None and self.pending_loss is not None:
            loss_val = self.pending_loss
        self.pending_loss = None

        # Handle optional GradScaler unscaling
        if scaler is not None and getattr(scaler, "is_enabled", lambda: True)():
            try:
                scaler.unscale_(self)
            except RuntimeError:
                pass  # already unscaled by caller

        # Extract and check for NaN / Inf overflow across all buckets BEFORE local accumulation
        raw_bucket_grads: list[torch.Tensor] = []
        for bucket in self.buckets:
            flat_g = bucket.flatten(self.params_dict)
            if not torch.isfinite(flat_g).all():
                # Non-finite gradient (overflow in AMP): skip micro-step safely
                return None
            raw_bucket_grads.append(flat_g)

        self._step_count += 1

        # Micro-Local Steps: accumulate gradients locally across H micro-batches
        if self.local_steps > 1:
            for acc, flat_g in zip(self._local_grad_accum, raw_bucket_grads):
                acc.add_(flat_g, alpha=1.0 / float(self.local_steps))
            self._local_step_counter += 1
            if self._local_step_counter < self.local_steps:
                return loss
            self._local_step_counter = 0
            flattened_list = [acc.clone() for acc in self._local_grad_accum]
            for acc in self._local_grad_accum:
                acc.zero_()
        else:
            flattened_list = raw_bucket_grads

        # Optional worker-side pre-residual gradient clipping
        if self.max_grad_norm is not None and self.max_grad_norm > 0.0 and flattened_list:
            sq_norm = sum(torch.sum(g * g) for g in flattened_list)
            total_norm = torch.sqrt(sq_norm).item()
            if total_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (total_norm + 1e-6)
                for g in flattened_list:
                    g.mul_(clip_coef)

        # 1. Process all buckets for this logical step
        bucket_payloads: list[BucketPayload] = []
        for idx, (bucket, ef, flat_g) in enumerate(zip(self.buckets, self.ef_managers, flattened_list)):
            # Error feedback accumulation: u = g + e
            u = ef.accumulate(flat_g)

            # Compress
            payload = self.compressor.encode(u)
            payload_bytes = self.compressor.encoded_bytes(payload)

            # Local reconstruction and tentative residual calculation: e_new = u - D(hat_g)
            reconstructed = self.compressor.decode(payload, target_device=bucket.device)
            ef.record_tentative(u, reconstructed)

            bucket_payloads.append(
                BucketPayload(
                    bucket_index=idx,
                    payload=payload,
                    payload_bytes=payload_bytes,
                    residual_norm=ef.norm(),
                )
            )

        # 2. Push unified step containing all buckets to Parameter Server
        push_version = max(0, self.client.known_version)
        ack = self.client.push_step(
            version=push_version,
            buckets=bucket_payloads,
            loss_val=loss_val,
        )

        # 3. Transactional safety across all buckets
        if ack.status == "rejected":
            for ef in self.ef_managers:
                ef.rollback()
        else:
            for ef in self.ef_managers:
                ef.commit()

        # 4. Adapt compression setting if updated by server controller
        if ack.compression_setting is not None:
            new_cfg = self.compression_config.with_strength(ack.compression_setting)
            if new_cfg != self.compression_config:
                self.compression_config = new_cfg
                self.compressor = make_compressor(self.compression_config)

        # 5. Synchronize model parameters from parameter server
        self.client.sync_model(self.model)
        self.last_sync_version = self.client.known_version

        return loss
