"""Distributed optimizer wrapper for ASGC.

Intercepts standard PyTorch optimizer.step() calls to handle:
- 1D contiguous gradient bucketing
- Error-feedback residual tracking (u = g + e)
- Bucket compression (Top-K / QSGD)
- Asynchronous push to parameter server
- Transactional residual rollback if update is rejected
- Parameter synchronization with the parameter server
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torch_asgc.client import ASGCClient
from torch_asgc.compression import make_compressor
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
    ) -> None:
        self.base_optimizer = base_optimizer
        self.model = model
        self.client = client
        self.error_feedback_enabled = error_feedback

        # Active compression state
        self.compression_mode = compression_mode
        self.current_rho = rho
        self.current_bits = bits
        self.compressor = make_compressor(compression_mode, rho=rho, bits=bits)

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
        self.last_sync_version = -1

        # Micro-Local Steps
        self.local_steps = max(1, local_steps)
        self._local_step_counter = 0

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        self.base_optimizer.param_groups = value

    @property
    def state(self) -> dict[Any, Any]:
        return self.base_optimizer.state

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        closure: Callable[[], float] | None = None,
        loss_val: float | None = None,
    ) -> float | None:
        """Perform a distributed optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                if loss_val is None and isinstance(loss, (torch.Tensor, float)):
                    loss_val = float(loss)

        # Micro-Local Steps: compute locally until synchronization boundary
        self._local_step_counter += 1
        if self._local_step_counter < self.local_steps:
            self.base_optimizer.step(closure=closure)
            return loss

        self._local_step_counter = 0

        # 1. Process all buckets for this logical step
        bucket_payloads: list[BucketPayload] = []
        for idx, (bucket, ef) in enumerate(zip(self.buckets, self.ef_managers)):
            flat_g = bucket.flatten(self.params_dict)

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
        ack = self.client.push_step(
            version=self.last_sync_version,
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

        # Adapt compression setting if updated by server controller
        if ack.compression_setting is not None:
            if self.compression_mode == "topk" and ack.compression_setting != self.current_rho:
                self.current_rho = ack.compression_setting
                self.compressor = make_compressor("topk", rho=self.current_rho)
            elif self.compression_mode in ("quantize", "qsgd") and int(ack.compression_setting) != self.current_bits:
                self.current_bits = int(ack.compression_setting)
                self.compressor = make_compressor("quantize", bits=self.current_bits)

        # 2. Synchronize model parameters from parameter server
        resp = self.client.fetch_parameters()
        if resp.is_delta and resp.delta is not None:
            # In-place add delta directly to parameters (zero load_state_dict overhead)
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in resp.delta:
                        d = resp.delta[name].to(device=p.device, dtype=p.dtype)
                        p.add_(d)
            self.last_sync_version = resp.version
        elif resp.params is not None:
            # Update local parameters from server snapshot
            self.model.load_state_dict(resp.params)
            self.last_sync_version = resp.version

        return loss
