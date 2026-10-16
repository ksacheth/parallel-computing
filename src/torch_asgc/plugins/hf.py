"""Hugging Face Transformers integration for ASGC.

Enables distributed asynchronous training with gradient compression in Hugging Face:
    from torch_asgc.plugins.hf import ASGCTrainer, ASGCTrainerCallback

    trainer = ASGCTrainer(
        model=model,
        args=training_args,
        client=client,
        compression_mode="topk",
        rho=0.05,
        train_dataset=dataset,
    )
    trainer.train()
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torch_asgc.client import ASGCClient
from torch_asgc.compression import CompressionConfig
from torch_asgc.optim import DistributedOptimizer

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import Trainer, TrainerCallback

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Fallback base classes if transformers is not installed
    class TrainerCallback:  # type: ignore[no-redef]
        pass

    class Trainer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.model = kwargs.get("model", args[0] if args else None)
            optimizers = kwargs.get("optimizers", (None, None))
            self.optimizer = kwargs.get(
                "optimizer",
                optimizers[0] if isinstance(optimizers, (list, tuple)) and len(optimizers) > 0 else None,
            )
            self.callbacks = list(kwargs.get("callbacks", []))

        def create_optimizer(self) -> Any:
            if getattr(self, "optimizer", None) is None and getattr(self, "model", None) is not None:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            return self.optimizer

        def training_step(self, model: nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
            out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            loss = out["loss"] if isinstance(out, dict) and "loss" in out else out.sum()
            loss.backward()
            return loss.detach()


class ASGCTrainerCallback(TrainerCallback):
    """Hugging Face TrainerCallback for ASGC telemetry and lifecycle logging."""

    def __init__(self, client: ASGCClient, optimizer: DistributedOptimizer | None = None) -> None:
        self.client = client
        self.optimizer = optimizer
        self.last_loss: float | None = None

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Capture loss telemetry at the end of each training step and forward to optimizer."""
        if hasattr(state, "log_history") and state.log_history:
            latest = state.log_history[-1]
            if "loss" in latest:
                self.last_loss = float(latest["loss"])
                opt = kwargs.get("optimizer", self.optimizer)
                if isinstance(opt, DistributedOptimizer) and opt.pending_loss is None:
                    opt.pending_loss = self.last_loss


class ASGCTrainer(Trainer):
    """Hugging Face Trainer subclass that automatically wraps models and optimizers with ASGC."""

    def __init__(
        self,
        client: ASGCClient,
        compression_mode: str = "topk",
        rho: float = 0.1,
        bits: int = 8,
        error_feedback: bool = True,
        bucket_cap_mb: float = 25.0,
        local_steps: int = 1,
        compression: CompressionConfig | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE and not kwargs.get("_test_mode", False):
            raise ImportError(
                "Hugging Face transformers is required to use ASGCTrainer. "
                "Install with: pip install transformers"
            )

        self.asgc_client = client
        self.compression_config = (
            compression
            if compression is not None
            else CompressionConfig(mode=compression_mode, rho=rho, bits=bits)
        )
        self.error_feedback = error_feedback
        self.bucket_cap_mb = bucket_cap_mb
        self.local_steps = local_steps

        # Automatically register ASGC callback
        callbacks = list(kwargs.get("callbacks", []))
        self._asgc_callback = next(
            (cb for cb in callbacks if isinstance(cb, ASGCTrainerCallback)),
            None,
        )
        if self._asgc_callback is None:
            self._asgc_callback = ASGCTrainerCallback(client=client)
            callbacks.append(self._asgc_callback)
            kwargs["callbacks"] = callbacks

        kwargs_clean = {k: v for k, v in kwargs.items() if k != "_test_mode"}
        super().__init__(*args, **kwargs_clean)
        if getattr(self, "model", None) is None and "model" in kwargs:
            self.model = kwargs["model"]

    @property
    def compression_mode(self) -> str:
        return self.compression_config.mode

    @property
    def rho(self) -> float:
        return self.compression_config.rho

    @property
    def bits(self) -> int:
        return self.compression_config.bits

    def create_optimizer(self) -> Optimizer:
        """Create base Hugging Face optimizer and wrap with DistributedOptimizer."""
        super().create_optimizer()
        if not isinstance(self.optimizer, DistributedOptimizer):
            self.optimizer = DistributedOptimizer(
                base_optimizer=self.optimizer,
                model=self.model,
                client=self.asgc_client,
                compression=self.compression_config,
                error_feedback=self.error_feedback,
                bucket_cap_mb=self.bucket_cap_mb,
                local_steps=self.local_steps,
            )
        if self._asgc_callback is not None:
            self._asgc_callback.optimizer = self.optimizer
        return self.optimizer

    def training_step(self, model: nn.Module, inputs: dict[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run training step and record scalar loss onto DistributedOptimizer for ASGC telemetry."""
        loss = super().training_step(model, inputs, *args, **kwargs)
        if isinstance(getattr(self, "optimizer", None), DistributedOptimizer) and isinstance(loss, torch.Tensor):
            self.optimizer.pending_loss = float(loss.detach().item())
        if self._asgc_callback is not None and isinstance(loss, torch.Tensor):
            self._asgc_callback.last_loss = float(loss.detach().item())
        return loss


def wrap_hf_trainer(
    trainer: Any,
    client: ASGCClient,
    compression_mode: str = "topk",
    rho: float = 0.1,
    bits: int = 8,
    compression: CompressionConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Helper to wrap an existing Hugging Face Trainer instance's optimizer and callbacks with ASGC."""
    cfg = (
        compression
        if compression is not None
        else CompressionConfig(mode=compression_mode, rho=rho, bits=bits)
    )
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        if not isinstance(trainer.optimizer, DistributedOptimizer):
            trainer.optimizer = DistributedOptimizer(
                base_optimizer=trainer.optimizer,
                model=trainer.model,
                client=client,
                compression=cfg,
                **kwargs,
            )

    cb = ASGCTrainerCallback(
        client=client,
        optimizer=trainer.optimizer if isinstance(getattr(trainer, "optimizer", None), DistributedOptimizer) else None,
    )
    if hasattr(trainer, "add_callback") and callable(trainer.add_callback):
        trainer.add_callback(cb)
    elif hasattr(trainer, "callbacks") and isinstance(trainer.callbacks, list):
        if not any(isinstance(existing, ASGCTrainerCallback) for existing in trainer.callbacks):
            trainer.callbacks.append(cb)
    else:
        trainer.callbacks = [cb]
    return trainer
