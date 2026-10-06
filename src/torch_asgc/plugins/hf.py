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
from torch_asgc.model import DistributedModel
from torch_asgc.optim import DistributedOptimizer

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments

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


class ASGCTrainerCallback(TrainerCallback):
    """Hugging Face TrainerCallback for ASGC telemetry and lifecycle logging."""

    def __init__(self, client: ASGCClient) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            pass  # Allow instantiation for testing or standalone usage
        self.client = client
        self.last_loss: float | None = None

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Capture loss telemetry at the end of each training step."""
        if hasattr(state, "log_history") and state.log_history:
            latest = state.log_history[-1]
            if "loss" in latest:
                self.last_loss = float(latest["loss"])


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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE and not kwargs.get("_test_mode", False):
            raise ImportError(
                "Hugging Face transformers is required to use ASGCTrainer. "
                "Install with: pip install transformers"
            )

        self.asgc_client = client
        self.compression_mode = compression_mode
        self.rho = rho
        self.bits = bits
        self.error_feedback = error_feedback
        self.bucket_cap_mb = bucket_cap_mb
        self.local_steps = local_steps

        # Automatically register ASGC callback
        callbacks = kwargs.get("callbacks", [])
        asgc_callback = ASGCTrainerCallback(client=client)
        if asgc_callback not in callbacks:
            callbacks = list(callbacks) + [asgc_callback]
            kwargs["callbacks"] = callbacks

        kwargs_clean = {k: v for k, v in kwargs.items() if k != "_test_mode"}
        super().__init__(*args, **kwargs_clean)
        if getattr(self, "model", None) is None and "model" in kwargs:
            self.model = kwargs["model"]

    def create_optimizer(self) -> Optimizer:
        """Create base Hugging Face optimizer and wrap with DistributedOptimizer."""
        super().create_optimizer()
        if not isinstance(self.optimizer, DistributedOptimizer):
            self.optimizer = DistributedOptimizer(
                base_optimizer=self.optimizer,
                model=self.model,
                client=self.asgc_client,
                compression_mode=self.compression_mode,
                rho=self.rho,
                bits=self.bits,
                error_feedback=self.error_feedback,
                bucket_cap_mb=self.bucket_cap_mb,
                local_steps=self.local_steps,
            )
        return self.optimizer


def wrap_hf_trainer(
    trainer: Any,
    client: ASGCClient,
    compression_mode: str = "topk",
    rho: float = 0.1,
    **kwargs: Any,
) -> Any:
    """Helper to wrap an existing Hugging Face Trainer instance's optimizer with ASGC."""
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        if not isinstance(trainer.optimizer, DistributedOptimizer):
            trainer.optimizer = DistributedOptimizer(
                base_optimizer=trainer.optimizer,
                model=trainer.model,
                client=client,
                compression_mode=compression_mode,
                rho=rho,
                **kwargs,
            )
    return trainer
