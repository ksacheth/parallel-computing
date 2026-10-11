"""PyTorch Lightning integration for ASGC.

Enables distributed asynchronous training with gradient compression in Lightning:
    trainer = pl.Trainer(
        strategy=ASGCStrategy(client=client, compression_mode="topk", rho=0.1),
        max_epochs=10,
    )
"""
from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from torch_asgc.client import ASGCClient
from torch_asgc.compression import CompressionConfig
from torch_asgc.model import DistributedModel
from torch_asgc.optim import DistributedOptimizer

# Support both `lightning.pytorch` (modern) and legacy `pytorch_lightning`
_LIGHTNING_AVAILABLE = False
try:
    from lightning.pytorch.strategies import SingleDeviceStrategy
    from lightning.pytorch.strategies.strategy import Strategy

    _LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        from pytorch_lightning.strategies import SingleDeviceStrategy
        from pytorch_lightning.strategies.strategy import Strategy

        _LIGHTNING_AVAILABLE = True
    except ImportError:
        # Fallback base class if lightning is not installed
        class Strategy:  # type: ignore[no-redef]
            pass

        class SingleDeviceStrategy(Strategy):  # type: ignore[no-redef]
            def __init__(self, device: torch.device = torch.device("cpu"), **kwargs: Any) -> None:
                self._root_device = device


class ASGCStrategy(SingleDeviceStrategy):
    """PyTorch Lightning Strategy for Adaptive Staleness-Aware Gradient Compression."""

    strategy_name = "asgc"

    def __init__(
        self,
        client: ASGCClient,
        compression_mode: str = "topk",
        rho: float = 0.1,
        bits: int = 8,
        error_feedback: bool = True,
        bucket_cap_mb: float = 25.0,
        local_steps: int = 1,
        device: torch.device | str = "cpu",
        compression: CompressionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if not _LIGHTNING_AVAILABLE and not kwargs.get("_test_mode", False):
            raise ImportError(
                "PyTorch Lightning is required to use ASGCStrategy. "
                "Install with: pip install lightning"
            )

        root_device = torch.device(device) if isinstance(device, str) else device
        super().__init__(device=root_device)
        self.client = client
        self.compression_config = (
            compression
            if compression is not None
            else CompressionConfig(mode=compression_mode, rho=rho, bits=bits)
        )
        self.error_feedback = error_feedback
        self.bucket_cap_mb = bucket_cap_mb
        self.local_steps = local_steps
        self._wrapped_model: DistributedModel | None = None
        self._trainer: Any | None = None

    @property
    def compression_mode(self) -> str:
        return self.compression_config.mode

    @property
    def rho(self) -> float:
        return self.compression_config.rho

    @property
    def bits(self) -> int:
        return self.compression_config.bits

    def setup_model(self, model: nn.Module) -> nn.Module:
        """Wrap the LightningModule with DistributedModel."""
        self._wrapped_model = DistributedModel(model, client=self.client)
        return self._wrapped_model

    def setup_optimizers(self, trainer: Any) -> list[Optimizer]:
        """Wrap Lightning's configured optimizers with DistributedOptimizer."""
        self._trainer = trainer
        optimizers = getattr(trainer, "optimizers", [])
        wrapped_optimizers: list[Optimizer] = []

        for opt in optimizers:
            if isinstance(opt, DistributedOptimizer):
                wrapped_optimizers.append(opt)
            else:
                dist_opt = DistributedOptimizer(
                    base_optimizer=opt,
                    model=self._wrapped_model if self._wrapped_model is not None else trainer.lightning_module,
                    client=self.client,
                    compression=self.compression_config,
                    error_feedback=self.error_feedback,
                    bucket_cap_mb=self.bucket_cap_mb,
                    local_steps=self.local_steps,
                )
                wrapped_optimizers.append(dist_opt)

        if hasattr(trainer, "optimizers"):
            trainer.optimizers = wrapped_optimizers
        return wrapped_optimizers

    def _extract_loss_val(self, kwargs: dict[str, Any]) -> float | None:
        if "loss_val" in kwargs and kwargs["loss_val"] is not None:
            return float(kwargs["loss_val"])
        trainer = self._trainer
        if trainer is not None:
            metrics = getattr(trainer, "callback_metrics", None)
            if isinstance(metrics, dict):
                for key in ("train_loss", "train_loss_step", "loss"):
                    if key in metrics and metrics[key] is not None:
                        val = metrics[key]
                        return float(val.detach().item() if isinstance(val, torch.Tensor) else val)
        return None

    def optimizer_step(
        self,
        optimizer: Optimizer,
        closure: Callable[[], Any] | None = None,
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute optimizer step with ASGC compression and live loss telemetry."""
        loss_val = self._extract_loss_val(kwargs)
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "loss_val"}
        if isinstance(optimizer, DistributedOptimizer):
            return optimizer.step(closure=closure, loss_val=loss_val, **clean_kwargs)
        return optimizer.step(closure=closure, **clean_kwargs)
