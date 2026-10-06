"""Ecosystem plugins for torch-asgc (PyTorch Lightning, Hugging Face Transformers)."""
from __future__ import annotations

from torch_asgc.plugins.hf import ASGCTrainer, ASGCTrainerCallback, wrap_hf_trainer
from torch_asgc.plugins.lightning import ASGCStrategy

__all__ = [
    "ASGCStrategy",
    "ASGCTrainer",
    "ASGCTrainerCallback",
    "wrap_hf_trainer",
]
