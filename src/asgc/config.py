"""Typed experiment configuration: YAML in, picklable dataclasses out."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class RunConfig:
    name: str
    seed: int = 0
    results_dir: str = "results"
    data_root: str = "data"


@dataclass
class WorkloadConfig:
    dataset: str = "fashion_mnist"  # fashion_mnist | cifar10
    model: str = "small_cnn"  # small_cnn | resnet18
    batch_size: int = 128
    num_workers: int = 3
    lr: float = 0.05
    lr_decay_frac: float = 0.0  # drop lr by lr_gamma at this fraction of max_updates; 0 disables
    lr_gamma: float = 0.1
    max_updates: int = 500
    max_wall_time_s: float = 600.0
    eval_interval: int = 50
    target_accuracy: float | None = None


@dataclass
class HeterogeneityConfig:
    # worker id -> artificial compute delay in seconds; missing workers run undelayed
    compute_delay: dict[int, float] = field(default_factory=dict)
    comm_delay_ms: float = 0.0

    def worker_delay(self, worker_id: int) -> float:
        return self.compute_delay.get(worker_id, 0.0)


@dataclass
class CompressionConfig:
    mode: str = "identity"  # identity | topk | quantize
    rho: float = 0.25  # topk active fraction k/d
    bits: int = 8  # quantize bit width (levels = 2^(bits-1) - 1)


@dataclass
class StalenessConfig:
    """Fixed bounded-staleness policy. Disabled reproduces plain async SGD.

    Regions: tau <= s_low accept at full weight; s_low < tau <= s_max
    downweight by 1/(1 + beta*tau); tau > s_max reject.
    """

    enabled: bool = False
    s_low: int = 2
    s_max: int = 6
    beta: float = 0.5


@dataclass
class ErrorFeedbackConfig:
    """Worker-side error feedback: e <- u - D(C(u)) with u = g + e.

    Disabled by default. Under identity compression EF is a mathematical
    no-op (the reconstruction equals u, so the residual stays zero).
    """

    enabled: bool = False


@dataclass
class ExperimentConfig:
    run: RunConfig
    workload: WorkloadConfig
    heterogeneity: HeterogeneityConfig = field(default_factory=HeterogeneityConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    staleness: StalenessConfig = field(default_factory=StalenessConfig)
    error_feedback: ErrorFeedbackConfig = field(default_factory=ErrorFeedbackConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    heterogeneity = raw.get("heterogeneity", {}) or {}
    # YAML may parse delay keys as ints or strings depending on quoting
    heterogeneity["compute_delay"] = {
        int(k): float(v) for k, v in (heterogeneity.get("compute_delay", {}) or {}).items()
    }
    return ExperimentConfig(
        run=RunConfig(**(raw.get("run", {}) or {})),
        workload=WorkloadConfig(**(raw.get("workload", {}) or {})),
        heterogeneity=HeterogeneityConfig(**heterogeneity),
        compression=CompressionConfig(**(raw.get("compression", {}) or {})),
        staleness=StalenessConfig(**(raw.get("staleness", {}) or {})),
        error_feedback=ErrorFeedbackConfig(**(raw.get("error_feedback", {}) or {})),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
