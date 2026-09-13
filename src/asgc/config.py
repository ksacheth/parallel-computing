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
class ControllerConfig:
    """Runtime controller adapting compression strength and s_max together.

    Disabled by default reproduces the fixed policies of stages 2-4. The
    control vector per the methodology is c = (compression, tau_max): rho for
    topk mode, bits for quantize mode; s_low and beta stay fixed. All
    decisions are bounded per interval and clamped to the hard bounds.
    """

    enabled: bool = False
    interval_s: float = 30.0
    policy: str = "threshold"  # threshold | score
    # threshold policy: instability tolerances and the communication-pressure
    # point (bytes/s) above which stable training justifies stronger compression
    worsen_tol: float = 0.01
    variability_tol: float = 0.02
    reject_frac_max: float = 0.15
    comm_pressure_min: float = 1_000_000.0
    # score policy: R = a*trend + b*variability + c*comm + d*staleness
    score_weights: dict[str, float] = field(
        default_factory=lambda: {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    )
    # hard bounds and per-interval anti-oscillation deltas
    rho_bounds: tuple[float, float] = (0.05, 0.5)
    bits_bounds: tuple[int, int] = (2, 8)
    s_max_bounds: tuple[int, int] = (2, 10)
    max_delta_rho: float = 0.05
    max_delta_bits: int = 1
    max_delta_s_max: int = 1


@dataclass
class ExperimentConfig:
    run: RunConfig
    workload: WorkloadConfig
    heterogeneity: HeterogeneityConfig = field(default_factory=HeterogeneityConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    staleness: StalenessConfig = field(default_factory=StalenessConfig)
    error_feedback: ErrorFeedbackConfig = field(default_factory=ErrorFeedbackConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


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
        controller=ControllerConfig(**(raw.get("controller", {}) or {})),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
