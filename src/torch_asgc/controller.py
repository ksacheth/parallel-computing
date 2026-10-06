"""Runtime adaptive controller for ASGC.

Monitors streaming metrics (loss trend, loss variance, staleness, rejection rate)
and dynamically adjusts compression ratio and staleness thresholds (S_max).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import statistics


@dataclass
class AdaptiveConfig:
    enabled: bool = True
    interval_s: float = 10.0
    mode: str = "score"  # "threshold" or "score"
    # Compression bounds
    rho_bounds: tuple[float, float] = (0.01, 1.0)
    max_delta_rho: float = 0.05
    bits_bounds: tuple[int, int] = (2, 8)
    max_delta_bits: int = 1
    # Staleness bounds
    s_max_bounds: tuple[int, int] = (2, 32)
    max_delta_s_max: int = 2
    # Threshold sensitivities
    worsen_tol: float = 0.02
    variability_tol: float = 0.05
    reject_frac_max: float = 0.20


@dataclass
class TelemetryWindow:
    losses: list[float] = field(default_factory=list)
    stalenesses: list[int] = field(default_factory=list)
    rejected_count: int = 0
    total_updates: int = 0
    total_bytes: int = 0
    duration_s: float = 1.0

    def add_update(self, staleness: int, accepted: bool, num_bytes: int) -> None:
        self.stalenesses.append(staleness)
        self.total_updates += 1
        self.total_bytes += num_bytes
        if not accepted:
            self.rejected_count += 1

    def add_loss(self, loss: float) -> None:
        self.losses.append(loss)

    @property
    def loss_trend(self) -> float:
        if len(self.losses) < 2:
            return 0.0
        return self.losses[-1] - self.losses[0]

    @property
    def loss_variance(self) -> float:
        if len(self.losses) < 2:
            return 0.0
        return statistics.variance(self.losses)

    @property
    def mean_staleness(self) -> float:
        if not self.stalenesses:
            return 0.0
        return sum(self.stalenesses) / len(self.stalenesses)

    @property
    def rejected_frac(self) -> float:
        if self.total_updates == 0:
            return 0.0
        return self.rejected_count / self.total_updates


@dataclass
class ControllerDecision:
    new_compression_strength: float
    new_s_max: int
    reason: str


class AdaptiveController:
    """Adjusts compression strength and S_max based on telemetry."""

    def __init__(self, config: AdaptiveConfig, compression_mode: str = "topk") -> None:
        self.config = config
        self.compression_mode = compression_mode

    def decide(
        self,
        window: TelemetryWindow,
        current_strength: float,
        current_s_max: int,
    ) -> ControllerDecision:
        if not self.config.enabled:
            return ControllerDecision(
                new_compression_strength=current_strength,
                new_s_max=current_s_max,
                reason="controller disabled",
            )

        c = self.config
        unstable = (
            window.loss_trend > c.worsen_tol
            or window.loss_variance > c.variability_tol
            or window.rejected_frac > c.reject_frac_max
        )

        if unstable:
            # Move towards safety: less compression (higher rho/bits) and tighter S_max
            if self.compression_mode == "topk":
                new_strength = min(c.rho_bounds[1], current_strength + c.max_delta_rho)
            else:
                new_strength = min(float(c.bits_bounds[1]), current_strength + float(c.max_delta_bits))

            new_s_max = max(c.s_max_bounds[0], current_s_max - c.max_delta_s_max)
            return ControllerDecision(
                new_compression_strength=new_strength,
                new_s_max=new_s_max,
                reason="unstable signals: moving to safer compression and tighter staleness",
            )

        # Stable signals: increase compression (lower rho/bits) and relax S_max
        if self.compression_mode == "topk":
            new_strength = max(c.rho_bounds[0], current_strength - c.max_delta_rho)
        else:
            new_strength = max(float(c.bits_bounds[0]), current_strength - float(c.max_delta_bits))

        new_s_max = min(c.s_max_bounds[1], current_s_max + c.max_delta_s_max)
        return ControllerDecision(
            new_compression_strength=new_strength,
            new_s_max=new_s_max,
            reason="stable signals: increasing compression and loosening staleness",
        )
