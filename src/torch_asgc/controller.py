"""Runtime adaptive controller for ASGC.

Monitors streaming telemetry (loss trend, loss rebound variability, communication
pressure, and staleness/rejection rate) and dynamically adjusts compression
strength and staleness thresholds (S_max) via either Threshold rules or the
weighted Score policy R = a*L + b*V + c*C + d*T.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import statistics


def _mean_rebound(losses: list[float]) -> float:
    """Average upward move between consecutive loss observations in the window."""
    if len(losses) < 2:
        return 0.0
    steps = [b - a for a, b in zip(losses, losses[1:])]
    return sum(max(0.0, d) for d in steps) / len(steps)


@dataclass
class AdaptiveConfig:
    enabled: bool = True
    interval_s: float = 10.0
    mode: str = "score"  # "threshold" or "score"
    adapt_compression: bool = True
    adapt_staleness: bool = True
    # Compression bounds
    rho_bounds: tuple[float, float] = (0.01, 1.0)
    max_delta_rho: float = 0.05
    bits_bounds: tuple[int, int] = (2, 8)
    max_delta_bits: int = 1
    # Staleness bounds
    s_max_bounds: tuple[int, int] = (2, 32)
    max_delta_s_max: int = 2
    # Threshold & score sensitivities
    worsen_tol: float = 0.02
    variability_tol: float = 0.05
    reject_frac_max: float = 0.20
    comm_pressure_min: float = 50_000.0
    instability_persistence: int = 1
    deadband: float = 0.25
    score_weights: dict[str, float] = field(
        default_factory=lambda: {"a": 0.35, "b": 0.25, "c": 0.20, "d": 0.20}
    )


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
    def loss_variability(self) -> float:
        return _mean_rebound(self.losses)

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

    @property
    def bytes_per_s(self) -> float:
        return self.total_bytes / max(self.duration_s, 1.0)

    @property
    def updates_per_s(self) -> float:
        return self.total_updates / max(self.duration_s, 1.0)


@dataclass
class ControllerDecision:
    new_compression_strength: float
    new_s_max: int
    reason: str


class AdaptiveController:
    """Adjusts compression strength and S_max based on live telemetry."""

    def __init__(self, config: AdaptiveConfig, compression_mode: str = "topk") -> None:
        if config.mode not in ("threshold", "score"):
            raise ValueError(f"unknown controller mode: {config.mode!r}")
        self.config = config
        self.compression_mode = "quantize" if compression_mode.lower() == "qsgd" else compression_mode.lower()
        self._unstable_streak = 0

    def _bounds_and_delta(self) -> tuple[tuple[float, float], float]:
        c = self.config
        if self.compression_mode == "quantize":
            return (float(c.bits_bounds[0]), float(c.bits_bounds[1])), float(c.max_delta_bits)
        return (float(c.rho_bounds[0]), float(c.rho_bounds[1])), float(c.max_delta_rho)

    def _step_safer(self, current_strength: float, current_s_max: int, reason: str) -> ControllerDecision:
        c = self.config
        (lo, hi), delta = self._bounds_and_delta()
        new_strength = min(current_strength + delta, hi) if c.adapt_compression else current_strength
        new_s_max = max(current_s_max - c.max_delta_s_max, c.s_max_bounds[0]) if c.adapt_staleness else current_s_max
        return ControllerDecision(
            new_compression_strength=new_strength,
            new_s_max=new_s_max,
            reason=reason,
        )

    def _step_aggressive(self, current_strength: float, current_s_max: int, reason: str) -> ControllerDecision:
        c = self.config
        (lo, hi), delta = self._bounds_and_delta()
        new_strength = max(current_strength - delta, lo) if c.adapt_compression else current_strength
        new_s_max = min(current_s_max + c.max_delta_s_max, c.s_max_bounds[1]) if c.adapt_staleness else current_s_max
        return ControllerDecision(
            new_compression_strength=new_strength,
            new_s_max=new_s_max,
            reason=reason,
        )

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

        if window.total_updates == 0 and len(window.losses) < 2:
            return ControllerDecision(
                new_compression_strength=current_strength,
                new_s_max=current_s_max,
                reason="hold (empty window)",
            )

        c = self.config
        if c.mode == "threshold":
            unstable = (
                window.loss_trend > c.worsen_tol
                or window.loss_variability > c.variability_tol
                or window.rejected_frac > c.reject_frac_max
            )
            if unstable:
                self._unstable_streak += 1
                if self._unstable_streak < c.instability_persistence:
                    return ControllerDecision(
                        new_compression_strength=current_strength,
                        new_s_max=current_s_max,
                        reason=f"hold (unstable {self._unstable_streak}/{c.instability_persistence})",
                    )
                return self._step_safer(current_strength, current_s_max, reason="instability")

            self._unstable_streak = 0
            if window.bytes_per_s >= c.comm_pressure_min:
                return self._step_aggressive(current_strength, current_s_max, reason="stable+comm_pressure")
            return ControllerDecision(
                new_compression_strength=current_strength,
                new_s_max=current_s_max,
                reason="hold",
            )

        # Score policy: R = a*L + b*V + c*C + d*T
        w = c.score_weights
        trend = window.loss_trend / max(c.worsen_tol, 1e-9)
        variability = window.loss_variability / max(c.variability_tol, 1e-9)
        comm = 1.0 - window.bytes_per_s / max(c.comm_pressure_min, 1e-9)
        staleness = (
            window.rejected_frac / max(c.reject_frac_max, 1e-9)
            + 0.1 * (window.mean_staleness / max(float(current_s_max), 1.0))
        )
        r = w["a"] * trend + w["b"] * variability + w["c"] * comm + w["d"] * staleness
        if r > c.deadband:
            return self._step_safer(current_strength, current_s_max, reason=f"safer (R={r:.3f})")
        if r < -c.deadband:
            return self._step_aggressive(current_strength, current_s_max, reason=f"aggressive (R={r:.3f})")
        return ControllerDecision(
            new_compression_strength=current_strength,
            new_s_max=current_s_max,
            reason=f"hold (R={r:.3f})",
        )
