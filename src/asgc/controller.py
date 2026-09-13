"""Runtime controller: adapt compression strength and s_max from the event stream.

Pure functions of an event window — the server feeds the same event dicts that
go to the JSONL log, never worker internals. Every decision moves the control
vector (compression strength, s_max) by at most the configured per-interval
delta and is clamped to the hard bounds, so settings cannot oscillate.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from asgc.config import ControllerConfig


@dataclass
class WindowStats:
    loss_trend: float  # eval-loss change across the window; positive = worsening
    loss_variability: float  # eval-loss stdev; 0.0 with fewer than two evals
    mean_staleness: float  # mean arrival staleness of updates in the window
    rejected_frac: float  # rejected arrivals / arrivals
    bytes_per_s: float  # communication pressure
    updates_per_s: float  # applied updates per second
    residual_norm_var: float  # variance of worker residual norms (stage 4 signal)


def compute_window_stats(events: list[dict]) -> WindowStats:
    evals = [e for e in events if e.get("event") == "eval"]
    updates = [e for e in events if e.get("event") == "update"]
    losses = [e["test_loss"] for e in evals]
    elapsed = [e["elapsed_s"] for e in events]
    span = max(elapsed) - min(elapsed) if elapsed else 0.0
    span = max(span, 1.0)  # rates stay finite on degenerate windows
    residuals = [e["residual_norm"] for e in updates]
    return WindowStats(
        loss_trend=losses[-1] - losses[0] if len(losses) >= 2 else 0.0,
        loss_variability=statistics.pstdev(losses) if len(losses) >= 2 else 0.0,
        mean_staleness=sum(e["staleness"] for e in updates) / len(updates) if updates else 0.0,
        rejected_frac=sum(e["decision"] == "rejected" for e in updates) / len(updates) if updates else 0.0,
        bytes_per_s=sum(e["payload_bytes"] for e in updates) / span,
        updates_per_s=len(updates) / span,
        residual_norm_var=statistics.pvariance(residuals) if len(residuals) >= 2 else 0.0,
    )


@dataclass
class ControlDecision:
    value: float  # new compression strength: rho (topk) or bits (quantize)
    s_max: int
    reason: str


def _bounds_and_delta(config: ControllerConfig, mode: str) -> tuple[tuple[float, float], float]:
    if mode == "quantize":
        return tuple(config.bits_bounds), config.max_delta_bits
    return tuple(config.rho_bounds), config.max_delta_rho


class ThresholdPolicy:
    """Rule table per the proposal: instability (worsening trend, loss
    variability, or excessive rejections) moves safer; stable training under
    communication pressure moves aggressive; otherwise hold."""

    def __init__(self, config: ControllerConfig, mode: str) -> None:
        self.c = config
        self.mode = mode

    def decide(self, stats: WindowStats, value: float, s_max: int) -> ControlDecision:
        c = self.c
        (lo, hi), delta = _bounds_and_delta(c, self.mode)
        unstable = (
            stats.loss_trend > c.worsen_tol
            or stats.loss_variability > c.variability_tol
            or stats.rejected_frac > c.reject_frac_max
        )
        if unstable:
            return ControlDecision(
                value=min(value + delta, hi),
                s_max=max(s_max - c.max_delta_s_max, c.s_max_bounds[0]),
                reason="instability",
            )
        if stats.bytes_per_s >= c.comm_pressure_min:
            return ControlDecision(
                value=max(value - delta, lo),
                s_max=min(s_max + c.max_delta_s_max, c.s_max_bounds[1]),
                reason="stable+comm_pressure",
            )
        return ControlDecision(value=value, s_max=s_max, reason="hold")


class ScorePolicy:
    """R = a*trend + b*variability + c*comm + d*staleness over terms
    normalized so 1.0 sits at the corresponding tolerance; positive R moves
    safer, negative moves aggressive, |R| inside the deadband holds."""

    deadband = 0.25

    def __init__(self, config: ControllerConfig, mode: str) -> None:
        self.c = config
        self.mode = mode

    def decide(self, stats: WindowStats, value: float, s_max: int) -> ControlDecision:
        c = self.c
        w = c.score_weights
        trend = stats.loss_trend / max(c.worsen_tol, 1e-9)
        variability = stats.loss_variability / max(c.variability_tol, 1e-9)
        # pressure above the threshold pulls R negative -> aggressive; cheap
        # communication pulls slightly safer (no reason to compress harder)
        comm = 1.0 - stats.bytes_per_s / max(c.comm_pressure_min, 1e-9)
        staleness = stats.rejected_frac / max(c.reject_frac_max, 1e-9)
        r = w["a"] * trend + w["b"] * variability + w["c"] * comm + w["d"] * staleness
        reason = f"hold (R={r:.3f})"
        (lo, hi), delta = _bounds_and_delta(c, self.mode)
        if r > self.deadband:
            return ControlDecision(
                value=min(value + delta, hi),
                s_max=max(s_max - c.max_delta_s_max, c.s_max_bounds[0]),
                reason=f"safer (R={r:.3f})",
            )
        if r < -self.deadband:
            return ControlDecision(
                value=max(value - delta, lo),
                s_max=min(s_max + c.max_delta_s_max, c.s_max_bounds[1]),
                reason=f"aggressive (R={r:.3f})",
            )
        return ControlDecision(value=value, s_max=s_max, reason=reason)


def make_policy(config: ControllerConfig, mode: str) -> ThresholdPolicy | ScorePolicy:
    if config.policy == "threshold":
        return ThresholdPolicy(config, mode)
    if config.policy == "score":
        return ScorePolicy(config, mode)
    raise ValueError(f"unknown controller policy: {config.policy!r}")
