"""Bounded-staleness policy and dynamic threshold scaling.

Every worker update is tagged with the server version it was computed from.
Upon arrival at the parameter server, the staleness is tau = v_server - v_worker.

The policy applies:
- Full weight (w = 1.0) if tau <= S_low
- Downweighting (w = 1 / (1 + beta * tau)) if S_low < tau <= S_max
- Rejection (accept = False, w = 0.0) if tau > S_max
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StalenessConfig:
    enabled: bool = True
    s_low: int = 2
    s_max: int = 10
    beta: float = 0.5


@dataclass
class StalenessDecision:
    accept: bool
    weight: float
    kind: str  # "accepted", "downweighted", "rejected"


def decide_staleness(tau: int, config: StalenessConfig) -> StalenessDecision:
    """Evaluate bounded staleness policy on an update with staleness tau."""
    if not config.enabled:
        return StalenessDecision(accept=True, weight=1.0, kind="accepted")

    if tau > config.s_max:
        return StalenessDecision(accept=False, weight=0.0, kind="rejected")

    if tau <= config.s_low:
        return StalenessDecision(accept=True, weight=1.0, kind="accepted")

    w = 1.0 / (1.0 + config.beta * float(tau))
    return StalenessDecision(accept=True, weight=w, kind="downweighted")


def scale_s_max_for_workers(num_workers: int, kappa: float = 1.5, s_low: int = 2) -> int:
    """Dynamically compute a sensible default S_max based on active cluster worker count."""
    return max(s_low + 1, round(kappa * num_workers))
