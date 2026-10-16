"""Fixed bounded-staleness policy: accept, downweight, or reject arriving updates.

Pure and stateless by design: the server measures tau = v_s - v_i and asks this
module for a decision, so the policy is unit-testable without any processes.
"""
from __future__ import annotations

from dataclasses import dataclass

from asgc.config import StalenessConfig


@dataclass
class Decision:
    accept: bool
    weight: float  # gradient scale for accepted updates; 1.0 when not downweighted


def decide(tau: int, config: StalenessConfig) -> Decision:
    if not config.enabled:
        return Decision(accept=True, weight=1.0)
    if tau > config.s_max:
        return Decision(accept=False, weight=0.0)
    if tau <= config.s_low:
        return Decision(accept=True, weight=1.0)
    return Decision(accept=True, weight=1.0 / (1.0 + config.beta * tau))
