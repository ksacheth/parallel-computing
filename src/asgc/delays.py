"""Artificial delays that simulate worker heterogeneity on a single machine."""
from __future__ import annotations

import time

from asgc.config import HeterogeneityConfig


def apply_compute_delay(heterogeneity: HeterogeneityConfig, worker_id: int) -> None:
    """Emulate slower worker hardware; runs inside the worker's compute window."""
    time.sleep(heterogeneity.worker_delay(worker_id))
