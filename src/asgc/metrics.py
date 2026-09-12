"""JSONL event logging: the single source of truth for controller inputs and plots.

The server owns the writer; every event is one JSON line in
``<results_dir>/<run_name>/events.jsonl``, truncated at the start of each run.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunStartEvent:
    event: str
    config_name: str
    num_workers: int
    dataset: str
    model: str
    batch_size: int
    compute_delay: dict[int, float]
    comm_delay_ms: float
    started_at: float


@dataclass
class UpdateEvent:
    event: str
    worker_id: int
    version: int  # server version after the event; unchanged for rejections
    staleness: int  # server version at arrival minus worker version
    payload_bytes: int
    fetch_s: float
    compute_s: float
    decision: str  # accepted | downweighted | rejected
    weight: float  # gradient scale applied; 0.0 for rejections
    elapsed_s: float  # seconds since run start


@dataclass
class EvalEvent:
    event: str
    version: int
    test_accuracy: float
    test_loss: float
    elapsed_s: float


@dataclass
class RunEndEvent:
    event: str
    applied_updates: int
    rejected_updates: int
    rejected_frac: float
    total_bytes: int
    updates_per_sec: float
    elapsed_s: float
    mean_staleness: float
    max_staleness: int
    final_test_accuracy: float
    final_test_loss: float


class EventWriter:
    def __init__(self, results_dir: str, run_name: str) -> None:
        self.run_dir = Path(results_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, event: Any) -> None:
        self._fh.write(json.dumps(asdict(event)) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_events(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Key numbers for the end-of-run summary printed by the entry point."""
    run_end = next((e for e in events if e["event"] == "run_end"), {})
    updates = [e for e in events if e["event"] == "update"]
    evals = [e for e in events if e["event"] == "eval"]
    final_eval = evals[-1] if evals else {}
    return {
        "applied_updates": run_end.get("applied_updates", len(updates)),
        "rejected_updates": run_end.get("rejected_updates", 0),
        "rejected_frac": run_end.get("rejected_frac", 0.0),
        "final_test_accuracy": run_end.get("final_test_accuracy", final_eval.get("test_accuracy")),
        "final_test_loss": run_end.get("final_test_loss", final_eval.get("test_loss")),
        "total_bytes": run_end.get("total_bytes", sum(e["payload_bytes"] for e in updates)),
        "updates_per_sec": run_end.get("updates_per_sec", 0.0),
        "mean_staleness": run_end.get("mean_staleness", 0.0),
        "max_staleness": run_end.get("max_staleness", 0),
        "elapsed_s": run_end.get("elapsed_s", 0.0),
    }
