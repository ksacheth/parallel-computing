"""Stage gate for the synchronous SGD baseline."""
import subprocess
import sys
from pathlib import Path

from asgc.metrics import read_events

ROOT = Path(__file__).resolve().parents[1]
SYNC_CONFIG = ROOT / "configs" / "smoke_fmnist_sync.yaml"


def test_sync_gate_applies_rounds_of_mean_gradients():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(SYNC_CONFIG)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr

    events = read_events(ROOT / "results" / "smoke_fmnist_sync" / "events.jsonl")
    updates = [e for e in events if e["event"] == "update"]
    run_end = next(e for e in events if e["event"] == "run_end")

    # one round per version, one gradient from every worker per round
    assert run_end["applied_updates"] == 600
    assert len(updates) == 600 * 3
    versions = [e["version"] for e in updates]
    assert sorted(set(versions)) == list(range(1, 601))
    assert all(e["staleness"] == 0 for e in updates), "sync updates are never stale"
    assert all(abs(e["weight"] - 1 / 3) < 1e-9 for e in updates), "each worker contributes 1/N"

    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["test_loss"] < evals[0]["test_loss"]
    assert evals[-1]["test_accuracy"] > 0.70
