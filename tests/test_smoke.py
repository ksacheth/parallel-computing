"""End-to-end stage gate: run the smoke config through the real entry point."""
import subprocess
import sys
from pathlib import Path

from asgc.config import load_config
from asgc.metrics import read_events

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "smoke_fmnist.yaml"


def test_smoke_config_parses():
    config = load_config(SMOKE_CONFIG)
    assert config.workload.dataset == "fashion_mnist"
    assert config.workload.num_workers == 3
    assert config.workload.max_updates == 7500
    assert config.compression.mode == "identity"
    assert config.heterogeneity.worker_delay(1) == 0.05
    assert config.heterogeneity.worker_delay(0) == 0.0


def test_smoke_run_converges():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(SMOKE_CONFIG)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr

    events = read_events(ROOT / "results" / "smoke_fmnist" / "events.jsonl")
    kinds = {e["event"] for e in events}
    assert {"run_start", "update", "eval", "run_end"} <= kinds

    updates = [e for e in events if e["event"] == "update"]
    assert len(updates) == 7500
    versions = [e["version"] for e in updates]
    assert versions == sorted(versions), "server versions must be non-decreasing"
    assert len(set(versions)) == len(versions), "each applied update must get a unique version"

    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["test_loss"] < evals[0]["test_loss"], "test loss must decrease during training"
    assert evals[-1]["test_accuracy"] > 0.85, "smoke run must converge to reasonable accuracy"

    run_end = next(e for e in events if e["event"] == "run_end")
    assert run_end["applied_updates"] == 7500
    assert run_end["total_bytes"] > 0
