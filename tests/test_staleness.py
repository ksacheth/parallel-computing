import subprocess
import sys
from pathlib import Path

from asgc.config import StalenessConfig
from asgc.metrics import read_events
from asgc.staleness import decide

ROOT = Path(__file__).resolve().parents[1]
STALENESS_CONFIG = ROOT / "configs" / "smoke_fmnist_staleness.yaml"


def test_recent_updates_get_full_weight():
    config = StalenessConfig(enabled=True, s_low=2, s_max=6, beta=0.5)
    for tau in (0, 1, 2):
        decision = decide(tau, config)
        assert decision.accept
        assert decision.weight == 1.0


def test_boundary_at_s_low_is_full_weight_not_downweighted():
    config = StalenessConfig(enabled=True, s_low=2, s_max=6, beta=0.5)
    decision = decide(config.s_low, config)
    assert decision.accept
    assert decision.weight == 1.0


def test_moderately_stale_updates_are_downweighted():
    config = StalenessConfig(enabled=True, s_low=2, s_max=6, beta=0.5)
    for tau in (3, 4, 5, 6):
        decision = decide(tau, config)
        assert decision.accept
        assert decision.weight == 1.0 / (1.0 + config.beta * tau)


def test_weight_formula_spot_check():
    # tau=2 with s_low=0 lands in the downweight region: 1/(1+0.5*2) = 0.5
    config = StalenessConfig(enabled=True, s_low=0, s_max=6, beta=0.5)
    assert decide(2, config).weight == 0.5


def test_boundary_at_s_max_is_downweighted_not_rejected():
    config = StalenessConfig(enabled=True, s_low=2, s_max=6, beta=0.5)
    decision = decide(config.s_max, config)
    assert decision.accept
    assert 0.0 < decision.weight < 1.0


def test_excessively_stale_updates_are_rejected():
    config = StalenessConfig(enabled=True, s_low=2, s_max=6, beta=0.5)
    for tau in (7, 20, 100):
        decision = decide(tau, config)
        assert not decision.accept


def test_disabled_policy_accepts_everything():
    config = StalenessConfig(enabled=False, s_low=2, s_max=6, beta=0.5)
    for tau in (0, 5, 50, 5000):
        decision = decide(tau, config)
        assert decision.accept
        assert decision.weight == 1.0


def test_staleness_gate_rejects_deeply_stale_updates():
    """End-to-end gate: a heavily delayed worker produces real rejections while
    fast workers keep the version stream moving."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(STALENESS_CONFIG)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr

    events = read_events(ROOT / "results" / "smoke_fmnist_staleness" / "events.jsonl")
    updates = [e for e in events if e["event"] == "update"]
    decisions = {e["decision"] for e in updates}
    assert "accepted" in decisions and "rejected" in decisions, "gate must exercise both paths"

    # The server is single-threaded and writes events in processing order, so
    # walking the log replays the version counter exactly: applied updates step
    # by one, rejections never move it.
    prev = 0
    for e in updates:
        if e["decision"] == "rejected":
            assert e["version"] == prev, "a rejection must not advance the server version"
        else:
            assert e["version"] == prev + 1, "an accepted update must advance the version by exactly one"
            prev = e["version"]

    run_end = next(e for e in events if e["event"] == "run_end")
    assert run_end["rejected_updates"] > 0
    assert run_end["max_staleness"] > 4, "the delayed worker must reach beyond s_max = 4"

    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["test_loss"] < evals[0]["test_loss"]
    assert evals[-1]["test_accuracy"] > 0.80
