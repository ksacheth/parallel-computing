import subprocess
import sys
from pathlib import Path

import pytest

from asgc.config import ControllerConfig, load_config
from asgc.controller import ScorePolicy, ThresholdPolicy, compute_window_stats, make_policy
from asgc.metrics import read_events

ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_CONFIG = ROOT / "configs" / "smoke_fmnist_adaptive.yaml"


def make_stats(**overrides):
    base = {
        "loss_trend": 0.0,
        "loss_variability": 0.0,
        "mean_staleness": 0.5,
        "rejected_frac": 0.0,
        "bytes_per_s": 2_000_000.0,
        "updates_per_s": 20.0,
        "residual_norm_var": 0.01,
    }
    base.update(overrides)
    from asgc.controller import WindowStats

    return WindowStats(**base)


def synth_events(losses=(0.5, 0.55), staleness=(0, 1, 2), rejected_every=0):
    events = []
    t = 0.0
    for i, loss in enumerate(losses):
        events.append({"event": "eval", "version": i, "test_loss": loss, "test_accuracy": 0.8, "elapsed_s": t})
        t += 10.0
    for i, tau in enumerate(staleness):
        decision = "rejected" if (rejected_every and i % rejected_every == 0) else "accepted"
        events.append({
            "event": "update", "version": i, "staleness": tau, "decision": decision,
            "payload_bytes": 100_000, "residual_norm": 0.2, "elapsed_s": t,
        })
        t += 1.0
    return events


def test_window_stats_extract_signals():
    stats = compute_window_stats(synth_events())
    assert stats.loss_trend == pytest.approx(0.05)
    assert stats.mean_staleness == pytest.approx(1.0)
    assert stats.rejected_frac == 0.0
    # three updates of 100 kB spread over the 22 s event span
    assert stats.bytes_per_s == pytest.approx(300_000 / 22)
    assert stats.updates_per_s == pytest.approx(3 / 22)
    assert stats.residual_norm_var == pytest.approx(0.0)


def test_threshold_policy_moves_safer_on_instability():
    policy = ThresholdPolicy(ControllerConfig(enabled=True), mode="topk")
    decision = policy.decide(make_stats(loss_trend=0.05), value=0.25, s_max=4)
    assert decision.value == pytest.approx(0.30)  # one bounded delta up
    assert decision.s_max == 3  # one bounded step down
    assert decision.reason == "instability"


def test_threshold_policy_moves_aggressive_when_stable_under_pressure():
    policy = ThresholdPolicy(ControllerConfig(enabled=True), mode="topk")
    decision = policy.decide(make_stats(bytes_per_s=5_000_000.0), value=0.25, s_max=4)
    assert decision.value == pytest.approx(0.20)
    assert decision.s_max == 5
    assert decision.reason == "stable+comm_pressure"


def test_threshold_policy_holds_when_stable_and_cheap():
    policy = ThresholdPolicy(ControllerConfig(enabled=True), mode="topk")
    decision = policy.decide(make_stats(bytes_per_s=100.0), value=0.25, s_max=4)
    assert decision.value == pytest.approx(0.25)
    assert decision.s_max == 4
    assert decision.reason == "hold"


def test_policy_clamps_to_hard_bounds():
    config = ControllerConfig(enabled=True, rho_bounds=(0.05, 0.5), max_delta_rho=0.05)
    policy = ThresholdPolicy(config, mode="topk")
    at_max = policy.decide(make_stats(loss_trend=0.05), value=0.5, s_max=4)
    assert at_max.value == 0.5  # clamped, never beyond the bound
    at_min = policy.decide(make_stats(bytes_per_s=5_000_000.0), value=0.05, s_max=4)
    assert at_min.value == 0.05


def test_score_policy_direction_follows_the_weighted_score():
    config = ControllerConfig(enabled=True, policy="score")
    policy = ScorePolicy(config, mode="topk")
    # heavy rejections dominate -> R > 0 -> safer
    safer = policy.decide(make_stats(rejected_frac=0.5), value=0.25, s_max=4)
    assert safer.value == pytest.approx(0.30) and safer.s_max == 3
    # cheap, stable, rejection-free -> R < 0 -> aggressive
    aggressive = policy.decide(make_stats(bytes_per_s=10_000_000.0), value=0.25, s_max=4)
    assert aggressive.value == pytest.approx(0.20) and aggressive.s_max == 5


def test_make_policy_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_policy(ControllerConfig(enabled=True, policy="wizard"), mode="topk")


def _events_with_eval_losses(losses):
    events = []
    t = 0.0
    for i, loss in enumerate(losses):
        events.append({"event": "eval", "version": i, "test_loss": loss, "test_accuracy": 0.5, "elapsed_s": t})
        t += 5.0
    for i in range(6):
        events.append({
            "event": "update", "version": i, "staleness": 1, "decision": "accepted",
            "payload_bytes": 100_000, "residual_norm": 0.2, "elapsed_s": t,
        })
        t += 1.0
    return events


def test_variability_signal_ignores_fast_monotone_progress():
    stats = compute_window_stats(_events_with_eval_losses((2.0, 1.0, 0.5, 0.3)))
    assert stats.loss_variability == 0.0, "any monotone decay, however steep or convex, must read as stable"


def test_variability_signal_flags_rebounds():
    stats = compute_window_stats(_events_with_eval_losses((0.5, 0.9, 0.5, 0.9)))
    assert stats.loss_variability > 0.1, "loss bouncing back up between evals must read as instability"


def test_persistence_requires_consecutive_unstable_windows():
    config = ControllerConfig(enabled=True, instability_persistence=2)
    policy = ThresholdPolicy(config, mode="topk")
    first = policy.decide(make_stats(loss_trend=0.05), value=0.25, s_max=4)
    assert first.value == 0.25 and first.s_max == 4, "first unstable window must hold"
    second = policy.decide(make_stats(loss_trend=0.05), value=0.25, s_max=4)
    assert second.value == 0.30 and second.s_max == 3, "second consecutive window must move safer"


def test_adaptive_gate_adapts_within_bounds_and_converges():
    config = load_config(ADAPTIVE_CONFIG)
    assert config.controller.enabled and config.controller.policy == "threshold"

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(ADAPTIVE_CONFIG)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr

    events = read_events(ROOT / "results" / "smoke_fmnist_adaptive" / "events.jsonl")
    controls = [e for e in events if e["event"] == "control"]
    assert len(controls) >= 5, "the controller must decide repeatedly over the run"

    lo, hi = config.controller.rho_bounds
    for c in controls:
        assert lo <= c["new_compression"] <= hi
        s_lo, s_hi = config.controller.s_max_bounds
        assert s_lo <= c["new_s_max"] <= s_hi
        assert c["reason"], "every decision must carry its reason for post-hoc analysis"
    for prev, curr in zip(controls, controls[1:]):
        assert abs(curr["new_compression"] - prev["new_compression"]) <= config.controller.max_delta_rho + 1e-9
        assert abs(curr["new_s_max"] - prev["new_s_max"]) <= config.controller.max_delta_s_max
    assert max(c["new_compression"] for c in controls) - min(c["new_compression"] for c in controls) > 0, \
        "the compression strength must actually adapt during the run"

    run_end = next(e for e in events if e["event"] == "run_end")
    assert run_end["applied_updates"] > 0
    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["test_loss"] < evals[0]["test_loss"]
    assert evals[-1]["test_accuracy"] > 0.80
