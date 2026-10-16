import pytest
from torch_asgc.controller import AdaptiveConfig, AdaptiveController, TelemetryWindow


def test_controller_disabled_returns_unchanged_state():
    ctrl = AdaptiveController(
        AdaptiveConfig(enabled=False, mode="score"),
        compression_mode="topk",
    )
    window = TelemetryWindow(
        losses=[2.0, 2.5, 3.0, 3.5],
        stalenesses=[10, 12],
        rejected_count=10,
        total_updates=10,
        total_bytes=100_000,
        duration_s=1.0,
    )
    decision = ctrl.decide(window, current_strength=0.1, current_s_max=8)
    assert decision.new_compression_strength == pytest.approx(0.1)
    assert decision.new_s_max == 8
    assert decision.reason == "controller disabled"


def test_threshold_policy_instability_relaxes_compression_and_tightens_staleness():
    cfg = AdaptiveConfig(
        enabled=True,
        mode="threshold",
        instability_persistence=2,
        max_delta_rho=0.05,
        max_delta_s_max=2,
    )
    ctrl = AdaptiveController(cfg, compression_mode="topk")

    unstable_window = TelemetryWindow(
        losses=[1.0, 1.5, 2.0, 2.5],  # positive loss_trend > worsen_tol
        stalenesses=[6, 7],
        rejected_count=5,
        total_updates=7,  # high rejected_frac
        total_bytes=1000,
        duration_s=1.0,
    )

    # First unstable window increments streak to 1 (< persistence=2) -> hold
    d1 = ctrl.decide(unstable_window, current_strength=0.1, current_s_max=8)
    assert d1.new_compression_strength == pytest.approx(0.1)
    assert d1.new_s_max == 8
    assert "hold" in d1.reason

    # Second unstable window reaches persistence=2 -> safer (increase rho, decrease s_max)
    d2 = ctrl.decide(unstable_window, current_strength=0.1, current_s_max=8)
    assert d2.new_compression_strength == pytest.approx(0.15)
    assert d2.new_s_max == 6
    assert d2.reason == "instability"


def test_threshold_policy_comm_pressure_tightens_compression_and_loosens_staleness():
    cfg = AdaptiveConfig(
        enabled=True,
        mode="threshold",
        comm_pressure_min=50_000.0,
        max_delta_rho=0.05,
        max_delta_s_max=2,
    )
    ctrl = AdaptiveController(cfg, compression_mode="topk")

    # Stable decreasing loss and high bandwidth (100_000 B/s >= 50_000 B/s)
    w = TelemetryWindow(
        losses=[2.0, 1.8, 1.6, 1.4],
        stalenesses=[1, 1],
        rejected_count=0,
        total_updates=10,
        total_bytes=100_000,
        duration_s=1.0,
    )
    d = ctrl.decide(w, current_strength=0.2, current_s_max=6)
    assert d.new_compression_strength == pytest.approx(0.15)  # tighter compression
    assert d.new_s_max == 8  # looser staleness
    assert d.reason == "stable+comm_pressure"


def test_score_policy_and_per_knob_switches():
    # Only adapt compression (quantize bits), keep staleness fixed
    cfg = AdaptiveConfig(
        enabled=True,
        mode="score",
        adapt_compression=True,
        adapt_staleness=False,
        max_delta_bits=1,
        max_delta_s_max=2,
    )
    ctrl = AdaptiveController(cfg, compression_mode="quantize")

    # High instability + staleness -> R > deadband (0.25) -> safer (increase bits), s_max unchanged
    w = TelemetryWindow(
        losses=[1.0, 2.0, 3.5, 5.0],
        stalenesses=[8, 8, 8],
        rejected_count=8,
        total_updates=10,
        total_bytes=500,
        duration_s=1.0,
    )
    d = ctrl.decide(w, current_strength=4.0, current_s_max=8)
    assert d.new_compression_strength == pytest.approx(5.0)
    assert d.new_s_max == 8
    assert d.reason.startswith("safer")
