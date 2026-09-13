import subprocess
import sys
from pathlib import Path

import torch

from asgc.compression import IdentityCompressor, TopKCompressor
from asgc.config import load_config
from asgc.metrics import read_events

ROOT = Path(__file__).resolve().parents[1]
TOPK_EF_CONFIG = ROOT / "configs" / "smoke_fmnist_topk_ef.yaml"


def test_ef_conservation_with_topk():
    """Telescoping invariant: d_t = g_t + e_{t-1} - e_t, so summing over all
    steps gives  sum(transmitted) = sum(original gradients) - e_final — every
    bit of computed gradient mass is eventually transmitted except what still
    sits in the residual. The heart of error feedback."""
    compressor = TopKCompressor(rho=0.5)
    like = [torch.zeros(8)]
    e = torch.zeros(8)
    total_g = torch.zeros(8)
    total_received = torch.zeros(8)
    torch.manual_seed(0)
    for _ in range(50):
        g = torch.randn(8)
        u = g + e
        total_g += g
        (piece,) = compressor.encode([u])
        (decoded,) = compressor.decode([piece], like)
        e = u - decoded
        total_received += decoded
    # float32 accumulation over 50 steps leaves ~1e-4 rounding noise; a real
    # logic leak (mass lost or double-counted) would be orders of magnitude larger
    assert torch.allclose(total_received, total_g - e, atol=1e-3, rtol=1e-4)


def test_ef_residual_stays_zero_under_identity():
    """Under identity compression the reconstruction equals u exactly, so the
    residual must remain zero — EF is a no-op, never a perturbation."""
    compressor = IdentityCompressor()
    like = [torch.zeros(5)]
    e = torch.zeros(5)
    torch.manual_seed(0)
    for _ in range(10):
        g = torch.randn(5)
        u = g + e
        (piece,) = compressor.encode([u])
        (decoded,) = compressor.decode([piece], like)
        e = u - decoded
    assert torch.equal(e, torch.zeros(5))


def test_ef_does_not_change_wire_bytes():
    compressor = TopKCompressor(rho=0.25)
    (piece,) = compressor.encode([torch.randn(1000)])
    assert compressor.encoded_bytes([piece]) == 250 * (4 + 4)


def test_ef_gate_recovers_accuracy_at_the_same_bytes():
    """Stage gate: identical config to the stage 3 no-EF gate except
    error_feedback.enabled — same wire bytes, accuracy should recover toward
    the uncompressed reference."""
    config = load_config(TOPK_EF_CONFIG)
    assert config.error_feedback.enabled
    assert config.compression.mode == "topk" and config.compression.rho == 0.25

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(TOPK_EF_CONFIG)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr

    events = read_events(ROOT / "results" / "smoke_fmnist_topk_ef" / "events.jsonl")
    run_end = next(e for e in events if e["event"] == "run_end")
    assert run_end["compression_ratio"] >= 1.8, "EF must not change the wire format"

    updates = [e for e in events if e["event"] == "update"]
    assert any(e["residual_norm"] > 0 for e in updates), "residual norms must be logged when EF is on"

    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["test_loss"] < evals[0]["test_loss"]
    assert evals[-1]["test_accuracy"] > 0.84, "EF should recover most of the no-EF accuracy gap"
