import subprocess
import sys
from pathlib import Path

import pytest
import torch

from asgc.compression import (
    IdentityCompressor,
    QuantizeCompressor,
    TopKCompressor,
    make_compressor,
)
from asgc.config import CompressionConfig
from asgc.metrics import read_events

ROOT = Path(__file__).resolve().parents[1]
COMPRESSION_CONFIG = ROOT / "configs" / "smoke_fmnist_compression.yaml"


def test_identity_roundtrip_is_exact():
    grads = [torch.randn(4, 3), torch.randn(7)]
    compressor = make_compressor(CompressionConfig(mode="identity"))
    payload = compressor.encode(grads)
    decoded = compressor.decode(payload, grads)
    for original, rebuilt in zip(grads, decoded):
        assert torch.equal(original, rebuilt)


def test_identity_bytes_match_dense_size():
    compressor = IdentityCompressor()
    payload = compressor.encode([torch.randn(4, 3), torch.randn(7)])
    assert compressor.encoded_bytes(payload) == (12 + 7) * 4


def test_topk_selects_largest_magnitude_entries():
    u = torch.tensor([0.1, -3.0, 0.5, 2.0])
    compressor = TopKCompressor(rho=0.5)  # k = 2 of 4
    (piece,) = compressor.encode([u])
    assert set(piece.indices.tolist()) == {1, 3}
    assert piece.values.dtype == torch.float32
    assert piece.indices.dtype == torch.int32
    # values carry their signs
    assert dict(zip(piece.indices.tolist(), piece.values.tolist())) == {1: -3.0, 3: 2.0}


def test_topk_decode_scatters_into_zeros():
    u = torch.tensor([0.1, -3.0, 0.5, 2.0])
    compressor = TopKCompressor(rho=0.5)
    (piece,) = compressor.encode([u])
    (rebuilt,) = compressor.decode([piece], [u])
    assert rebuilt[1].item() == -3.0 and rebuilt[3].item() == 2.0
    assert rebuilt[0].item() == 0.0 and rebuilt[2].item() == 0.0


def test_topk_one_point_oh_keeps_everything():
    u = torch.randn(10)
    compressor = TopKCompressor(rho=1.0)
    (piece,) = compressor.encode([u])
    assert piece.indices.numel() == 10
    (rebuilt,) = compressor.decode([piece], [u])
    assert torch.equal(rebuilt, u)


def test_topk_bytes_count_indices_and_values():
    compressor = TopKCompressor(rho=0.25)
    (piece,) = compressor.encode([torch.randn(1000)])
    # k = 250: 250 * (4 bytes index + 4 bytes value) = 2000 = 0.5x dense
    assert compressor.encoded_bytes([piece]) == 2000


def test_topk_rejects_invalid_rho():
    with pytest.raises(ValueError):
        TopKCompressor(rho=0.0)


def test_quantize_codes_stay_in_range_and_error_is_bounded():
    torch.manual_seed(0)
    u = torch.randn(500) * 3.0
    compressor = QuantizeCompressor(bits=8)  # s = 127
    (piece,) = compressor.encode([u])
    assert piece.codes.dtype == torch.int8
    assert piece.codes.abs().max() <= 127
    (rebuilt,) = compressor.decode([piece], [u])
    # stochastic rounding keeps every element within one code of the truth;
    # the encoded scale already includes 1/s, so the bound is scale itself
    assert (rebuilt - u).abs().max() <= piece.scale.item() + 1e-6


def test_quantize_zero_tensor_is_safe():
    u = torch.zeros(10)
    compressor = QuantizeCompressor(bits=8)
    (piece,) = compressor.encode([u])
    (rebuilt,) = compressor.decode([piece], [u])
    assert torch.equal(rebuilt, u)
    assert not rebuilt.isnan().any()


def test_quantize_bytes_count_codes_plus_scale():
    compressor = QuantizeCompressor(bits=8)
    (piece,) = compressor.encode([torch.randn(1000)])
    assert compressor.encoded_bytes([piece]) == 1000 + 4


def test_quantize_rejects_invalid_bits():
    with pytest.raises(ValueError):
        QuantizeCompressor(bits=1)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        make_compressor(CompressionConfig(mode="superzip"))


def test_compression_gate_runs_and_compresses():
    """Stage gate: Top-K rho=0.25 without error feedback still converges
    (loosely, by design — EF in stage 4 is the recovery mechanism) at CR ~2x."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(COMPRESSION_CONFIG)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr

    events = read_events(ROOT / "results" / "smoke_fmnist_compression" / "events.jsonl")
    run_end = next(e for e in events if e["event"] == "run_end")
    assert run_end["compression_ratio"] >= 1.8
    assert run_end["applied_updates"] > 0

    evals = [e for e in events if e["event"] == "eval"]
    assert evals[-1]["test_loss"] < evals[0]["test_loss"]
    assert evals[-1]["test_accuracy"] > 0.75
