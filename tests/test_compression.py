import pytest
import torch

from asgc.compression import IdentityCompressor, make_compressor


def test_identity_roundtrip_is_exact():
    grads = [torch.randn(4, 3), torch.randn(7)]
    compressor = make_compressor("identity")
    decoded = compressor.decode(compressor.encode(grads))
    for original, rebuilt in zip(grads, decoded):
        assert torch.equal(original, rebuilt)


def test_encoded_bytes_counts_all_elements():
    compressor = IdentityCompressor()
    payload = compressor.encode([torch.randn(4, 3), torch.randn(7)])
    assert compressor.encoded_bytes(payload) == (12 + 7) * 4


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        make_compressor("topk")  # arrives in stage 3
