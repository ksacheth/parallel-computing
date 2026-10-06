import torch
from torch_asgc.compression import IdentityCompressor, QuantizeCompressor, TopKCompressor


def test_identity_compression():
    comp = IdentityCompressor()
    tensor = torch.randn(100)
    piece = comp.encode(tensor)
    decoded = comp.decode(piece, torch.device("cpu"))
    assert torch.allclose(decoded, tensor)
    assert comp.encoded_bytes(piece) == 100 * 4


def test_topk_compression():
    comp = TopKCompressor(rho=0.1)
    tensor = torch.zeros(100)
    tensor[5] = 10.0
    tensor[42] = -20.0
    piece = comp.encode(tensor)
    assert piece.indices.numel() == 10
    decoded = comp.decode(piece, torch.device("cpu"))
    assert decoded[5] == 10.0
    assert decoded[42] == -20.0
    assert decoded[0] == 0.0


def test_quantize_compression():
    comp = QuantizeCompressor(bits=8, stochastic=False)
    tensor = torch.linspace(-1.0, 1.0, 50)
    piece = comp.encode(tensor)
    decoded = comp.decode(piece, torch.device("cpu"))
    # Max error bounded by scale / levels
    assert (decoded - tensor).abs().max() < 0.1
