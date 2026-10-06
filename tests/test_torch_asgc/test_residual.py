import torch
from torch_asgc.engine.residual import ErrorFeedbackManager


def test_residual_accumulate_commit():
    device = torch.device("cpu")
    ef = ErrorFeedbackManager(size=4, device=device, enabled=True)

    g1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u1 = ef.accumulate(g1)
    assert torch.allclose(u1, g1)

    # Suppose compressor keeps only entries >= 3.0
    reconstructed1 = torch.tensor([0.0, 0.0, 3.0, 4.0])
    ef.record_tentative(u1, reconstructed1)

    # Server accepts -> commit
    ef.commit()
    assert torch.allclose(ef.residual, torch.tensor([1.0, 2.0, 0.0, 0.0]))

    # Step 2: residual is accumulated into new gradient
    g2 = torch.tensor([0.5, 0.5, 0.5, 0.5])
    u2 = ef.accumulate(g2)
    assert torch.allclose(u2, torch.tensor([1.5, 2.5, 0.5, 0.5]))


def test_residual_rollback_on_rejection():
    device = torch.device("cpu")
    ef = ErrorFeedbackManager(size=4, device=device, enabled=True)

    g1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u1 = ef.accumulate(g1)

    reconstructed1 = torch.tensor([0.0, 0.0, 3.0, 4.0])
    ef.record_tentative(u1, reconstructed1)

    # Server rejects -> rollback
    ef.rollback()
    # Residual should equal u1, preserving all gradient mass for next step!
    assert torch.allclose(ef.residual, torch.tensor([1.0, 2.0, 3.0, 4.0]))
