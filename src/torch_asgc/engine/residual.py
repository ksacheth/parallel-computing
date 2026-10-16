"""Error-feedback residual management with transactional rollback safety.

In error-feedback training, the worker maintains a residual vector e.
At each step:
    u = g + e
    hat_g = C(u)
    e_tentative = u - D(hat_g)

Crucially, in asynchronous parameter-server training with bounded staleness,
updates can be rejected by the server if their arrival staleness exceeds S_max.
If an update is rejected, subtracting hat_g would permanently lose that gradient mass.
This manager provides transactional commit/rollback semantics:
- commit(): called when update is accepted or downweighted (e <- e_tentative)
- rollback(): called when update is rejected by the server (e <- u)
"""
from __future__ import annotations

import torch


class ErrorFeedbackManager:
    """Tracks error-feedback residual for a 1D gradient buffer with rollback safety."""

    def __init__(self, size: int, device: torch.device, enabled: bool = True) -> None:
        self.size = size
        self.device = device
        self.enabled = enabled
        self.residual = torch.zeros(size, dtype=torch.float32, device=device)
        self._tentative_residual: torch.Tensor | None = None
        self._tentative_u: torch.Tensor | None = None

    def accumulate(self, grads: torch.Tensor) -> torch.Tensor:
        """Add the accumulated residual to incoming gradients: u = g + e."""
        if not self.enabled or self.size == 0:
            return grads

        # Always work in FP32 for master residual stability
        g_fp32 = grads.detach().to(dtype=torch.float32, device=self.device)
        u = g_fp32 + self.residual
        self._tentative_u = u.clone()
        return u

    def record_tentative(self, u: torch.Tensor, reconstructed: torch.Tensor) -> None:
        """Calculate tentative residual after local compression decode: e_new = u - D(C(u))."""
        if not self.enabled or self.size == 0:
            return
        rec_fp32 = reconstructed.detach().to(dtype=torch.float32, device=self.device)
        self._tentative_residual = (u - rec_fp32).detach()

    def commit(self) -> None:
        """Commit the tentative residual (server accepted or downweighted update)."""
        if not self.enabled or self._tentative_residual is None:
            return
        self.residual.copy_(self._tentative_residual)
        self._tentative_residual = None
        self._tentative_u = None

    def rollback(self) -> None:
        """Roll back the residual (server rejected update); preserves unapplied gradient mass."""
        if not self.enabled or self._tentative_u is None:
            return
        # Restore e <- u so discarded gradient mass is carried into the next step
        self.residual.copy_(self._tentative_u)
        self._tentative_residual = None
        self._tentative_u = None

    def norm(self) -> float:
        """Compute the L2 norm of the current residual vector."""
        if not self.enabled or self.size == 0:
            return 0.0
        return torch.linalg.vector_norm(self.residual).item()

    def reset(self) -> None:
        """Clear the residual to zero."""
        self.residual.zero_()
        self._tentative_residual = None
        self._tentative_u = None
