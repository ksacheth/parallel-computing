"""Distributed model wrapper for ASGC."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from torch_asgc.client import ASGCClient


class DistributedModel(nn.Module):
    """Wraps an arbitrary torch.nn.Module to integrate with ASGC training.

    Preserves standard forward execution, module introspection, and attribute access.
    Delegates state_dict and load_state_dict directly to the underlying module
    to avoid 'module.' key prefix mismatches.
    """

    def __init__(self, module: nn.Module, client: ASGCClient | None = None) -> None:
        super().__init__()
        self.module = module
        self.client = client

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def named_parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args: Any, **kwargs: Any) -> Any:
        return self.module.parameters(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False) -> Any:
        return self.module.load_state_dict(state_dict, strict=strict, assign=assign)

    def __getattr__(self, name: str) -> Any:
        # Proxy attribute lookup to the underlying module if not present on wrapper
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
