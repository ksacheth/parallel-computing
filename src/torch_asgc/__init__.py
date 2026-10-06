"""torch-asgc: Adaptive Staleness-Aware Gradient Compression for Distributed PyTorch.

Public API:
- DistributedOptimizer: Wrap standard PyTorch optimizers (AdamW, SGD, etc.).
- DistributedModel: Wrap torch.nn.Module.
- ParameterServer: Authoritative model parameter server.
- init_client / connect: Connect worker to a parameter server.
- StalenessConfig, AdaptiveConfig: Configuration classes.
"""
from __future__ import annotations

from torch_asgc.client import ASGCClient, connect, connect as init_client
from torch_asgc.compression import make_compressor
from torch_asgc.controller import AdaptiveConfig
from torch_asgc.model import DistributedModel
from torch_asgc.optim import DistributedOptimizer
from torch_asgc.server import ParameterServer
from torch_asgc.staleness import StalenessConfig
from torch_asgc.transport.local import LocalTransportHub

__version__ = "0.1.0"

__all__ = [
    "DistributedOptimizer",
    "DistributedModel",
    "ParameterServer",
    "ASGCClient",
    "init_client",
    "connect",
    "StalenessConfig",
    "AdaptiveConfig",
    "LocalTransportHub",
    "make_compressor",
]
