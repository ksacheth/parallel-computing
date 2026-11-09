"""Workload definitions shared by ps_server.py and ps_worker.py.

The parameter server and every remote worker must build the *same* model
architecture, so both scripts resolve a task name through this one registry.
Datasets and models come from the research harness (``src/asgc``) so the
multi-machine runs train exactly the workloads behind findings F1-F17.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asgc.data import _load as load_datasets  # noqa: E402
from asgc.models.cnn_mnist import build_model  # noqa: E402


@dataclass(frozen=True)
class Task:
    dataset: str
    model: str
    lr: float
    batch_size: int


TASKS = {
    "fmnist": Task(dataset="fashion_mnist", model="small_cnn", lr=0.05, batch_size=128),
    "cifar10": Task(dataset="cifar10", model="resnet18", lr=0.02, batch_size=128),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]


def make_model(task: Task, seed: int = 0) -> nn.Module:
    """Seeded so a restarted server rebuilds the identical initial weights."""
    torch.manual_seed(seed)
    return build_model(task.model)


def train_loader(
    task: Task, data_root: str, rank: int, num_shards: int, batch_size: int, num_workers: int = 2
) -> DataLoader:
    """Worker shard: every ``num_shards``-th example starting at ``rank`` (disjoint across ranks)."""
    train, _ = load_datasets(task.dataset, data_root)
    indices = list(range(rank, len(train), num_shards))
    return DataLoader(
        Subset(train, indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def eval_tensors(task: Task, data_root: str, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tensorized test subset (and nothing else) so repeated evals skip PIL decoding."""
    _, test = load_datasets(task.dataset, data_root)
    n = min(max_samples, len(test)) if max_samples > 0 else len(test)
    xs, ys = zip(*(test[i] for i in range(n)))
    return torch.stack(xs), torch.tensor(ys)


def bn_calibration_tensors(task: Task, data_root: str, samples: int) -> torch.Tensor | None:
    """Training images for re-estimating BatchNorm stats (the server never runs train forwards)."""
    model = build_model(task.model)
    if not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        return None
    train, _ = load_datasets(task.dataset, data_root)
    generator = torch.Generator().manual_seed(1234)
    picks = torch.randperm(len(train), generator=generator)[:samples].tolist()
    return torch.stack([train[i][0] for i in picks])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    bn_x: torch.Tensor | None,
    batch_size: int = 500,
) -> tuple[float, float]:
    """Return (accuracy, mean cross-entropy). Recalibrates BN running stats first."""
    if bn_x is not None:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.reset_running_stats()
                m.momentum = None  # cumulative average over the calibration sweep
        model.train()
        for chunk in bn_x.split(batch_size):
            model(chunk)
    model.eval()
    eval_set = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    loss_sum, correct = 0.0, 0
    for xb, yb in eval_set:
        logits = model(xb)
        loss_sum += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return correct / len(y), loss_sum / len(y)
