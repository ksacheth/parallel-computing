"""Dataset loading and mini-batch partitioning across workers."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from asgc.config import ExperimentConfig

_NORMALIZATION = {
    "fashion_mnist": ((0.2860,), (0.3530,)),
}


def _transform(dataset: str) -> transforms.Compose:
    mean, std = _NORMALIZATION[dataset]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _load(dataset: str, root: str) -> tuple[Any, Any]:
    if dataset == "fashion_mnist":
        train = datasets.FashionMNIST(root, train=True, download=True, transform=_transform(dataset))
        test = datasets.FashionMNIST(root, train=False, download=True, transform=_transform(dataset))
        return train, test
    raise ValueError(f"unknown dataset: {dataset!r}")


def prepare_datasets(config: ExperimentConfig) -> None:
    """Force the download once from the parent so spawned children never race on it."""
    _load(config.workload.dataset, config.run.data_root)


def worker_loader(config: ExperimentConfig, worker_id: int) -> DataLoader:
    """Worker-owned partition: every num_workers-th training example, shuffled."""
    train, _ = _load(config.workload.dataset, config.run.data_root)
    indices = list(range(worker_id, len(train), config.workload.num_workers))
    return DataLoader(Subset(train, indices), batch_size=config.workload.batch_size, shuffle=True)


def test_loader(config: ExperimentConfig) -> DataLoader:
    """Evaluation loader over a one-time tensorized test set, so repeated evals
    never pay the PIL-decode transform cost (the server evals every interval)."""
    _, test = _load(config.workload.dataset, config.run.data_root)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in DataLoader(test, batch_size=1024, shuffle=False):
        xs.append(x)
        ys.append(y)
    tensorized = TensorDataset(torch.cat(xs), torch.cat(ys))
    return DataLoader(tensorized, batch_size=512, shuffle=False)


def cycle(loader: DataLoader) -> Iterator:
    """Endless batch stream; shuffle=True re-shuffles on every pass."""
    while True:
        yield from loader
