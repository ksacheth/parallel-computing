"""Workload definitions shared by ps_server.py and ps_worker.py.

The parameter server and every remote worker must build the *same* model
architecture, so both scripts resolve a task name through this one registry.

- Image tasks reuse the research harness datasets and models (``src/asgc``),
  so multi-machine runs train the workloads behind findings F1-F13.
- Text tasks train a character-level GPT on Tiny Shakespeare. ``gpt-mini``
  matches the transformer notebooks (F14-F17); ``gpt`` is the larger nanoGPT
  Shakespeare-char configuration (~10.7M parameters).
"""
from __future__ import annotations

import sys
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asgc.data import _load as load_datasets  # noqa: E402
from asgc.models.cnn_mnist import build_model  # noqa: E402
from asgc.models.gpt import GPT, GPTConfig  # noqa: E402

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_CHARS = 1_115_394  # exact length; every machine must tokenize the identical corpus
TRAIN_FRACTION = 0.9


@dataclass(frozen=True)
class Task:
    dataset: str
    model: str
    lr: float
    batch_size: int
    optimizer: str = "sgd"  # server-side optimizer: sgd | adamw
    weight_decay: float = 0.0
    max_grad_norm: float | None = None  # worker-side clipping default
    gpt: GPTConfig | None = None  # set for text tasks

    @property
    def is_text(self) -> bool:
        return self.gpt is not None


TASKS = {
    "fmnist": Task(dataset="fashion_mnist", model="small_cnn", lr=0.05, batch_size=128),
    "cifar10": Task(dataset="cifar10", model="resnet18", lr=0.02, batch_size=128),
    "gpt-mini": Task(
        dataset="shakespeare_char", model="gpt", lr=5e-4, batch_size=32, optimizer="adamw",
        weight_decay=0.1, max_grad_norm=1.0,
        gpt=GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=192, dropout=0.1),
    ),
    "gpt": Task(
        dataset="shakespeare_char", model="gpt", lr=1e-3, batch_size=64, optimizer="adamw",
        weight_decay=0.1, max_grad_norm=1.0,
        gpt=GPTConfig(vocab_size=65, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name]


def make_model(task: Task, seed: int = 0) -> nn.Module:
    """Seeded so a restarted server rebuilds the identical initial weights."""
    torch.manual_seed(seed)
    if task.is_text:
        return GPT(task.gpt)
    return build_model(task.model)


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for both (batch, classes) image logits and (batch, seq, vocab) token logits."""
    return nn.functional.cross_entropy(logits.flatten(0, -2), targets.flatten())


# ---------------------------------------------------------------- text data

def shakespeare_ids(data_root: str) -> torch.Tensor:
    """Character ids of Tiny Shakespeare (downloaded once, verified by length)."""
    path = Path(data_root) / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SHAKESPEARE_URL, timeout=60) as response:
            path.write_bytes(response.read())
    text = path.read_text(encoding="utf-8")
    if len(text) != SHAKESPEARE_CHARS:
        raise SystemExit(f"{path} has {len(text)} chars, expected {SHAKESPEARE_CHARS}; delete it and retry")
    stoi = {ch: i for i, ch in enumerate(sorted(set(text)))}
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def _split(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cut = int(len(ids) * TRAIN_FRACTION)
    return ids[:cut], ids[cut:]


def _text_batches(ids: torch.Tensor, block: int, batch_size: int, seed: int) -> Iterator:
    generator = torch.Generator().manual_seed(seed)
    while True:
        starts = torch.randint(0, len(ids) - block - 1, (batch_size,), generator=generator)
        windows = torch.stack([ids[s : s + block + 1] for s in starts.tolist()])
        yield windows[:, :-1], windows[:, 1:]


def _endless(loader: DataLoader) -> Iterator:
    while True:  # reshuffles on every pass
        yield from loader


def train_batches(
    task: Task, data_root: str, shard: tuple[int, int], batch_size: int, num_workers: int = 2
) -> tuple[Iterator, str]:
    """Endless (x, y) stream over this worker's disjoint shard, plus a description for the log.

    ``shard`` is ``(rank, num_shards)``. Text shards are contiguous slices of the
    training split; image shards take every ``num_shards``-th example from ``rank``.
    """
    rank, num_shards = shard
    if task.is_text:
        train, _ = _split(shakespeare_ids(data_root))
        size = len(train) // num_shards
        part = train[rank * size : (rank + 1) * size]
        stream = _text_batches(part, task.gpt.block_size, batch_size, seed=1000 + rank)
        return stream, f"shard {rank}/{num_shards}: {len(part):,} training characters"
    train, _ = load_datasets(task.dataset, data_root)
    loader = DataLoader(
        Subset(train, list(range(rank, len(train), num_shards))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return _endless(loader), f"shard {rank}/{num_shards}: {len(loader)} batches per pass"


# ---------------------------------------------------------------- evaluation

def eval_tensors(task: Task, data_root: str, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed evaluation set: test images, or non-overlapping validation windows for text."""
    if task.is_text:
        _, val = _split(shakespeare_ids(data_root))
        block = task.gpt.block_size
        windows = val[: (len(val) - 1) // block * block + 1]
        x = windows[:-1].view(-1, block)
        y = windows[1:].view(-1, block)
        n = min(max_samples, len(x)) if max_samples > 0 else len(x)
        return x[:n], y[:n]
    _, test = load_datasets(task.dataset, data_root)
    n = min(max_samples, len(test)) if max_samples > 0 else len(test)
    xs, ys = zip(*(test[i] for i in range(n)))
    return torch.stack(xs), torch.tensor(ys)


def bn_calibration_tensors(task: Task, data_root: str, samples: int) -> torch.Tensor | None:
    """Training images for re-estimating BatchNorm stats (the server never runs train forwards)."""
    model = make_model(task)
    if not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        return None
    train, _ = load_datasets(task.dataset, data_root)
    generator = torch.Generator().manual_seed(1234)
    picks = torch.randperm(len(train), generator=generator)[:samples].tolist()
    return torch.stack([train[i][0] for i in picks])


def _recalibrate_bn(model: nn.Module, bn_x: torch.Tensor, batch_size: int) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over the calibration sweep
    model.train()
    for chunk in bn_x.split(batch_size):
        model(chunk)


@torch.no_grad()
def evaluate(model: nn.Module, eval_set: tuple[torch.Tensor, torch.Tensor], bn_x: torch.Tensor | None) -> tuple[float, float]:
    """Return (top-1 accuracy, mean cross-entropy); per token for text tasks."""
    if bn_x is not None:
        _recalibrate_bn(model, bn_x, batch_size=500)
    model.eval()
    x, y = eval_set
    batch_size = 500 if x.dim() > 2 else 16
    loss_sum, correct, count = 0.0, 0, 0
    for xb, yb in DataLoader(TensorDataset(x, y), batch_size=batch_size):
        logits = model(xb).flatten(0, -2)
        targets = yb.flatten()
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += targets.numel()
    return correct / count, loss_sum / count
