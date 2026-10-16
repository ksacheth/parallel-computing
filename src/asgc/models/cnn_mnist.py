"""Workload models. ResNet-18 for CIFAR-10 lands in stage 6."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    """Two-conv CNN (~55k parameters) so per-step full-model fetches stay cheap."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    if name == "resnet18":
        from asgc.models.resnet_cifar import ResNet18

        return ResNet18()
    raise ValueError(f"unknown model: {name!r}")
