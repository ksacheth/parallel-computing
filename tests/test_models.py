import torch

from asgc import data as data_mod
from asgc.models.cnn_mnist import build_model


def test_small_cnn_is_small():
    model = build_model("small_cnn")
    assert sum(p.numel() for p in model.parameters()) < 100_000


def test_resnet18_forward_shape_and_size():
    model = build_model("resnet18")
    n_params = sum(p.numel() for p in model.parameters())
    assert 11_000_000 < n_params < 11_500_000  # the standard CIFAR ResNet-18
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_cifar_loaders_partition_and_normalize():
    config = load_smoke_config()
    train_batch = next(iter(data_mod.worker_loader(config, worker_id=0)))
    x, y = train_batch
    assert x.shape == (4, 3, 32, 32) and y.shape == (4,)
    loader = data_mod.test_loader(config)
    xt, yt = next(iter(loader))
    assert xt.shape[1:] == (3, 32, 32)
    assert xt.min() >= -4.0 and xt.max() <= 4.0  # normalized, not raw [0, 255]


def load_smoke_config():
    from asgc.config import ExperimentConfig, RunConfig, WorkloadConfig

    return ExperimentConfig(
        run=RunConfig(name="unit_cifar", data_root="data"),
        workload=WorkloadConfig(dataset="cifar10", model="resnet18", batch_size=4, num_workers=3),
    )
