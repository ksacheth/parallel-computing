import threading
import time
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn

from torch_asgc import (
    DistributedModel,
    DistributedOptimizer,
    LocalTransportHub,
    ParameterServer,
    StalenessConfig,
    connect,
)
from torch_asgc.plugins import (
    ASGCStrategy,
    ASGCTrainer,
    ASGCTrainerCallback,
    wrap_hf_trainer,
)
from torch_asgc.plugins.hf import _TRANSFORMERS_AVAILABLE
from torch_asgc.plugins.lightning import _LIGHTNING_AVAILABLE


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1, bias=False)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def running_server_and_client():
    server_model = SimpleModel()
    worker_model = SimpleModel()
    worker_model.load_state_dict(server_model.state_dict())

    hub = LocalTransportHub()
    server_opt = torch.optim.SGD(server_model.parameters(), lr=0.1)
    server = ParameterServer(
        model=server_model,
        optimizer=server_opt,
        bind_address=hub,
        staleness_config=StalenessConfig(s_low=2, s_max=10, beta=0.5),
        compression_mode="topk",
        initial_strength=0.5,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = connect(hub, worker_id="worker_plugin_test")
    yield server, client, worker_model
    server.close()


def test_lightning_strategy_import_error(running_server_and_client):
    _, client, _ = running_server_and_client
    if not _LIGHTNING_AVAILABLE:
        with pytest.raises(ImportError, match="PyTorch Lightning is required"):
            ASGCStrategy(client=client)


def test_lightning_strategy_setup_and_optimizer_step(running_server_and_client):
    server, client, worker_model = running_server_and_client

    strategy = ASGCStrategy(
        client=client,
        compression_mode="topk",
        rho=0.5,
        _test_mode=True,
    )

    # 1. Setup model
    wrapped_model = strategy.setup_model(worker_model)
    assert isinstance(wrapped_model, DistributedModel)

    # 2. Setup optimizers on mock trainer
    base_opt = torch.optim.SGD(wrapped_model.parameters(), lr=0.1)
    mock_trainer = SimpleNamespace(
        lightning_module=wrapped_model,
        optimizers=[base_opt],
        callback_metrics={"train_loss": torch.tensor(0.777)},
    )
    optimizers = strategy.setup_optimizers(mock_trainer)
    assert len(optimizers) == 1
    dist_opt = optimizers[0]
    assert isinstance(dist_opt, DistributedOptimizer)
    assert mock_trainer.optimizers[0] is dist_opt

    # 3. Test optimizer_step execution through strategy
    x = torch.randn(2, 4)
    loss = wrapped_model(x).sum()
    loss.backward()

    initial_version = server.version
    strategy.optimizer_step(dist_opt)
    dist_opt.zero_grad()

    # Allow server to process push
    time.sleep(0.1)
    assert server.version > initial_version
    assert len(server.telemetry.losses) >= 1
    assert server.telemetry.losses[-1] == pytest.approx(0.777, abs=1e-5)


def test_hf_trainer_import_error(running_server_and_client):
    _, client, worker_model = running_server_and_client
    if not _TRANSFORMERS_AVAILABLE:
        with pytest.raises(ImportError, match="Hugging Face transformers is required"):
            ASGCTrainer(client=client, model=worker_model)


def test_hf_trainer_callback_telemetry(running_server_and_client):
    _, client, _ = running_server_and_client
    callback = ASGCTrainerCallback(client=client)

    state = SimpleNamespace(log_history=[{"step": 1, "loss": 0.354}])
    callback.on_step_end(args=None, state=state, control=None)

    assert callback.last_loss == pytest.approx(0.354)


def test_hf_trainer_create_optimizer_and_step(running_server_and_client):
    server, client, worker_model = running_server_and_client

    trainer = ASGCTrainer(
        model=worker_model,
        client=client,
        compression_mode="topk",
        rho=0.5,
        _test_mode=True,
    )

    # Verify callback is registered
    assert any(isinstance(cb, ASGCTrainerCallback) for cb in trainer.callbacks)

    # Verify create_optimizer wraps with DistributedOptimizer
    opt = trainer.create_optimizer()
    assert isinstance(opt, DistributedOptimizer)

    # Verify optimizer step works end-to-end
    x = torch.randn(2, 4)
    loss = worker_model(x).sum()
    loss.backward()

    initial_version = server.version
    opt.step()
    opt.zero_grad()

    time.sleep(0.1)
    assert server.version > initial_version


def test_wrap_hf_trainer_helper(running_server_and_client):
    _, client, worker_model = running_server_and_client

    mock_trainer = SimpleNamespace(
        model=worker_model,
        optimizer=torch.optim.SGD(worker_model.parameters(), lr=0.01),
    )

    wrapped = wrap_hf_trainer(mock_trainer, client=client, compression_mode="topk", rho=0.5)
    assert isinstance(wrapped.optimizer, DistributedOptimizer)
