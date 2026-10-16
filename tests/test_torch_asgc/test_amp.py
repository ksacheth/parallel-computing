import threading
import torch
import torch.nn as nn
from torch_asgc import (
    DistributedModel,
    DistributedOptimizer,
    LocalTransportHub,
    ParameterServer,
    connect,
)


def test_grad_scaler_step_unscales_gradients():
    torch.manual_seed(42)
    server_model = nn.Linear(4, 2)
    worker_model = nn.Linear(4, 2)
    worker_model.load_state_dict(server_model.state_dict())

    hub = LocalTransportHub()
    server = ParameterServer(
        model=server_model,
        optimizer=torch.optim.SGD(server_model.parameters(), lr=0.1),
        bind_address=hub,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = connect(hub, worker_id="worker_amp")
    model = DistributedModel(worker_model, client=client)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dist_opt = DistributedOptimizer(
        base_optimizer=base_opt,
        model=model,
        client=client,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        scaler = torch.amp.GradScaler(device, init_scale=1024.0, enabled=True)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=False)

    try:
        x = torch.randn(4, 4)
        loss = model(x).sum()

        if scaler.is_enabled():
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()

            scaled_weight_grad = model.weight.grad.clone()

            # Pass scaler to dist_opt.step
            dist_opt.step(loss_val=loss.item(), scaler=scaler)
            scaler.update()

            # Verify gradients were properly unscaled by 1024 before compression
            expected_unscaled = scaled_weight_grad / 1024.0
            assert torch.allclose(model.weight.grad, expected_unscaled, atol=1e-5)
        else:
            loss.backward()
            dist_opt.step(loss_val=loss.item())

        assert server.version == 1
    finally:
        server.close()


def test_grad_scaler_inf_nan_overflow_is_safely_skipped():
    torch.manual_seed(42)
    server_model = nn.Linear(4, 2)
    worker_model = nn.Linear(4, 2)
    worker_model.load_state_dict(server_model.state_dict())

    hub = LocalTransportHub()
    server = ParameterServer(
        model=server_model,
        optimizer=torch.optim.SGD(server_model.parameters(), lr=0.1),
        bind_address=hub,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = connect(hub, worker_id="worker_overflow")
    model = DistributedModel(worker_model, client=client)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dist_opt = DistributedOptimizer(
        base_optimizer=base_opt,
        model=model,
        client=client,
    )

    try:
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()

        # Inject an Inf/NaN gradient simulating an AMP half-precision overflow
        with torch.no_grad():
            model.weight.grad[0, 0] = float("inf")

        # Optimizer step should detect Inf/NaN and skip without corrupting server
        result = dist_opt.step(loss_val=loss.item())
        assert result is None

        # Server model must NOT have been updated (version remains 0)
        assert server.version == 0
        assert server.telemetry.total_updates == 0
    finally:
        server.close()
