import threading
import time
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


def test_public_api_e2e_training_convergence():
    torch.manual_seed(42)

    # 1. Define arbitrary model and loss
    class RegressionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1, bias=False)

        def forward(self, x):
            return self.linear(x)

    server_model = RegressionNet()
    worker_model = RegressionNet()

    # Align initial weights
    worker_model.load_state_dict(server_model.state_dict())

    # 2. Setup transport and server
    hub = LocalTransportHub()
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=0.05)
    server = ParameterServer(
        model=server_model,
        optimizer=server_optimizer,
        bind_address=hub,
        staleness_config=StalenessConfig(s_low=2, s_max=10, beta=0.5),
        compression_mode="topk",
        initial_strength=0.5,
    )

    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # 3. Setup worker with public DistributedOptimizer
    client = connect(hub, worker_id="worker_0")
    model = DistributedModel(worker_model, client=client)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    dist_optimizer = DistributedOptimizer(
        base_optimizer=base_optimizer,
        model=model,
        client=client,
        compression_mode="topk",
        rho=0.5,
        error_feedback=True,
    )

    # Synthetic target: y = 2 * x_0 + 3 * x_1
    true_w = torch.tensor([[2.0], [3.0], [0.0], [0.0]])
    criterion = nn.MSELoss()

    initial_loss = None
    final_loss = None

    # Run 15 training steps
    try:
        for step in range(15):
            x = torch.randn(16, 4)
            y = x @ true_w

            dist_optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            if step == 0:
                initial_loss = loss.item()

            dist_optimizer.step(loss_val=loss.item())
            final_loss = loss.item()

        assert initial_loss is not None and final_loss is not None
        # Training loss should drop over 15 steps
        assert final_loss < initial_loss
        assert server.version == 15
    finally:
        server.close()


def test_default_worker_id_and_lr_scheduler_compatibility():
    model_s = nn.Linear(4, 1)
    model_w = nn.Linear(4, 1)
    model_w.load_state_dict(model_s.state_dict())

    hub = LocalTransportHub()
    server = ParameterServer(
        model=model_s,
        optimizer=torch.optim.SGD(model_s.parameters(), lr=0.1),
        bind_address=hub,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        # worker_id=None should auto-generate matching ID across client & transport
        client = connect(hub)
        dist_model = DistributedModel(model_w, client=client)
        base_opt = torch.optim.SGD(dist_model.parameters(), lr=0.1)
        dist_opt = DistributedOptimizer(
            base_optimizer=base_opt,
            model=dist_model,
            client=client,
            max_grad_norm=1.0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(dist_opt, step_size=1, gamma=0.5)

        x = torch.randn(4, 4)
        loss = dist_model(x).sum()
        loss.backward()
        dist_opt.step()
        scheduler.step()

        assert dist_opt.param_groups[0]["lr"] == 0.05
        assert server.version == 1
        assert isinstance(dist_opt.state_dict(), dict)
    finally:
        server.close()


def test_adamw_staleness_downweighting_scales_update_step():
    from torch_asgc.engine.optimizer import StatefulServerOptimizer

    torch.manual_seed(7)
    m_full = nn.Linear(4, 2, bias=False)
    m_down = nn.Linear(4, 2, bias=False)
    m_down.load_state_dict(m_full.state_dict())
    init_w = m_full.weight.detach().clone()

    eng_full = StatefulServerOptimizer(
        torch.optim.AdamW(m_full.parameters(), lr=0.01, weight_decay=0.0),
        list(m_full.named_parameters()),
    )
    eng_down = StatefulServerOptimizer(
        torch.optim.AdamW(m_down.parameters(), lr=0.01, weight_decay=0.0),
        list(m_down.named_parameters()),
    )

    grad = {"weight": torch.randn_like(init_w)}
    eng_full.step_with_gradients(grad, weight=1.0)
    eng_down.step_with_gradients(grad, weight=0.25)

    delta_full = m_full.weight.detach() - init_w
    delta_down = m_down.weight.detach() - init_w
    assert torch.allclose(delta_down, 0.25 * delta_full, atol=1e-6)


