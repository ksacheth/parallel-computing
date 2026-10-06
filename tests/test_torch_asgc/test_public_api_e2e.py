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
