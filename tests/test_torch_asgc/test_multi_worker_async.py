import threading
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


def test_two_workers_concurrent_asynchronous_training():
    torch.manual_seed(123)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x)

    server_model = MLP()
    hub = LocalTransportHub()
    server_optimizer = torch.optim.AdamW(server_model.parameters(), lr=0.05)
    server = ParameterServer(
        model=server_model,
        optimizer=server_optimizer,
        bind_address=hub,
        staleness_config=StalenessConfig(s_low=1, s_max=10, beta=0.2),
        compression_mode="topk",
        initial_strength=0.5,
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    true_w = torch.tensor([[1.5], [-2.0], [0.5], [1.0]])
    criterion = nn.MSELoss()

    def run_worker(worker_id: str, steps: int):
        worker_model = MLP()
        client = connect(hub, worker_id=worker_id)
        model = DistributedModel(worker_model, client=client)
        base_opt = torch.optim.AdamW(model.parameters(), lr=0.05)
        dist_opt = DistributedOptimizer(
            base_optimizer=base_opt,
            model=model,
            client=client,
            compression_mode="topk",
            rho=0.5,
            error_feedback=True,
        )

        for _ in range(steps):
            x = torch.randn(8, 4)
            y = x @ true_w
            dist_opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            dist_opt.step(loss_val=loss.item())

    try:
        t1 = threading.Thread(target=run_worker, args=("worker_A", 10))
        t2 = threading.Thread(target=run_worker, args=("worker_B", 10))
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert not t1.is_alive()
        assert not t2.is_alive()
        # Each worker submitted 10 updates -> 20 updates processed
        assert server.version == 20
    finally:
        server.close()
