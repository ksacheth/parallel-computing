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


def test_micro_local_steps_sync_frequency():
    torch.manual_seed(99)

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x)

    server_model = SmallNet()
    worker_model = SmallNet()
    worker_model.load_state_dict(server_model.state_dict())

    hub = LocalTransportHub()
    server_opt = torch.optim.SGD(server_model.parameters(), lr=0.05)
    server = ParameterServer(
        model=server_model,
        optimizer=server_opt,
        bind_address=hub,
        staleness_config=StalenessConfig(s_low=2, s_max=10, beta=0.5),
        compression_mode="identity",
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = connect(hub, worker_id="worker_local_steps")
    model = DistributedModel(worker_model, client=client)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.05)
    # Set local_steps = 3: communicate every 3rd step
    dist_opt = DistributedOptimizer(
        base_optimizer=base_opt,
        model=model,
        client=client,
        compression_mode="identity",
        local_steps=3,
    )

    try:
        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        criterion = nn.MSELoss()

        # Step 1 (local): should NOT increment server version
        dist_opt.zero_grad()
        loss1 = criterion(model(x), y)
        loss1.backward()
        dist_opt.step()
        assert server.version == 0

        # Step 2 (local): should NOT increment server version
        dist_opt.zero_grad()
        loss2 = criterion(model(x), y)
        loss2.backward()
        dist_opt.step()
        assert server.version == 0

        # Step 3 (sync boundary): pushes update to server, version becomes 1
        dist_opt.zero_grad()
        loss3 = criterion(model(x), y)
        loss3.backward()
        dist_opt.step()
        assert server.version == 1

        # Step 4 (local): server version remains 1
        dist_opt.zero_grad()
        loss4 = criterion(model(x), y)
        loss4.backward()
        dist_opt.step()
        assert server.version == 1

    finally:
        server.close()
