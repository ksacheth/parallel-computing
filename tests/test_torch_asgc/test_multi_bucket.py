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


def test_multi_bucket_increments_version_exactly_once_per_step():
    torch.manual_seed(42)

    class MultiLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 6)
            self.fc2 = nn.Linear(6, 4)
            self.fc3 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    server_model = MultiLayerNet()
    worker_model = MultiLayerNet()
    worker_model.load_state_dict(server_model.state_dict())

    # Set very small bucket cap (0.0001 MB ~ 100 bytes) so each layer forms its own bucket
    bucket_cap_mb = 0.0001

    hub = LocalTransportHub()
    server_opt = torch.optim.SGD(server_model.parameters(), lr=0.05)
    server = ParameterServer(
        model=server_model,
        optimizer=server_opt,
        bind_address=hub,
        staleness_config=StalenessConfig(s_low=2, s_max=10, beta=0.5),
        compression_mode="identity",
        bucket_cap_mb=bucket_cap_mb,
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = connect(hub, worker_id="worker_multi_bucket")
    model = DistributedModel(worker_model, client=client)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.05)
    dist_opt = DistributedOptimizer(
        base_optimizer=base_opt,
        model=model,
        client=client,
        compression_mode="identity",
        bucket_cap_mb=bucket_cap_mb,
    )

    # Confirm that the model is indeed partitioned into multiple buckets
    num_buckets = len(dist_opt.buckets)
    assert num_buckets >= 3, f"Expected at least 3 buckets, got {num_buckets}"

    try:
        x = torch.randn(4, 8)
        y = torch.randint(0, 2, (4,))
        criterion = nn.CrossEntropyLoss()

        # Run 5 training steps
        for step in range(5):
            dist_opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            dist_opt.step(loss_val=loss.item())

        # CRUCIAL ASSERTION:
        # Exactly 5 training steps were executed, so server version MUST be 5,
        # NOT 5 * num_buckets (which would have been 15+ before the fix)!
        assert server.version == 5, f"Expected server version 5, got {server.version}"
        assert dist_opt.last_sync_version == 5

        # All 5 updates should be accepted with zero rejections
        assert server.telemetry.total_updates == 5
        assert server.telemetry.rejected_count == 0

    finally:
        server.close()
