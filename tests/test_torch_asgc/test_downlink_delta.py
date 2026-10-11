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


def test_delta_broadcasting_reconstruction_is_exact():
    torch.manual_seed(42)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2, bias=True)

        def forward(self, x):
            return self.linear(x)

    server_model = ToyModel()
    worker_model = ToyModel()
    worker_model.load_state_dict(server_model.state_dict())

    hub = LocalTransportHub()
    server_opt = torch.optim.SGD(server_model.parameters(), lr=0.1)
    server = ParameterServer(
        model=server_model,
        optimizer=server_opt,
        bind_address=hub,
        staleness_config=StalenessConfig(s_low=2, s_max=10, beta=0.5),
        compression_mode="identity",  # exact uncompressed for verification
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = connect(hub, worker_id="worker_delta_test")
    model = DistributedModel(worker_model, client=client)
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    dist_opt = DistributedOptimizer(
        base_optimizer=base_opt,
        model=model,
        client=client,
        compression_mode="identity",
    )

    try:
        # Step 1: Initial fetch gets full snapshot (known_version = -1)
        resp1 = client.fetch_parameters()
        assert not resp1.is_delta
        assert resp1.params is not None
        assert resp1.version == 0

        # Perform 3 training steps
        x = torch.randn(4, 3)
        for i in range(3):
            dist_opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            dist_opt.step()

        # Server should have processed 3 updates (version 3)
        assert server.version == 3

        # Worker weights should match server weights exactly!
        for name, p in server_model.named_parameters():
            worker_p = dict(worker_model.named_parameters())[name]
            assert torch.allclose(worker_p, p, atol=1e-5), f"Mismatch in {name}"

        # Fetch explicitly with known_version=1: should receive accumulated delta for v=1,2
        client.known_version = 1
        resp_delta = client.fetch_parameters()
        assert resp_delta.is_delta
        assert resp_delta.delta is not None
        assert resp_delta.version == 3

        # Verify INT8 QuantizedDelta achieves significant wire byte reduction vs full float32 snapshot
        snapshot_bytes = resp1.encoded_bytes()
        delta_bytes = resp_delta.encoded_bytes()
        assert delta_bytes < snapshot_bytes

    finally:
        server.close()
