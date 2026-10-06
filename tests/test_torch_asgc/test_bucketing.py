import torch
import torch.nn as nn
from torch_asgc.engine.bucketing import GradientBucket, create_buckets


def test_bucket_flatten_and_unflatten():
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 3)
            self.fc2 = nn.Linear(3, 2)
            self.frozen = nn.Parameter(torch.ones(5), requires_grad=False)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    net = SimpleNet()
    buckets = create_buckets(net, bucket_cap_mb=1.0)
    assert len(buckets) == 1
    bucket = buckets[0]

    # Expected numel: fc1.weight (12) + fc1.bias (3) + fc2.weight (6) + fc2.bias (2) = 23
    assert bucket.total_numel == 23

    # Generate gradients
    x = torch.randn(2, 4)
    loss = net(x).sum()
    loss.backward()

    params_dict = dict(net.named_parameters())
    flat = bucket.flatten(params_dict)
    assert flat.shape == (23,)
    assert flat.dtype == torch.float32

    # Unflatten and compare values
    unflattened = bucket.unflatten_to_dict(flat)
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert torch.allclose(unflattened[name], p.grad)
