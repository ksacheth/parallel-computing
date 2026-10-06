# torch-asgc: Adaptive Staleness-Aware Gradient Compression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

An asynchronous distributed machine learning training library that addresses:
1. **High communication overhead** via gradient compression (Top-$K$ sparsification and QSGD quantization) with **error-feedback residual tracking**.
2. **Worker heterogeneity and stragglers** via bounded staleness attenuation ($w = 1 / (1 + \beta \tau)$) and dynamic staleness thresholds ($S_{max}$).
3. **Training stability** via an adaptive controller adjusting compression strength and staleness dynamically from streaming telemetry.

---

## 🚀 Quickstart: Public Interface (`torch-asgc`)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/asgc.git
cd asgc

# Install torch-asgc in editable mode
pip install -e .
```

### 1. Worker Training Script (`worker.py`)

Wrap any user-defined `nn.Module` and standard PyTorch optimizer (`AdamW`, `SGD`, etc.):

```python
import torch
import torch.nn as nn
from torch_asgc import DistributedModel, DistributedOptimizer, connect

# 1. Connect to the parameter server
client = connect("tcp://127.0.0.1:5555", worker_id="worker_0")

# 2. Define any model, criterion, and base optimizer
model = MyCustomModel().cuda()
criterion = nn.CrossEntropyLoss()
base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 3. Wrap with torch-asgc
model = DistributedModel(model, client=client)
optimizer = DistributedOptimizer(
    base_optimizer=base_optimizer,
    model=model,
    client=client,
    compression_mode="topk",  # "topk", "quantize", or "identity"
    rho=0.1,                  # Top-K active fraction
    error_feedback=True,      # Conserve residual gradient mass
)

# 4. Standard PyTorch training loop
for epoch in range(num_epochs):
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # Intercepts grads, applies error feedback, compresses,
        # pushes to server, and syncs updated weights
        optimizer.step(loss_val=loss.item())
```

### 2. Standalone Parameter Server (`server.py`)

Run the authoritative model parameter server:

```python
import torch
from torch_asgc import ParameterServer, StalenessConfig, AdaptiveConfig

model = MyCustomModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

server = ParameterServer(
    model=model,
    optimizer=optimizer,
    bind_address="tcp://0.0.0.0:5555",
    staleness_config=StalenessConfig(s_low=2, s_max=10, beta=0.5),
    compression_mode="topk",
    initial_strength=0.1,
    adaptive_config=AdaptiveConfig(enabled=True, interval_s=10.0),
)

# Start server loop
server.serve_forever()
```

---

## 🔬 Research Simulation Benchmark Suite

The repository also includes the deterministic simulation benchmark harness used to evaluate ASGC under controlled artificial worker delays.

### Running Unit Tests
```bash
# Run public library test suite
python -m pytest tests/test_torch_asgc -v

# Run full research benchmark verification suite
python -m pytest tests/ -v
```

### Running Experiments
```bash
# Smoke run on Fashion-MNIST
python scripts/run_experiment.py --config configs/smoke_fmnist.yaml

# Full 11-config x 3-seed evaluation matrix on CIFAR-10
python scripts/run_matrix.py --config configs/matrix_cifar_full.yaml

# Generate summary comparison figures and tables
python scripts/plot_results.py
```

---

## 📖 Architecture & Documentation

- [`docs/public_interface_gap_analysis.md`](docs/public_interface_gap_analysis.md): Technical gap analysis, mathematical proofs of error-feedback rollback, and full architectural specification.
- [`docs/methodology.tex`](docs/methodology.tex): Formal algorithm specification and equations.
- [`docs/proposal.tex`](docs/proposal.tex): Scope, motivation, and evaluation plan.
