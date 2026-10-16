# Architectural Gap Analysis & Public Interface Design for ASGC

This document outlines the technical audit, architectural gaps, and engineering roadmap required to transform the **Adaptive Staleness-Aware Gradient Compression (ASGC)** research prototype into an open, production-ready PyTorch distributed training library (`torch-asgc`).

---

## 1. Executive Summary

The current repository provides a working research prototype for **Adaptive Staleness-Aware Gradient Compression for Asynchronous Distributed Training**. It demonstrates key algorithmic concepts:
- Error-feedback residual tracking ($u = g + e \implies e \leftarrow u - \hat{g}$)
- Top-$K$ and QSGD quantization with exact wire byte accounting
- Bounded-staleness downweighting ($w = 1 / (1 + \beta\tau)$) and rejection ($\tau > S_{max}$)
- Runtime adaptation of compression ratio and staleness thresholds via streaming telemetry

However, the prototype was constructed under strict simulation constraints:
1. **Single-host simulated multiprocessing**: Relies exclusively on `multiprocessing.Queue` and sleep-injected delays (`src/asgc/transport.py`, `src/asgc/delays.py`).
2. **Inverted, monolithic control flow**: The library owns the training loop, model initialization, data loading, and loss calculation (`src/asgc/worker.py#L25-86`, `src/asgc/server.py#L79-333`).
3. **Downlink/uplink communication asymmetry**: Uplink gradients are aggressively compressed, but downlink model parameters are transmitted as full, uncompressed FP32 `state_dict` copies (`src/asgc/server.py#L155`).
4. **Naive SGD parameter updates**: Updates are applied as raw in-place additions `p.add_(-lr * g)` (`src/asgc/server.py#L276`), ignoring modern optimizer mechanics (momentum, AdamW second-order moments, decoupled weight decay, learning rate schedulers).
5. **Coupled evaluation inside the parameter server**: The server synchronously runs BatchNorm recalibration and test set evaluation (`src/asgc/server.py#L44-76`), halting worker processing.
6. **Error-feedback residual leak under update rejection**: Discarded gradients are subtracted from the worker's residual *before* the server decides whether to reject the update (`src/asgc/worker.py#L74`), permanently discarding unapplied gradient mass when an update is rejected.

---

## 2. Codebase Audit: Current Limitations by Module

### 2.1 `src/asgc/worker.py`
- **Thread Oversubscription Clamp (Line 26)**: `torch.set_num_threads(1)` is hardcoded. While appropriate for multi-process CPU simulation on a single machine, it cripples multi-core CPU training and worker-side data augmentation in production.
- **Hardcoded Model Construction (Lines 28–29)**: `model = build_model(config.workload.model).to(device)` limits execution to factory models (`SmallCNN`, `ResNet18`). A public library must accept any arbitrary user-defined `nn.Module`.
- **Hardcoded DataLoader & Data Ingestion (Lines 33, 56–57)**: `batches = cycle(worker_loader(config, worker_id))` assumes data is globally known and yields standard `(x, y)` tuples. It cannot handle dictionary batches (e.g. Hugging Face `{"input_ids", "attention_mask", "labels"}`), custom datasets, or streaming inputs.
- **Hardcoded Loss Function (Line 34)**: `criterion = nn.CrossEntropyLoss()` precludes regression, contrastive learning, reinforcement learning, or multi-task objectives.
- **Parameter Traversal & Residual Allocation (Lines 37–38, 63)**:
  - Allocates residuals for *all* parameters indiscriminately, including frozen layers (`requires_grad=False`).
  - Assumes `model.parameters()` iteration order is strictly identical and deterministic across all processes. For models with tied weights, duplicates lead to redundant residual allocations and indexing errors.
- **Synchronous Blocking Fetch (Lines 41–55)**:
  - Workers block on `transport.fetch()`. Full state loading `model.load_state_dict(response.params)` incurs CPU-GPU transfer bottlenecks and unpickling overhead on every step.
- **Local Decode & Premature Residual Update (Lines 73–75)**:
  - Worker updates $e \leftarrow u - \hat{g}$ *before* transmission without confirming whether the server accepted or rejected the update.
- **Lack of Modern Training Primitives (Lines 60–62)**: No gradient accumulation steps, no `torch.cuda.amp.autocast()`, no `GradScaler`, and no gradient clipping.

### 2.2 `src/asgc/server.py`
- **Tightly Coupled Server Responsibilities (Lines 83–88)**: The parameter server directly instantiates the model, evaluation loader, and worker 0's training loader. In production, parameter servers should function strictly as key-value parameter/gradient servers.
- **Synchronous Recalibration and Evaluation (Lines 44–76, 128–140)**:
  - `_bn_recalibrate` runs 64 forward batches to refresh BatchNorm statistics.
  - `_evaluate` runs a full evaluation pass synchronously inside `server_main`. During evaluation, the server cannot service fetch requests or push updates, causing workers to stall.
- **Uncompressed Downlink Transfer (Lines 153–156)**:
  - `snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}`.
  - While worker-to-server gradient pushes achieve 10x–20x compression, server-to-worker model pulls are 100% uncompressed FP32 tensors, creating an egress bottleneck.
- **Bypassing PyTorch Optimizers (Lines 244, 275–276)**:
  - Updates are applied directly via `p.add_(g, alpha=-lr * decision.weight)`.
  - The server cannot maintain momentum buffers, Adam second moments ($v_t$), weight decay, or adaptive learning rates.
- **Hardcoded Learning Rate Schedule (Lines 143–147, 227–228)**:
  - Single step-decay milestone triggered when `version >= lr_milestone`.
  - Incompatible with PyTorch `torch.optim.lr_scheduler` (CosineAnnealing, MultiStep, Warmup).

### 2.3 `src/asgc/transport.py`
- **Multiprocessing Queue Dependency (Lines 65–86)**:
  - Relies entirely on `multiprocessing.Queue` initialized in a common parent process.
  - Cannot scale across multiple physical machines or heterogeneous networks.
  - PyTorch tensors are serialized through Python `pickle`, causing substantial serialization overhead.
- **Simulated Delays (Lines 54–62, 95–97)**:
  - Latency and bandwidth restrictions are simulated via `time.sleep()`.

### 2.4 `src/asgc/compression.py`
- **Per-Tensor Compression vs. Global Bucketing (Lines 87–94, 124–143)**:
  - Compresses each parameter tensor independently. For small bias vectors, Top-$K$ with rounding selects disproportionately high parameter fractions.
  - Reconstructing tensors in `decode()` allocates new tensors on every step without buffer recycling.
- **Index Overhead in Sparse Representation (Lines 28–33, 107–108)**:
  - `SparsePiece` uses flat `int32` indices and `float32` values: 8 bytes per non-zero value.
  - At active fraction $\rho = 0.5$, sparse overhead is $0.5 \times 8 = 4$ bytes/element (identical to uncompressed FP32). Compression benefits only occur when $\rho < 0.5$.
- **Absence of Sub-Byte Bit Packing (Lines 36–41, 152–154)**:
  - Quantized values are stored in `torch.int8`. At 2-bit or 4-bit quantization, each value still occupies 8 bits in memory and across the wire.

### 2.5 `src/asgc/staleness.py`
- **Scalar Version Staleness (Lines 19–27)**:
  - Staleness is defined strictly as $\tau = v_s - v_i$.
  - In large clusters (e.g. $W=64$ workers), the server version increments 64 times per logical round. A slightly delayed worker easily exceeds fixed bounds ($S_{max} \approx 6$), leading to severe update drops. Staleness bounds must scale dynamically with cluster worker count.

### 2.6 Critical Algorithmic Bug: Error-Feedback Leak on Rejection
- **Invariant**: Error feedback guarantees convergence under aggressive compression because discarded gradient mass $e = u - \hat{g}$ is accumulated into subsequent steps.
- **The Defect**:
  - In `worker.py#L74`: `residual = [ui - di for ui, di in zip(u, reconstructed)]`. The worker subtracts $\hat{g}$ from its local residual $e$ immediately.
  - In `server.py#L272-282`: If staleness $\tau > S_{max}$, the server rejects the update.
  - **Impact**: The server discards the gradient, but the worker already subtracted it from its residual. That gradient information is permanently lost, breaking the error-feedback conservation invariant ($u = g + e$) and causing systematic optimization bias.

---

## 3. Comparison with Distributed PyTorch Standards

| Framework / Feature | Communication | Compression / Staleness | User Interface Paradigm |
| :--- | :--- | :--- | :--- |
| **Current ASGC Prototype** | `multiprocessing.Queue` (single host) | Top-$K$, QSGD, Error Feedback, Bounded Staleness | Inverted control: library owns training loop and models |
| **PyTorch DDP Comm Hooks** | Collective AllReduce (NCCL/Gloo) | PowerSGD, FP16/BF16 hooks over bucketed tensors | Standard training loop: `DDP(model).register_comm_hook()` |
| **PyTorch Distributed RPC** | P2P RPC (`rpc_async`, `RRef`) | User-defined; `DistributedOptimizer` | Distributed autograd context: `dist_autograd.context()` |
| **BytePS** | Unified PS + AllReduce (P2P + RDMA/TCP) | Gradient partitioning, priority scheduling | Optimizer wrapping: `byteps.torch.DistributedOptimizer(opt)` |
| **Ray Train** | Actor-based IPC + NCCL/Gloo backends | Integrates with DeepSpeed / FSDP / DDP | `TorchTrainer` wrapping user-defined training functions |

*Note on PyTorch RPC*: While PyTorch core marked `torch.distributed.rpc` as maintenance mode in favor of collective synchronous training (NCCL AllReduce / FSDP) for homogeneous clusters, parameter servers remain the dominant paradigm for **heterogeneous hardware, asymmetric network links, edge nodes, and volatile spot instances**.

---

## 4. Architectural Requirements for `torch-asgc`

```
+-----------------------------------------------------------------------------------+
|                           User Application / Trainer                              |
|           (PyTorch Training Loop / PyTorch Lightning / HF Trainer)                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|               torch-asgc Client Engine (Worker Node)                              |
|  +-----------------------+  +--------------------------+  +--------------------+  |
|  | AsyncModelWrapper     |  | AsyncOptimizerWrapper    |  | ErrorFeedbackState |  |
|  | - Intercepts backward |  | - Gradient Bucketing     |  | - Residual storage |  |
|  | - Version tracking    |  | - Tensor Flattening      |  | - Rollback on rej. |  |
|  +-----------------------+  +--------------------------+  +--------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Compressor Engine (Top-K / QSGD / FP8 / Bit-packing / Adaptive Strength)    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                         |  Uplink: Compressed Gradients + Telemetry
                                         |  Downlink: Model Deltas / Quantized Weights
                                         v
+-----------------------------------------------------------------------------------+
|               torch-asgc Server Engine (Parameter Server Node)                    |
|  +-----------------------+  +--------------------------+  +--------------------+  |
|  | StalenessEngine       |  | StatefulOptimizerEngine  |  | AdaptiveController |  |
|  | - Dynamic S_max(W)    |  | - AdamW / Momentum / DC  |  | - Window Stats     |  |
|  | - Weight w(tau)       |  | - Weight Decay / Sched.  |  | - Dynamic Config   |  |
|  +-----------------------+  +--------------------------+  +--------------------+  |
+-----------------------------------------------------------------------------------+
                                         ^
                                         | Asynchronous Validation Metrics
+-----------------------------------------------------------------------------------+
|              Decoupled Evaluator Node (Runs val loop independently)               |
+-----------------------------------------------------------------------------------+
```

### 4.1 Clean User-Facing Public API
The user must retain ownership of model instantiation, loss computation, and loop execution. We define two primary entry points:
1. `asgc.DistributedOptimizer`: Intercepts `optimizer.step()`, handles error feedback, compresses gradients, dispatches them asynchronously, and fetches model updates.
2. `asgc.DistributedModel`: Wraps `nn.Module` to attach autograd backward hooks, enabling overlapped compression and communication during the backward pass.

### 4.2 Transport Layer Decoupling
Introduce an abstract `TransportBackend` protocol supporting multiple production implementations:
- **ZeroMQ (ROUTER-DEALER)**: Asynchronous, non-blocking TCP messaging for multi-node clusters.
- **Shared Memory (POSIX SHM / `torch.multiprocessing`)**: Zero-copy shared memory tensor transfers for multi-GPU training on a single node.
- **gRPC / Protocol Buffers**: Typed RPC interface with built-in TLS and load balancing for edge / Kubernetes setups.

### 4.3 Asymmetric Downlink Parameter Synchronization
To resolve the downlink egress bottleneck:
1. **Delta Parameter Broadcasting**: The server tracks recent parameter updates and transmits accumulated deltas $\Delta \theta = \sum_{t=v_{worker}}^{v_{server}} \eta_t w_t \hat{g}_t$ rather than full model snapshots.
2. **Quantized Downlink Weights**: Model weights sent to workers are quantized to FP8 or INT8 with per-channel scaling, dequantized on worker GPUs prior to the forward pass.
3. **Local Asynchronous Steps**: Workers perform $E_{local}$ local mini-batch steps before synchronizing weights with the server.

### 4.4 Advanced Optimizer Compatibility
1. **Momentum under Staleness**: Incorporate Delay-Compensated Momentum:
   $$v_t = \mu v_{t-1} + w(\tau) \hat{g}_t$$
2. **AdamW Second Moments**: Prevent Adam's scale-invariance from cancelling out staleness weights by updating the second moment using raw gradient magnitude, or using Delay-Compensated Adam (DC-Adam).
3. **Decoupled Weight Decay**: Apply weight decay according to server virtual clock time rather than irregular worker arrivals.
4. **Gradient Clipping**: Run `clip_grad_norm_` on uncompressed worker gradients prior to residual addition.

### 4.5 Automatic Mixed Precision (AMP) & Dynamic Models
1. **Contiguous Gradient Bucketing**: Flatten parameter gradients into fixed-size contiguous memory buffers (e.g. 25MB buckets, matching PyTorch DDP).
2. **AMP GradScaler Integration**: Unscale gradients before adding the error-feedback residual. Store worker residuals in FP32 (`master_residual`) to prevent underflow.
3. **Named Parameter Mapping**: Use `named_parameters()` to track gradients unambiguously, handling tied weights and skipping frozen layers (`requires_grad=False`).

### 4.6 Fault Tolerance & Invariant Conservation
1. **Residual Rollback on Rejection**: The server returns an acknowledgment (`accepted`, `downweighted`, or `rejected`). If rejected, the worker restores $e \leftarrow u$, preventing information loss.
2. **Dynamic Worker Membership**: Server tracks active workers via heartbeat leases, scaling staleness thresholds $S_{max}(t) = \kappa \cdot W_{active}(t)$.

### 4.7 Decoupled Telemetry & Evaluation
1. **In-Band Telemetry**: Workers piggyback loss values, computation time, and gradient norms directly onto `PushUpdate` packets.
2. **Streaming EMA Tracking**: The adaptive controller analyzes streaming exponential moving averages of training loss trends and variance.
3. **Independent Evaluator Process**: Validation loops run on an auxiliary evaluation worker, eliminating parameter server freezes.

---

## 5. Public API Specification

### 5.1 Standard PyTorch Training Script
```python
import torch
import torch.nn as nn
import torch_asgc as asgc

# 1. Connect client to parameter server
client = asgc.init_client(server_address="tcp://10.0.0.1:5555", rank=worker_rank)

model = MyCustomNet().cuda()
criterion = nn.CrossEntropyLoss()
base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 2. Wrap model and optimizer
model = asgc.DistributedModel(model, client=client)
optimizer = asgc.DistributedOptimizer(
    base_optimizer,
    client=client,
    error_feedback=True,
    bucket_cap_mb=25,
)

# 3. Standard training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch["input"]), batch["target"])
        loss.backward()
        
        # Intercepts grads, unscales AMP, applies EF, compresses, pushes, pulls deltas
        optimizer.step(loss_val=loss.item())
```

### 5.2 Parameter Server Launcher
```python
import torch
import torch_asgc as asgc

model = MyCustomNet().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

server = asgc.ParameterServer(
    model=model,
    optimizer=optimizer,
    bind_address="tcp://0.0.0.0:5555",
    staleness_config=asgc.StalenessConfig(s_low=2, s_max=12, beta=0.5),
    compression_config=asgc.CompressionConfig(mode="topk", rho=0.1),
    controller=asgc.AdaptiveController(enabled=True, interval_s=10.0),
)

server.serve_forever()
```

### 5.3 PyTorch Lightning Integration
```python
import lightning.pytorch as pl
import torch_asgc as asgc

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyCustomNet()
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        loss = self.criterion(self.model(batch[0]), batch[1])
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    strategy=asgc.lightning.ASGCStrategy(
        server_address="tcp://10.0.0.1:5555",
        bucket_cap_mb=25,
    ),
    max_epochs=10,
)
trainer.fit(LitModel(), train_dataloaders=dataloader)
```

---

## 6. Phased Implementation Roadmap

| Phase | Milestone | Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Invariant Hardening & Modularization** | Core Engine Decoupling | 1. Fix error-feedback residual rollback bug on rejected pushes.<br>2. Decouple evaluation and data loaders from `server.py`.<br>3. Implement contiguous 1D gradient bucketing in `compression.py`. |
| **Phase 2: Stateful Optimizer & AMP Integration** | Production Optimization | 1. Implement `StatefulServerOptimizer` (AdamW, momentum, weight decay).<br>2. Implement Delay-Compensated Momentum.<br>3. Integrate `torch.cuda.amp` unscaling and FP32 master residuals. |
| **Phase 3: Network Transport Layer** | Distributed Infrastructure | 1. Define `TransportBackend` abstraction.<br>2. Build `ZeroMQTransport` (TCP) for multi-node setups.<br>3. Build `SharedMemoryTransport` for single-node multi-GPU setups.<br>4. Implement downlink parameter delta synchronization. |
| **Phase 4: Public API & Ecosystem Wrappers** | Public Packaging | 1. Build `asgc.DistributedOptimizer` and `asgc.DistributedModel`.<br>2. Build PyTorch Lightning `ASGCStrategy` and Hugging Face callbacks.<br>3. Package as pip-installable library `torch-asgc` with unit and integration tests. |
