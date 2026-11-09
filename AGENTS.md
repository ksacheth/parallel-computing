# Parallel Computing Course Project: ASGC

## Project Description

**Title:** Adaptive Staleness-Aware Gradient Compression for Asynchronous Distributed Training

A distributed machine learning training framework addressing two core bottlenecks in data-parallel training:
1. **Communication overhead** from transmitting large gradient vectors across bandwidth-constrained networks.
2. **Worker heterogeneity (stragglers)** that causes idle barrier stalls in synchronous training or induces destabilizing stale gradients in asynchronous training.

The codebase comprises two complementary architectures:
- **`src/asgc/` (Research Simulation Benchmark):** Deterministic, reproducible multiprocess simulation harness evaluating async parameter servers, artificial worker delays, wire-exact gradient compression, error feedback, and adaptive controllers across Fashion-MNIST and CIFAR-10.
- **`src/torch_asgc/` (Production Public Interface):** Modular, uncoupled PyTorch distributed training library (`DistributedModel`, `DistributedOptimizer`, `ParameterServer`, binary TCP transport, mixed-precision AMP, bucketing, downlink delta broadcasting, local gradient accumulation, and framework plugins for PyTorch Lightning and Hugging Face).

---

## Domain Glossary

Consult [`CONTEXT.md`](CONTEXT.md) for the authoritative domain model, canonical vocabulary, and deprecated synonyms. Always use the terms defined there (e.g., *Server Version*, *Worker Version*, *Staleness*, *Bounded Staleness*, *Error Feedback*, *Residual Rollback*, *Active Fraction*, *Loss Rebound*).

---

## Agent Guide

### Normative Sources

- [`docs/methodology.tex`](docs/methodology.tex): Implementation specification containing the formal algorithmic equations and parameter server loop.
- [`docs/proposal.tex`](docs/proposal.tex): Project scope, theoretical motivation, and evaluation plan. Do not edit either LaTeX document unless explicitly instructed.
- [`docs/public_interface_gap_analysis.md`](docs/public_interface_gap_analysis.md): Technical audit, mathematical formulation of error-feedback residual rollback, and architecture of `torch-asgc`.
- [`docs/downlink_optimization_research.md`](docs/downlink_optimization_research.md): Empirical and algorithmic analysis of downlink communication bottlenecks and delta broadcast solutions.
- [`findings_for_report.md`](findings_for_report.md): Authoritative repository of measured experimental results, findings F1–F17, the evaluation scorecard, and §12 "Deviations from the written methodology".

### Environment & Commands

- Python 3.11+ with PyTorch (CPU sufficient; CUDA auto-detected when available).
- Install dependencies: `pip install -r requirements.txt`.
- Install public library in editable mode: `pip install -e .`.
- **Fast Unit Tests (Library):** `python -m pytest tests/test_torch_asgc -q` (33 tests covering TCP transport, AMP, bucketing, compression, and plugins; ~30–45 s. The Hugging Face integration test requires both `transformers` and `accelerate`).
- **Full Verification Suite:** `python -m pytest tests/ -q` (all unit and end-to-end integration training runs).
- **Multi-Machine Run (cloud server + Colab workers):** `python scripts/ps_server.py --task fmnist` on the server, `python scripts/ps_worker.py --server tcp://<ip>:29500 --rank N --num-shards W` per worker, both with the same `ASGC_AUTHKEY`. See `docs/distributed_deployment.md`.
- **Smoke Benchmark Run:** `python scripts/run_experiment.py --config configs/smoke_fmnist.yaml`.
- **Fashion-MNIST Headline Matrix:** `python scripts/run_matrix.py --matrix configs/matrix_fmnist_headline.yaml`.
- **Evaluation Matrix (CIFAR-10):** `python scripts/run_matrix.py --matrix configs/matrix_cifar_full.yaml`.
- **Generate Plots & Summary Tables:** `python scripts/plot_results.py --runs fmnist_sync_sgd fmnist_async_plain fmnist_async_topk_ef --target 0.85` (all arguments are required).
- Windows note: Workers run as separate processes via the `spawn` start method. Guard entry points with `if __name__ == "__main__"` and ensure configuration and transport payloads are picklable.

### Repository Architecture

- `src/torch_asgc/` — Decoupled public distributed training library:
  - `model.py`: `DistributedModel` wrapper synchronizing parameter snapshots with the server.
  - `optim.py`: `DistributedOptimizer` intercepting gradients, executing error feedback, compression, and residual rollback.
  - `server.py`: Authoritative `ParameterServer` managing TCP client connections, staleness filtering, version increments, and optimizer steps.
  - `client.py`: `connect()` factory and `ASGCClient` network lifecycle manager.
  - `transport/`: `TCPTransportClient` and `TCPTransportServer` with binary payload framing.
  - `compression.py`: Top-K sparsification, QSGD quantization, and Identity compressors with wire-byte calculation.
  - `staleness.py`: Staleness evaluation, soft attenuation ($w = 1/(1+\beta\tau)$), and hard bound cutoff ($S_{max}$).
  - `controller.py`: Telemetry-driven adaptation of compression active fraction $\rho$ and staleness cutoff $S_{max}$.
  - `engine/`: AMP mixed-precision integration, gradient bucketing, downlink delta broadcasting, and local gradient accumulation (`local_steps` averages micro-batch gradients; it does not take worker-side optimizer steps).
  - `plugins/`: PyTorch Lightning strategy (`ASGCStrategy`), Hugging Face Trainer subclass (`ASGCTrainer`), callback (`ASGCTrainerCallback`), and `wrap_hf_trainer` helper.
- `src/asgc/` — Research simulation harness:
  - `server.py`, `worker.py`, `transport.py`: Queue-based multiprocess parameter-server simulation.
  - `compression.py`, `staleness.py`, `controller.py`: Algorithmic modules.
  - `metrics.py`: JSONL event logging (`results/<run>/events.jsonl`) — the single source of truth for telemetry and audit.
  - `data.py`, `models/`: Dataset partitioning and model definitions (`SmallCNN`, `ResNet18`).
- `configs/` — Declarative YAML experiment configs (`smoke_*.yaml`, `baselines/`, `adaptive/`, `matrix_cifar_full.yaml`).
- `notebooks/` — Self-contained notebooks: `transformer_asgc_cuda.ipynb` (runs the GPT/T4 experiment), frozen-result notebooks for Fashion-MNIST and CIFAR-10 (no rerun required), and `torch_asgc_plugins_colab.ipynb` (Hugging Face / Lightning / TCP integration checks).
- `artifacts/transformer_asgc/` — Committed transformer evidence bundles (`run_20261102_104333` headline, `run_20261102_095934` cross-run comparison; checkpoints excluded).
- `docs/` — Specifications, per-stage implementation plans (`docs/plans/stage1-6`), research findings, and presentation slide scripts.
- `tests/` — Test suites (`test_torch_asgc/` for library components and root `tests/` for research simulation gates).

### Settled Design Invariants

Do not re-litigate these architectural decisions:
1. **Residual Rollback on Rejection (`torch_asgc` only):** When an update is rejected due to staleness exceeding the hard bound ($\tau > S_{max}$), the worker rolls back its residual state to the pre-transmission value ($e \leftarrow e_{prev} + g$), preventing permanent loss of gradient mass. This is driven by `PushAck` and is implemented in `src/torch_asgc`; the research harness `src/asgc` uses fire-and-forget pushes with no acknowledgement, so its reported F1–F12 runs do not exercise rollback (see findings §12 item 4 for the piggyback-ACK fix design).
2. **Loss Rebound vs. Raw Spread:** The adaptive controller assesses training instability via the mean positive loss rebound (average upward move), never raw variance, preventing startup fast-learning drops from being misclassified as instability. The research harness feeds it evaluation loss. `torch_asgc` currently feeds per-push training loss into the same statistic; the invariant is the rebound computation, and wiring smoothed/eval loss in the library is an open fix (findings §12.2).
3. **Exact Wire-Byte Accounting:** Wire byte measurements are computed at encode time (Top-K: 8 bytes/entry for float32 value + int32 index; Quantize: 1 byte/entry + scale; Identity: 4 bytes/float32), preventing metric drift.
4. **Decoupled Control Flow:** `torch_asgc` must not own user training loops, datasets, or loss functions; it operates exclusively by wrapping standard PyTorch `nn.Module` and optimizers.
5. **Event-Sourced Audit Trail:** The controller inspects only streaming telemetry emitted by the logging layer, preserving strict separation of concerns.

### Verified Empirical Findings Summary

Refer to [`findings_for_report.md`](findings_for_report.md) for full datasets and trace logs:
- **F1 (Asynchrony Speedup):** Plain async achieves 9.9× wall-clock (16.3× throughput) speedup over averaged synchronous SGD under worker compute heterogeneity. The transformer's barrier-emulation comparison is 1.46× and uses a different sync semantics (findings §12.3).
- **F2 & F5 (Top-K + EF Dominance):** Transmits exactly 50% of the uncompressed up-link while preserving or improving final test accuracy (+0.64 accuracy points on CIFAR-10 ResNet-18). On the transformer, 2.50× less up-link preserves loss within ~0.03 nats.
- **F6 (Error Feedback Recovery):** Error feedback recovers +1.82 accuracy points on CIFAR-10 compared to uncompensated Top-K at identical bytes; without it, fixed Top-K loses 1.18 points to uncompressed async.
- **F8 & F9 (Bandwidth Bottleneck Speedup):** Under a 250 Mbps up-link constraint, Top-K + EF delivers a 1.98× wall-clock speedup to 80% accuracy over uncompressed async SGD.
- **F11 (Adaptive Byte Savings):** The tuned adaptive controller achieves 3.65× compression on Fashion-MNIST at equal accuracy. On CIFAR-10 it reaches 8.13× unconstrained and 9.01× under a 250 Mbps up-link, but at a 2.4–2.7 accuracy-point cost; on the transformer it saves 27.2% up-link / 15.9% total wire versus fixed Top-K at comparable quality.
