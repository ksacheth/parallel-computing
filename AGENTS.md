# This is a course project for the course name parallel computing

## Project Description

**Title:** Adaptive Staleness-Aware Gradient Compression for Asynchronous Distributed Training

A research prototype of a distributed machine learning training framework that addresses two practical problems in data-parallel training: (1) high communication overhead from transmitting large gradient vectors, and (2) worker heterogeneity (stragglers) that delays synchronous training or produces stale gradients in asynchronous training.

### Core Idea

The system is a parameter-server architecture that jointly and adaptively controls two things during training:

- **Gradient compression** — Top-K sparsification (with tunable active fraction ρ = k/d) or quantization, combined with **error feedback** so discarded gradient information is carried into later updates instead of being lost.
- **Staleness control** — every worker update is tagged with the model version it was computed from. The server measures staleness τ = (server version − worker version) and applies a bounded-staleness policy: accept at full weight if τ ≤ S_low, downweight with w = 1/(1 + βτ) if S_low < τ ≤ S_max, and reject if τ > S_max.

An **adaptive controller** periodically inspects live training and system signals (loss slope/variance, gradient-norm variation, average staleness, rejected-update fraction, communication time, worker throughput) and adjusts compression strength and staleness thresholds together: more aggressive compression and looser staleness when training is stable, safer settings when instability appears. Changes are bounded to avoid oscillation.

### System Architecture

- **Parameter server:** holds model parameters θ, server version, adaptive staleness thresholds, current compression setting, loss/communication statistics, and worker-update metadata.
- **Workers:** fetch model + version → compute mini-batch gradient → add error-feedback residual (u = g + e) → compress → send compressed gradient + version → update residual (e ← u − ĝ).

### Experimental Plan

- **Framework:** PyTorch (or similar distributed deep-learning framework).
- **Workloads:** Fashion-MNIST with a small CNN for debugging; CIFAR-10 with ResNet-18 as the main experiment.
- **Heterogeneity:** simulated via controlled artificial computation/communication delays on workers; multiple random seeds for key comparisons.
- **Baselines:** synchronous SGD, plain asynchronous SGD, async + fixed staleness bound, async + fixed Top-K compression, async + fixed quantization, and the full adaptive method.
- **Metrics:** final test accuracy, training loss, wall-clock time to target accuracy, total/average bytes transmitted, compression ratio, updates/sec, mean/max staleness, rejected-update ratio, sync idle time, loss stability.
- **Ablations:** adaptive controller without error feedback, error feedback with fixed compression, adaptive compression with fixed staleness, adaptive staleness with fixed compression, full joint adaptation.

### Success Criteria

- Lower total communication than uncompressed async SGD.
- Faster wall-clock convergence to target accuracy under heterogeneous workers.
- No major loss instability vs. unrestricted async training.
- Final accuracy close to the strongest non-adaptive baseline.

### Implementation Stages

1. Asynchronous parameter-server training with model-version tracking.
2. Fixed bounded-staleness accept/downweight/reject policy.
3. Top-K sparsification and quantization with communication logging.
4. Error feedback.
5. Adaptive controller (threshold rules first, then a weighted score R = aL + bV + cC + dT over loss trend, loss variability, communication pressure, and staleness).
6. Heterogeneous-worker experiments, ablations, and analysis.

### Key References

Parameter server (Li et al. 2014), Async-SGD (Lian et al. 2015), SSP (Ho et al. 2013), staleness-aware Async-SGD (Zhang et al. 2016), error-runtime tradeoffs (Dutta et al. 2018), QSGD (Alistarh et al. 2017), Deep Gradient Compression (Lin et al. 2018), error feedback (Karimireddy et al. 2019), async decentralized SGD with quantized updates (Nadiradze et al. 2021), AsGrad analysis (Islamov et al. 2024).

---

# Agent Guide

## Normative Sources

- `docs/methodology.tex` is the implementation spec: implement the equations and Algorithm 1 exactly as written there.
- `docs/proposal.tex` provides scope, motivation, and the evaluation plan. Do not edit either LaTeX document unless explicitly asked.

## Environment and Commands

- Python 3.11+ and PyTorch. CPU is sufficient for all experiments; CUDA is optional.
- Install dependencies: `pip install -r requirements.txt`.
- Datasets auto-download via torchvision into `data/` on first use (CIFAR-10 is about 170 MB).
- Run tests: `python -m pytest tests/ -q`. All tests must pass before a stage is considered done.
- Smoke run: `python scripts/run_experiment.py --config configs/smoke_fmnist.yaml`.
- Windows note: workers run as separate processes and Windows uses the spawn start method. Guard entry points with `if __name__ == "__main__"` and keep config objects picklable.

## Repo Layout

Stages 1–2 are implemented and gated by `tests/test_smoke.py` (async parameter server) and `tests/test_staleness.py` (fixed bounded-staleness policy; `staleness.enabled: false` in configs reproduces plain async SGD). Placeholders remaining for later stages: Top-K/quantize in `compression.py` (stage 3), error-feedback residual in `worker.py` (stage 4), `controller.py` (stage 5), and `models/resnet_cifar.py` (stage 6).

- `src/asgc/` — core package.
  - `config.py`: YAML config loading as dataclasses, global seeding.
  - `transport.py`: serializable messages (`FetchResponse`, `PushUpdate`), queue-based transport, byte accounting at the payload level.
  - `compression.py`: `Compressor` interface with `TopK` (values + indices), `Quantize`, and `Identity` implementations; wire size including sparse-index overhead.
  - `staleness.py`: pure accept / downweight / reject policy with weight w = 1/(1 + βτ).
  - `controller.py`: threshold policy and weighted-score policy (R = aL + bV + cC + dT), bounded per-interval changes.
  - `server.py`: parameter-server loop, version counter, applies the staleness policy, periodically invokes the controller.
  - `worker.py`: worker loop (fetch → gradient → u = g + e → compress → push → update residual) and delay injection.
  - `metrics.py`: JSONL event schema; the single source of truth for controller inputs and plots.
  - `data.py`: dataset loading and mini-batch partitioning across workers.
  - `models/`: `cnn_mnist.py` (debug workload), `resnet_cifar.py` (main workload).
- `configs/` — one YAML per experiment run; all hyperparameters and thresholds live here, never in code. `smoke_fmnist.yaml` is the stage gate; `baselines/` holds the six comparison runs; `adaptive/ablations/` holds the ablation runs.
- `scripts/` — `run_experiment.py` (the only entry point) and `plot_results.py` (JSONL to figures).
- `tests/` — unit tests per module plus `test_smoke.py` end-to-end.
- `results/` and `data/` are runtime artifacts created automatically on first run; keep them out of version control.

## Settled Design Decisions

Do not re-litigate these in future sessions:

- Single machine, simulated distribution. Workers are separate processes exchanging messages through the `transport.py` interface. Real multi-machine execution later means adding a transport implementation, not rewriting training logic.
- Framework is PyTorch; no alternative frameworks.
- Baselines and ablations run through the same pipeline, differing only in config (for example `controller.enabled: false` or `compression.mode: identity`).
- The controller reads only the event log produced by `metrics.py`, never worker internals.
- Wall-clock time includes injected delays. The delay schedule is part of the config and must be reported with results.

## Measurable Success Gates

- CIFAR-10 / ResNet-18: target accuracy 0.93; report wall-clock time to reach 0.90.
- Instability means NaN/Inf loss or a validation accuracy drop greater than 2 points between consecutive evaluations.
- "Close to the strongest non-adaptive baseline" means within 1 point final test accuracy.
- Communication gate: total transmitted bytes at most 50% of uncompressed async SGD at comparable accuracy. Calibrate this after stage 3 measurements, adjust once, then freeze.

## Testing Policy

- Unit tests must cover: Top-K returns correct indices and values; quantized values stay within the level range; error-feedback conservation (accumulated residual equals total discarded mass); staleness boundaries exactly at τ = S_low and τ = S_max; server version increments only on accepted updates.
- A stage is done when its tests pass and the smoke run converges (loss decreases, no NaN) on Fashion-MNIST with 3 workers.

## Stage-to-Module Mapping

1. Asynchronous parameter server with version tracking → `config.py`, `transport.py`, `models/`, `data.py`, `server.py`, `worker.py`, `metrics.py` with the `Identity` compressor and no staleness check.
2. Fixed bounded-staleness policy → `staleness.py` and `tests/test_staleness.py`.
3. Top-K / quantization with communication logging → `compression.py` and transport byte accounting.
4. Error feedback → residual state in `worker.py` and `tests/test_error_feedback.py`.
5. Adaptive controller → `controller.py` (threshold rules first, then the score policy) and `tests/test_controller.py`.
6. Heterogeneity experiments, ablations, analysis → `configs/` and `scripts/plot_results.py`.
