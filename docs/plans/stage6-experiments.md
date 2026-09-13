# Stage 6 Implementation Plan — Heterogeneous-Worker Experiments, Ablations, and Analysis

- **Branch:** `experiments`
- **Status:** complete — full machinery + committed matrix + first measured results (headline table below); the 33-run CIFAR matrix launches with one command
- **References:** [AGENTS.md](../../AGENTS.md), [docs/methodology.tex](../methodology.tex) (evaluation section), [docs/proposal.tex](../proposal.tex) (baselines, ablations, success criteria)

## Scope

The final stage: everything the evaluation needs. CIFAR-10 + ResNet-18 as the main workload, the synchronous SGD baseline, per-knob controller switches for the ablation matrix, the experiment runner and analysis tooling, the full config matrix, and first measured results on the debug workload.

**Prerequisite surfaced:** ResNet-18 gradients are ~45 MB; full-parameter fetch replies would dominate wall-clock. Stage 1's plan promised a cached-copy scheme — implemented here as version-aware fetches: workers send their known version, the server ships a full snapshot only when it differs.

**Honest note on scale:** the full CIFAR matrix (11 configs x 3 seeds = 33 runs) is a multi-hour-to-overnight job on CPU. This stage delivers the machinery, the committed configs, and a first measured slice (Fashion-MNIST headline comparison + a CIFAR smoke run); `scripts/run_matrix.py` launches the complete matrix unattended.

## Steps

### Step 1 — CIFAR-10 data + ResNet-18 (`data.py`, `models/resnet_cifar.py`)
- [x] CIFAR-10 loaders (standard normalization; RandomCrop+Flip augmentation on train only), tensorized test set as before.
- [x] CIFAR-variant ResNet-18 (3x3 stem, no maxpool, ~11.17M params); `build_model("resnet18")`.

*Verify:* unit tests — forward shape, parameter count band, small-CNN size guard. ✓

### Step 2 — version-aware fetch caching (`transport.py`, `server.py`, `worker.py`)
- [x] Fetch requests carry the worker's known version; the server ships a full snapshot only when its version differs, else a params-less reply the worker uses as "keep local copy".
- [x] Sync mode also holds fetch replies (step 3) — same plumbing.

*Verify:* existing gates unchanged; CIFAR smoke run feasible. ✓

### Step 3 — synchronous SGD baseline (`config.py`, `server.py`, gate)
- [x] `workload.sync: true`: the server buffers pushes per version, applies the mean of all worker gradients once per round, then releases the held fetch replies. Staleness policy bypassed (tau = 0 by construction); each push logged with weight 1/N.
- [x] Workers unchanged — their fetch simply blocks until the round completes.

*Verify:* sync gate — rounds applied == budget, one accepted update per worker per version, all tau zero, convergence. ✓ (after fixing two design bugs, see Outcome)

### Step 4 — per-knob controller switches (`config.py`, `controller.py`)
- [x] `controller.adapt_compression` / `adapt_staleness` (default true) so the ablation matrix can hold one knob fixed; policies respect the flags.

*Verify:* unit tests — disabled knob holds its value while the other adapts. ✓

### Step 5 — experiment configs
- [x] CIFAR-10 baselines (sync, plain async, fixed staleness, fixed Top-K, fixed quantization) + the full adaptive method + three ablation configs, filled from placeholders (lr 0.02 after the divergence diagnosis).
- [x] Fashion-MNIST headline comparison set under `configs/comparisons/fmnist_headline/` (six runs, one seed, same delay schedule) plus `configs/matrix_fmnist_headline.yaml` and `configs/matrix_cifar_full.yaml`.

*Verify:* all parse through `load_config`. ✓

### Step 6 — analysis tooling (`scripts/run_matrix.py`, `scripts/plot_results.py`, requirements)
- [x] `run_matrix.py`: run a list of configs x seeds sequentially (derived YAMLs under `results/`), the overnight driver for the full matrix.
- [x] `plot_results.py`: loss/accuracy-vs-time figures, the cross-run summary table (accuracy, loss, bytes, CR, updates/sec, staleness, rejection rate, time-to-target), bytes-vs-accuracy and staleness-vs-rejection comparison plots.
- [x] matplotlib added to requirements; GPU note added (CUDA wheels).

*Verify:* the summary table runs on the stage's own results. ✓

### Step 7 — first measured results
- [x] Fashion-MNIST headline set (6 configs, seed 0, identical 0.3 s straggler).
- [x] One CIFAR-10/ResNet-18 smoke run to validate the large-model path end to end.

### Step 8 — docs
- [x] AGENTS.md layout line, outcome section here, conventional commits, push.

## Outcome

**Headline comparison (Fashion-MNIST, small CNN, 3 workers with a 0.3 s straggler, seed 0; async runs 3000 updates, sync 1500 rounds; LR-decayed SGD):**

| run | acc | loss | GB sent | CR | upd/s | t→0.85 (s) |
|---|---|---|---|---|---|---|
| sync SGD | 86.11 | 0.393 | 0.996 | 1.0× | 2.7 | 274.3 |
| plain async | 87.20 | 0.352 | 0.664 | 1.0× | 44.7 | 27.8 |
| async + fixed staleness | 87.03 | 0.360 | 0.706 | 1.0× | 42.2 | 41.1 |
| async + fixed Top-K | 86.92 | 0.369 | 0.332 | 2.0× | 40.4 | 31.5 |
| async + Top-K + EF | **87.47** | 0.349 | **0.332** | 2.0× | 38.5 | 32.4 |
| **adaptive (full)** | 87.05 | 0.358 | 0.470 | 1.52× | 36.2 | 40.8 |

- **Top-K + error feedback dominates plain async on both axes**: best accuracy of all six methods at exactly half the transmitted bytes — the AGENTS.md communication gate (≤50% of uncompressed at comparable accuracy) is met outright on the debug workload.
- **Asynchrony's value is quantified**: ~10–16× the straggler-bound sync baseline's throughput (44.7 vs 2.7 updates/s), reaching target accuracy ~10× sooner.
- **The adaptive method adapted, bounded, and converged** (CR 1.52, ρ moved within bounds, 7.2% rejections) — but drifted conservative on this workload: eval-loss fluctuation trips `worsen_tol` tuned during the CPU era. Threshold retuning on the CIFAR matrix is the remaining experiment, exactly as the methodology predicted ("coefficients tuned using validation experiments").
- **Honest instability note**: under the strict >2-point single-eval drop definition, the fixed-staleness (3.3 pts) and Top-K+EF (3.6 pts) runs each show one transient drop on the coarse 250-version eval grid; final accuracies are unaffected. Multi-seed averaging in the full matrix is the right lens for this.

**Bugs the stage caught (all fixed, all documented by tests or re-runs):**
1. **Sync deadlock** — holding all fetch replies until a round completes prevents any round from ever completing; fixed by replying immediately to workers whose copy is behind the server, holding only same-version post-push fetches.
2. **Sync duplicate pushes** — a race between the fetch and push queues let workers recompute for a version they already served (14% wasted bytes); fixed by holding on `known_version == version` unconditionally (race-free since a worker's known version only advances via replies).
3. **BatchNorm eval artifact** — the server never runs training forwards, so ResNet evals used BN running stats frozen at initialization: accuracy pinned at chance despite learning weights. Fixed with a 64-batch server-side BN recalibration before every eval (no-op for BN-free models, identical for all methods). The first CIFAR smoke read 10.0%; after recalibration + lr 0.02 (0.05 diverged asynchronously though it trained fine synchronously) it reads 55.5% at 600 updates under staleness mean 2.8.

**GPU note:** the user's RTX 4050 required the CUDA PyTorch build (`torch==2.11.0+cu128`); `run.device: auto` now uses it everywhere, taking async runs from ~300 s to ~70 s and making CIFAR/ResNet-18 practical (600 updates in ~130 s). The full CIFAR matrix (33 runs) launches with `python scripts/run_matrix.py --matrix configs/matrix_cifar_full.yaml`.

## Definition of Done

- `pytest tests/ -q` green with the new model, sync, and controller-flag tests plus all five prior gates.
- The headline comparison table exists with real numbers against the proposal's success criteria.
- The full matrix is launchable with one command.
- Conventional commits pushed to `origin/experiments`.

## Risks

- **CIFAR/ResNet-18 wall-clock on CPU** — fetch caching mitigates the snapshot cost; compute is still ~1-2 s per update. Budgets set accordingly; the matrix is an overnight job, not an in-session one.
- **Sync round starvation on worker failure** — the wall-clock budget is the safety net; acceptable for the baseline's role.
- **Suite runtime** — two small gates added (sync, compact); matrix runs live outside the suite.
- **Single-seed headline numbers are indicative, not conclusive** — the committed matrix with seeds provides the real verdict.
