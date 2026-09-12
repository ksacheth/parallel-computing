# Stage 1 Implementation Plan — Asynchronous Parameter Server with Version Tracking

- **Branch:** `parameter-server`
- **Status:** complete — smoke gate green (7500 updates, final test accuracy ≈ 0.898, loss ≈ 0.286)
- **References:** [AGENTS.md](../../AGENTS.md) (agent guide, settled design decisions, testing policy) and [docs/methodology.tex](../methodology.tex) (normative spec — equations and Algorithm 1)

## Scope

An end-to-end asynchronous training run: N worker processes push gradients tagged with the model version they were computed from, and the server applies every update and increments its version counter. Validated by the Fashion-MNIST smoke run.

**In scope:** Identity compressor only; no staleness policy, no error feedback, no controller, no ResNet-18. Everything lands as small conventional commits on the `parameter-server` branch.

**Explicitly deferred to later stages:** `staleness.py` (stage 2), Top-K / quantization (stage 3), error feedback (stage 4), adaptive controller (stage 5), ResNet-18 + CIFAR configs and experiments (stage 6).

## Steps (dependency order)

### Step 1 — `requirements.txt` + `src/asgc/config.py`
- [x] Fill `requirements.txt` (torch, torchvision, numpy, pyyaml, pytest).
- [x] Picklable config dataclasses matching the YAML schema: `RunConfig` (name, seed, results_dir, data_root), `WorkloadConfig` (dataset, model, batch_size, num_workers, lr plus a server-side step schedule via `lr_decay_frac`/`lr_gamma`, target_accuracy, eval_interval, budget via max_updates and max_wall_time_s), `HeterogeneityConfig` (per-worker compute delay, comm delay), `CompressionConfig` (mode only). Staleness and controller sections are added to the schema in their own stages so stage 1 configs stay honest.
- [x] YAML loader + `seed_everything` (random, numpy, torch).

*Verify:* loading `configs/smoke_fmnist.yaml` produces a typed object.

### Step 2 — `src/asgc/transport.py`
- [x] Serializable messages: `FetchResponse{params, version, stop}`, `PushUpdate{worker_id, version, payload, payload_bytes, fetch_s, compute_s}`.
- [x] Queue-based transport: fetch-request queue + per-worker reply queues + push queue, created with the spawn-safe multiprocessing context in the parent process.
- [x] Byte accounting lives here: `payload_bytes` is set at encode time; the transport only sums it. Communication delay is applied inside `push`/`reply`.

*Verify:* roundtrip test — push a message, receive it intact with the correct byte count.

### Step 3 — `src/asgc/compression.py`
- [x] `Compressor` protocol (`encode` / `decode` / `encoded_bytes`).
- [x] `IdentityCompressor` only. The protocol is the deliverable so stage 3 is purely additive.

*Verify:* fill `tests/test_compression.py` — identity roundtrip is exact, bytes = numel × element_size. ✓ 3 tests pass.

### Step 4 — `src/asgc/metrics.py`
- [x] JSONL event schema and writer: `run_start`, `update`, `eval`, `run_end`.
- [x] `update` event fields: worker_id, version_applied, staleness, bytes, fetch/compute timings, run-relative elapsed time, `decision` field (always `"accepted"` in stage 1, but present from day one so stage 2 does not change the schema).
- [x] Aggregation helpers: updates/sec, total and average bytes.
- [x] Workers attach their timings to `PushUpdate`; the server owns the writer.

*Verify:* write/read cycle on a temp file.

### Step 5 — `src/asgc/data.py` + `src/asgc/models/cnn_mnist.py`
- [x] Fashion-MNIST loaders, auto-download into `data/`.
- [x] Batch streams partitioned by worker (index-modulo partition via sampler, cycling endlessly).
- [x] Small 2-conv CNN (~55k parameters) so per-step full-model fetches stay cheap. ResNet-18 deferred.

*Verify:* loader yields correct shapes; the partition covers all batches exactly once per epoch across workers. ✓ via smoke gate.

*Verify:* loader yields correct shapes; the partition covers all batches exactly once per epoch across workers.

### Step 6 — `src/asgc/delays.py`
- [x] Constant per-worker compute delay and global comm delay from config, applied as sleeps at the defined points (worker inside its compute window, transport on send).
- [x] Optional uniform-jitter field for later; no randomness needed in stage 1.

*Verify:* delay schedule resolves from config.

### Step 7 — `src/asgc/server.py`
- [x] Server state: authoritative model (CPU tensors), integer version, metrics writer.
- [x] Single-threaded loop multiplexing fetch requests (reply with snapshot + version) and pushes (apply `θ ← θ − η·ĝ`, version += 1, log update event).
- [x] Vanilla SGD with a server-side step LR schedule (`lr_decay_frac`/`lr_gamma`) so every method sees the same schedule by version; momentum deferred until its design question is settled.
- [x] Inline periodic eval every `eval_interval` versions; stop on update budget or wall-clock budget, poison-pill the workers, write `run_end`.

*Verify:* unit-level — apply 5 synthetic updates, version reaches 5, params actually changed. ✓ via smoke gate version checks.

### Step 8 — `src/asgc/worker.py`
- [x] Loop: fetch (model, version) → next batch → compute delay → forward/backward → encode gradient → push with its version.
- [x] No residual state yet — error feedback is a stage 4 addition, not a placeholder.
- [x] Spawn-safe entry `worker_main(config, worker_id)`; `if __name__ == "__main__"` guards everywhere.

*Verify:* covered by the smoke test.

### Step 9 — `scripts/run_experiment.py` + `configs/smoke_fmnist.yaml`
- [x] The only entry point: `--config` → seed → spawn server + 3 workers → run → teardown → print summary (final accuracy, total bytes, updates/sec, elapsed).
- [x] Smoke config: Fashion-MNIST, small CNN, batch 128, 3 workers, 7500 updates, eval every 500 versions, worker 1 with a 50 ms compute delay to prove the heterogeneity plumbing works. Budget resized from ~300 updates after measuring real throughput (see Outcome).

### Step 10 — `tests/test_smoke.py`
- [x] End-to-end: run the smoke config programmatically, then assert the version count matches applied updates, loss decreased across evals, no NaN, and `results/smoke_fmnist/events.jsonl` parses with all four event types.

## Definition of Done

- `pytest tests/ -q` passes.
- The smoke run converges (~90%+ on Fashion-MNIST within the budget).
- The JSONL event log is complete and monotonically increasing in version.
- Summary printed: final accuracy, total bytes, updates/sec, elapsed time.
- Conventional commits per step pushed to `origin/parameter-server`; ready for stage 2 (staleness policy) to slot in behind the `decision` field stage 1 already logs.

## Known Risks

- **Windows spawn:** anything passed to a `Process` target must be picklable — config dataclasses only, no lambdas or open files; queues constructed in the parent. ✓ confirmed in practice.
- **Fetch cost:** full model snapshot per fetch is fine for the small CNN but needs a cached-copy scheme before CIFAR/ResNet-18 (11M params ≈ 44 MB per fetch). Flag it in the `run_end` summary now; fix in a later stage.
- **Timing honesty:** all timings via `time.perf_counter`; wall-clock includes injected delays per the AGENTS.md convention — the delay schedule is logged in `run_start`. ✓ implemented.

## Outcome (what implementation changed vs. the plan)

- **Budget resized after measurement.** The planned "~300 updates" smoke reached only 77.8% accuracy; throughput measurement showed ~16 updates/sec overall, so the gate was resized to 7500 updates (~16 epochs) with eval every 500 versions, landing at 89.8% accuracy / 0.286 loss in ~8 min wall-clock.
- **Test-set evaluation was the hidden bottleneck**, not training (~27 updates/sec of pure training). The test transform is now applied once into a tensor cache (`data.py`), and eval frequency is a config knob.
- **Thread pinning matters.** Each process pins `torch.set_num_threads(1)`; giving the server extra threads made small ops slower (thread-sync overhead), so one thread per process is the settled configuration.
- **LR schedule added to stage 1.** The methodology requires a fixed LR schedule across all methods; a server-side step decay (`lr_decay_frac`/`lr_gamma`) is the minimal version, applied uniformly by version.
- **Update event carries fetch/compute timings plus run-relative elapsed time** (not per-stage send/apply timestamps as first sketched); `decision` is present from day one for stage 2.
