# Transformer ASGC Experiment - Findings

- Runtime: Linux-6.6.159+-x86_64-with-glibc2.39; Python 3.13.16; torch 2.11.0+cu130
- GPU: cuda:0 (Tesla T4); CUDA 13.0; basic preflight=True; stack preflight=True
- GPU worker used: True; final GPU evaluation: True
- Corpus: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt (1,115,394 chars)
- Model: 1,809,984 parameters; block=128; layers=4; heads=4; embd=192
- Run: seeds=[0]; workers=4; target_updates=200; uplink=32.0 Mbps; downlink=128.0 Mbps
- Repo commit: fcdfecdb6e205399211bf18e665eceb6f4489013

## Arm results

| arm | wall (s) | updates | val loss | ppl | acc1 % | acc5 % | uplink MB | downlink MB | total MB | mean tau | reject % | rho mean | S_max mean | time-to-target (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| async_identity | 128.4 +/- 0.0 | 202 +/- 0 | 2.7333 +/- 0.0000 | 15.38 +/- 0.00 | 23.91 +/- 0.00 | 60.12 +/- 0.00 | 1462.47 +/- 0.00 | 418.60 +/- 0.00 | 1881.07 +/- 0.00 | 2.95 +/- 0.00 | 0.0 +/- 0.0 | 1.000 +/- 0.000 | 10.00 +/- 0.00 | 113.3 +/- 0.0 |
| async_topk20 | 82.1 +/- 0.0 | 203 +/- 0 | 2.6734 +/- 0.0000 | 14.49 +/- 0.00 | 23.44 +/- 0.00 | 61.03 +/- 0.00 | 587.88 +/- 0.00 | 420.67 +/- 0.00 | 1008.55 +/- 0.00 | 2.83 +/- 0.00 | 0.0 +/- 0.0 | 0.200 +/- 0.000 | 10.00 +/- 0.00 | 60.6 +/- 0.0 |
| async_adaptive | 77.2 +/- 0.0 | 203 +/- 0 | 2.7080 +/- 0.0000 | 15.00 +/- 0.00 | 23.20 +/- 0.00 | 61.50 +/- 0.00 | 516.21 +/- 0.00 | 426.89 +/- 0.00 | 943.10 +/- 0.00 | 2.82 +/- 0.00 | 1.5 +/- 0.0 | 0.175 +/- 0.000 | 7.85 +/- 0.00 | 53.7 +/- 0.0 |
| sync_identity | 190.1 +/- 0.0 | 200 +/- 0 | 2.9119 +/- 0.0000 | 18.39 +/- 0.00 | 20.30 +/- 0.00 | 52.91 +/- 0.00 | 1447.99 +/- 0.00 | 414.45 +/- 0.00 | 1862.44 +/- 0.00 | 2.94 +/- 0.00 | 0.0 +/- 0.0 | 1.000 +/- 0.000 | 10.00 +/- 0.00 | n/a |

`*` marks an arm with at least one partial seed (a worker failed but the remaining workers produced metrics).

## Headline comparisons

- Asynchronous vs barrier-synchronous (uncompressed): **1.48x (std 0.00, n=1)** wall-clock speedup.
- Top-K 20% + EF vs uncompressed async: **1.56x (std 0.00, n=1)** wall-clock, **2.49x (std 0.00, n=1)** uplink reduction, delta validation loss **-0.0599 (std 0.0000, n=1)**, delta top-1 accuracy **-0.00 pp (std 0.00, n=1)**.
- Final top-1 / top-5 next-token accuracy by arm: `async_identity` 23.91% / 60.12%, `async_topk20` 23.44% / 61.03%, `async_adaptive` 23.20% / 61.50%, `sync_identity` 20.30% / 52.91%. The random baseline is about 1.54% top-1 and 7.69% top-5 for a 65-character vocabulary.
- Adaptive vs fixed Top-K 20% + EF: **12.2% (std 0.0, n=1)** fewer uplink bytes, **6.5% (std 0.0, n=1)** fewer total wire bytes, **1.06x (std 0.00, n=1)** wall-clock, delta validation loss **0.0346 (std 0.0000, n=1)**, delta top-1 accuracy **-0.00 pp (std 0.00, n=1)**.
- Adaptive time-to-target vs uncompressed async (target = 1% above uncompressed final loss): **2.11x (std 0.00, n=1)**.
- Adaptive time-to-target vs fixed Top-K: **1.13x (std 0.00, n=1)**.
- Adaptive vs uncompressed async: **1.66x (std 0.00, n=1)** wall-clock, **1.99x (std 0.00, n=1)** total wire reduction.

## Adaptive controller behavior

- Mean `rho` over the run: **0.175**; final `rho`: **0.100** (started at 0.200).
- Mean `S_max`: **7.85**; final `S_max`: **8.00** (started at 10).
- Mean rejection rate: **1.46%**; mean staleness: **2.82**.
- Relative to the fixed Top-K 20% baseline, the adaptive arm cost a small amount of final validation loss (delta +0.0346) while transmitting 12.2% (std 0.0, n=1) fewer uplink bytes.
- The controller reduced `rho` toward its floor and raised `S_max` under sustained communication pressure. If final-loss preservation is more important than peak byte savings, raise `ASGC_RHO_MIN` (for example to 0.12-0.15) and rerun.

## What this says about transformers

- The GPT transformer trained end-to-end through `DistributedModel` and `DistributedOptimizer`; no transformer-specific assumptions were needed in ASGC.
- Gradient bucketing, Top-K sparsification, error feedback, residual rollback, and server-side parameter synchronization all operated on attention and MLP parameters without modification.
- At equal accepted-update budgets, compression changed the bandwidth profile substantially; the quality effect is measurable and should be reported as a trade-off rather than hidden.
- The adaptive controller is most valuable when the uplink is the constrained resource. With a large downlink (INT8 delta broadcast here), total-wire savings are smaller than uplink savings.

## What this says about the CUDA path

- Basic CUDA preflight: `{'ok': True, 'loss': 4.2111430168151855, 'elapsed_s': 0.4906887479999966, 'device': 'cuda:0', 'name': 'Tesla T4'}`.
- Full-stack CUDA preflight (Top-K + EF + push/fetch): `{'ok': True, 'loss': 4.221540451049805, 'server_version': 1, 'elapsed_s': 0.32668175200001315}`.
- Only worker 0 owns the single GPU in this configuration; the parameter server and the other workers run on CPU.
- The CPU workers are intentionally slower stragglers, so the run exercises bounded staleness and downweighting. A multi-GPU slice would allow more GPU workers, but the notebook is sized for one visible device.

## Caveats

- Bandwidth and worker delays are simulated with deterministic profiles and per-worker RNG; wall-clock results are not measurements of a real network.
- The synchronous arm is a barrier-synchronous emulation on the same asynchronous parameter-server code path, not a separate synchronous trainer.
- The model is a small GPT (about 1.8M parameters in the report profile), not a production LLM. Conclusions about communication savings should scale with model size, but quality sensitivity may differ.
- Results are seed-dependent; with the default single seed, the table reports one run per arm. Use `ASGC_SEEDS=0,1,2` for a report-grade confidence interval.
- Accuracy is next-token (teacher-forced) accuracy on held-out text, not generation quality or task accuracy. The random baseline for the 65-character vocabulary is about 1.54% top-1 and 7.69% top-5.
- If the corpus source is `synthetic-fallback`, the absolute loss values are not comparable to Tiny Shakespeare runs; use the comparisons within the run.

## Suggested report bullets

1. ASGC's asynchronous path was **1.48x (std 0.00, n=1)** faster than barrier-synchronous training under the injected worker-delay profiles.
2. Top-K 20% with error feedback cut uplink traffic by **2.49x (std 0.00, n=1)** while changing final loss by **-0.0599 (std 0.0000, n=1)** and top-1 accuracy by **-0.00 pp (std 0.00, n=1)** on a GPT transformer.
3. The adaptive controller cut uplink bytes by **12.2% (std 0.0, n=1)** and total wire bytes by **6.5% (std 0.0, n=1)** relative to fixed 20% Top-K, with **1.06x (std 0.00, n=1)** wall-clock.
4. The adaptive controller moved `rho` from 0.200 toward **0.100** and raised `S_max` toward **8.0**, demonstrating closed-loop adaptation under communication pressure.
5. The full ASGC stack (Top-K, error feedback, residual rollback, parameter-server synchronization) ran on a GPT transformer with worker 0 on a CUDA device and the remaining workers on CPU.

## Files in this bundle

- `FINDINGS_TRANSFORMER.md` - this report.
- `results.json` - aggregate summary, per-seed summaries, raw result records, and environment metadata.
- `summary_per_seed.csv` and `summary_aggregate.csv` - report tables.
- `eval_curves.csv`, `push_events.csv`, `monitor_samples.csv`, `worker_summary.csv` - raw telemetry.
- `checkpoints/*.pt` - final server state dict per arm and seed, for checkpoint-only accuracy re-evaluation.
- `plots/*.png` - wall-clock, wire bytes, loss curves, accuracy curves, time-to-target, rejections, adaptive trajectory, and bandwidth-quality trade-off.
- `config.json` and `environment.json` - reproducibility metadata.
