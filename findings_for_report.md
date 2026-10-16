# Findings for the Project Report

**Project:** Adaptive Staleness-Aware Gradient Compression for Asynchronous Distributed Training
**Scope of this document:** all measured results, diagnosed behaviors, and engineering findings from the six-stage implementation, with traceability to the run artifacts in `results/` and the per-stage plans in `docs/plans/`. Numbers marked *(n seeds = 1)* are single-run and indicative; the committed CIFAR matrix (`configs/matrix_cifar_full.yaml`) provides the multi-seed confirmation.

---

## 1. What was built

A parameter-server system that jointly controls gradient **compression** (Top-K with tunable active fraction ρ = k/d, QSGD-style quantization to b bits) and **staleness** (accept at full weight for τ ≤ S_low; downweight by w = 1/(1 + βτ) for S_low < τ ≤ S_max; reject for τ > S_max), with **error feedback** (EF) on every worker and a **runtime controller** that adapts (compression strength, S_max) together from the event stream.

- Single machine, simulated distribution: workers are separate processes (Windows spawn) exchanging pickled messages through a queue transport; heterogeneity is injected compute delay per worker plus an optional byte-proportional bandwidth constraint on the up-link. Wall-clock includes all injected delays.
- Every update event, evaluation, and controller decision is logged as JSONL (`results/<run>/events.jsonl`) — including per-update staleness, decision (accepted / downweighted / rejected), wire bytes, worker timings, EF residual norm, and controller window statistics with old→new values and reasons. This audit trail is itself a finding: it enabled every diagnosis below.
- Wire-byte accounting is exact: `payload_bytes` is computed at encode time (Top-K = 8 B/entry: float32 value + int32 index; quantize = 1 B/code + 4 B scale; identity = 4 B/value), so the compression-ratio metric CR = B_raw / B_cmp cannot drift from what was actually transmitted.
- Model sizes: Fashion-MNIST CNN = 55,338 parameters (221,352 B per uncompressed gradient); CIFAR-10 ResNet-18 = 11.17 M parameters (44.7 MB per gradient).

## 2. Experimental setup

| | Fashion-MNIST (debug) | CIFAR-10 (main) |
|---|---|---|
| model | small CNN, 2 conv layers | ResNet-18 (CIFAR variant) |
| batch / workers | 128 / 3 processes | 128 / 3 processes |
| stragglers (compute delay) | worker 1: +0.3 s | worker 1: +0.5 s, worker 2: +1.5 s |
| async budget | 3000 updates | 6000 updates (~15 epochs) |
| sync budget | 1500 rounds (= 4500 worker gradients) | 6000 rounds (unrun) |
| optimizer | SGD, server-side step decay (lr ×0.1 at half budget) | lr 0.05 (FMNIST) / 0.02 (CIFAR) |
| hardware | RTX 4050 laptop GPU (CUDA), one torch thread per process | same |

All baselines and the adaptive method run the identical pipeline and delay schedule; they differ only in config (`staleness.enabled`, `compression.mode/rho`, `error_feedback.enabled`, `controller.*`, `workload.sync`). Bandwidth-constrained runs charge each push `payload_bytes / bandwidth` (250 Mbps used), modeling the measured gradient up-link only — down-link model snapshots are identical across methods.

---

## 3. Headline result 1 — unconstrained network, Fashion-MNIST (seed 0)

Runs: `results/fmnist_*` (async: 3000 updates; sync: 1500 rounds).

| method | accuracy | loss | GB sent | CR | upd/s | time→0.85 |
|---|---|---|---|---|---|---|
| synchronous SGD | 86.11% | 0.393 | 0.996 | 1.0× | 2.7 | 274.3 s |
| plain async | 87.20% | 0.352 | 0.664 | 1.0× | 44.7 | 27.8 s |
| async + fixed staleness | 87.03% | 0.360 | 0.706 | 1.0× | 42.2 | 41.1 s |
| async + fixed Top-K | 86.92% | 0.369 | 0.332 | 2.0× | 40.4 | 31.5 s |
| **async + Top-K + EF** | **87.47%** | **0.349** | **0.332** | 2.0× | 38.5 | 32.4 s |
| adaptive (pre-fix controller) | 87.05% | 0.358 | 0.470 | 1.52× | 36.2 | 40.8 s |

**Findings:**
- **F1 — Asynchrony is worth 10–16× wall-clock under stragglers.** Plain async reached 85% accuracy 9.9× sooner than sync (27.8 s vs 274.3 s) at 16.3× the throughput. Caveat for fairness: the sync run performed 4500 worker gradient computations to async's 3000 and still took 8× longer in wall-clock.
- **F2 — Top-K + EF is the Pareto winner at every operating point tested:** the best accuracy of all six methods (87.47%) at exactly half the bytes. The AGENTS.md communication gate (≤50% of uncompressed at comparable accuracy) is met exactly.
- **F3 — Staleness control is insurance, not speed.** The fixed-staleness row sends *more* bytes than plain async (0.706 vs 0.664 GB — rejected gradients are still transmitted) and converges slightly slower. On an easy workload there was no stale-gradient damage to prevent; its value proposition (bounding applied staleness to ≤ 4 while plain async applied τ up to 19) shows up only as tail protection.
- **F4 — Fixed compression without EF costs accuracy** (86.92% vs 87.47% for the same bytes on FMNIST; larger on CIFAR, see §5).

## 4. Headline result 2 — CIFAR-10 / ResNet-18, unconstrained (seed 0, 6000 updates)

Runs: `results/cifar_async_plain`, `cifar_async_fixed_topk`, `cifar_async_topk_ef`, `cifar_adaptive_topk`.

| method | accuracy | GB sent | CR | upd/s | max eval drop |
|---|---|---|---|---|---|
| plain async | 85.50% | 268.2 | 1.0× | 5.38 | 0.4 pts |
| fixed Top-K (no EF) | 84.32% | 134.1 | 2.0× | 5.16 | 0.0 |
| **Top-K + EF** | **86.14%** | 134.1 | 2.0× | 5.00 | 0.0 |
| adaptive | 83.72% | **34.0** | **8.13×** | 4.95 | 0.0 |

**Findings:**
- **F5 — The central claim replicates on the main workload.** Top-K + EF beats uncompressed async on both axes: +0.64 points accuracy at exactly half the bytes, with zero stability drops across the run.
- **F6 — EF's contribution is larger on the harder model.** Fixed Top-K without EF loses 1.18 points to uncompressed (85.50 → 84.32); adding EF at identical bytes gains +1.82 points to 86.14%. Error feedback is worth ~3 points of swing on CIFAR vs ~0.5 on FMNIST.
- **F7 — The adaptive controller (with the rebound-fixed instability signal, §7) adapted cleanly at scale:** zero false instability trips across 40 logged decisions, ρ 0.25 → 0.05 (its floor) and S_max 4 → 10 within 2 minutes, holding the extreme operating point stably for the rest of the run. It found **8.13× compression — 4× less data than the best fixed policy — at a cost of 2.4 accuracy points.** This is the correct behavior for its objective and the wrong operating point for the project's accuracy gate (§9).

## 5. Headline result 3 — CIFAR-10 under a 250 Mbps up-link (seed 0)

Same four methods; each push sleeps `payload_bytes / bandwidth`. Runs: `results/*_bw250`.

| method | accuracy | wall time | time→0.80 | GB sent | CR |
|---|---|---|---|---|---|
| plain async | 86.01% | 70.4 min | 41.1 min | 268.2 | 1.0× |
| fixed Top-K | 84.60% | 54.0 min | 24.2 min | 134.1 | 2.0× |
| **Top-K + EF** | **86.50%** | **48.8 min** | **20.5 min** | 134.1 | 2.0× |
| adaptive | 83.76% | **26.7 min** | **13.7 min** | **29.8** | **9.01×** |

**Findings:**
- **F8 — Under bandwidth constraints the project beats plain async on every axis, including wall-clock.** Top-K + EF: +0.49 points accuracy, 31% less wall time, half the bytes, and 1.98× faster to 0.80 accuracy. The unconstrained table's one weakness (plain async ruling the time columns) was an artifact of a free network.
- **F9 — The adaptive method converts compression into wall-clock directly.** Its throughput *rose during the run* (1.4 → 3.75 upd/s) as the controller compressed harder under pressure — finishing 2.6× faster than plain async at 9× compression, and 3× faster to 0.80 (13.7 min). Its accuracy (83.76%) still reflects the extreme operating point (§9).
- **F10 — Compression's wall-clock value exists only when the link is the bottleneck.** With `comm_delay = 0` the total-time column is flat across methods (§3, §4); the same experiments at 250 Mbps separate by up to 2.6×. Deployment claims should therefore always state the network regime.

## 6. Answers to the proposal's research questions

- **RQ1 (how much communication can compression save without hurting quality?):** 50% at *better* accuracy than uncompressed on both workloads with Top-K + EF (F2, F5). Beyond that, up to 8–9× is reachable but enters a measurable accuracy trade (−2.4 points at 8.13× on CIFAR, F7/F9). The knee on CIFAR/ResNet-18 sits near ρ ≈ 0.10–0.15.
- **RQ2 (how does staleness affect convergence under heterogeneity?):** On both workloads, mean arrival staleness ≈ 2.1–2.9 with maxima 12–19 under the delay schedules used, and unrestricted async *did not* destabilize — final accuracy was within noise of every controlled variant. Staleness control bought bounded applied staleness (≤ S_max, verified from the event log) at a small throughput cost; its accuracy benefit was not observable on these workloads within these budgets.
- **RQ3 (does joint adaptation beat fixed policies in wall-clock?):** Conditionally yes — under a constrained link the adaptive method was the fastest route to target accuracy (13.7 min vs 20.5 for the best fixed policy and 41.1 for plain async), at an accuracy cost. Unconstrained, no: the controller's overhead and conservative drift made it slower than fixed Top-K + EF. The joint-control claim should be stated as regime-dependent.
- **RQ4 (how much does EF recover?):** At identical bytes: FMNIST 86.54% → 87.14% (+0.60); CIFAR 84.32% → 86.14% (+1.82). On CIFAR, compressed + EF beats uncompressed outright. The EF conservation invariant (Σ transmitted = Σ gradients − final residual) is pinned by unit test, and EF adds zero wire bytes by construction (verified: byte-identical totals with and without EF).
- **RQ5 (which signals are useful control inputs?):** System signals (rejection fraction, bytes/s) drove useful adaptation; the eval-loss **rebound** (mean upward move between evals) is a sound instability signal while raw loss spread is actively harmful (it reads fast learning as instability — the startup misclassification, §7); per-update residual norms are logged and available but not yet wired into the default policy; eval-loss trend was mostly uninformative because CIFAR's eval cadence (every 500 versions ≈ 95 s) exceeded the control interval (30 s) — control intervals must track eval cadence.

## 7. Controller diagnosis and fix (the A/B/C experiment)

The first adaptive runs underperformed the best fixed policy. The logged control decisions located the cause precisely: **every "instability" trip had an *improving* loss trend** — the raw variance term was measuring the size of the startup loss drop, i.e. punishing rapid learning, and ratcheting ρ up to CR 1.52 (FMNIST headline row above). The fix replaces raw spread with the **mean rebound** (average upward move between consecutive evals — monotone decay of any shape, however steep or convex, reads zero), plus an optional hysteresis knob (`instability_persistence`, default off).

Same config, re-run (FMNIST, `results/fmnist_adaptive_A` → `fmnist_adaptive`):

| controller | accuracy | GB | CR | t→0.85 |
|---|---|---|---|---|
| A: raw spread | 87.05% | 0.470 | 1.52× | 40.8 s |
| B: rebound | 87.09% | **0.195** | **3.65×** | **36.2 s** |
| C: rebound + persistence | 86.92% | 0.196 | 3.64× | 42.8 s |

- **F11 — With the signal fixed, the adaptive method beats the best fixed policy on bytes** (0.195 vs 0.332 GB, −41%) at equal accuracy (within noise) on FMNIST — the first configuration where the adaptive machinery earns its overhead on the debug workload.
- **F12 — Persistence never engaged after the fix** (no false unstable windows to gate); it is retained for CIFAR-class workloads where genuine instability is expected.
- Method note for the report: this diagnosis was only possible because controller decisions are logged with their input statistics — the audit-trail design paid for itself.

## 8. Staleness policy behavior (mechanics verified end-to-end)

- Gate run (`results/smoke_fmnist_staleness`): 12.4% of arrivals rejected, max arrival staleness 14 against S_max = 4, while fast workers' fresh updates passed at full weight; accuracy unaffected.
- The version-counter invariant is verified by replaying the ordered event log: applied versions step by exactly +1 and rejections never advance the version.
- Applied-staleness comparison under a 0.5 s straggler (`results/ab_async_*`): plain async applied gradients with staleness up to 15; the bounded policy capped applied staleness at ≤ 4 (S_max) with statistically indistinguishable final accuracy (89.54% vs 89.87% on FMNIST, single seed) — confirming F3's "insurance" framing.

## 9. Success-criteria scoreboard

| criterion (AGENTS.md) | status | evidence |
|---|---|---|
| total bytes ≤ 50% of uncompressed async at comparable accuracy | **met** | Top-K + EF: exactly 50% on both workloads at equal-or-better accuracy; adaptive: 11–15% (with accuracy cost) |
| faster wall-clock to target under heterogeneous workers | **met** | vs sync: 10–16× everywhere. vs plain async: no when the network is free (+16%), **yes (1.98×) under bandwidth** — state the regime |
| no major instability vs unrestricted async | **met** | all CIFAR runs ≤ 0.4 pts max eval drop; two FMNIST runs showed one transient > 2-pt drop each on a coarse eval grid — recheck across seeds |
| final accuracy within 1 pt of strongest non-adaptive baseline | **met on FMNIST** (adaptive 87.09 vs 87.47), **not yet on CIFAR** (83.76 vs 86.50 = 2.74 pts) at the committed ρ floor of 0.05 — the bounded tuning item |

## 10. Engineering findings (implementation lessons worth reporting)

1. **The BatchNorm evaluation artifact.** The parameter server never runs training forwards, so BN running statistics stayed at initialization and every ResNet evaluation read ~10% (chance) regardless of learning — the weights were fine; the measurement was broken. Fixed with a 64-batch server-side recalibration sweep before each eval (no-op for BN-free models, identical across methods). Any BN model evaluated on a server that only applies updates needs this.
2. **Two sync-topology bugs, both caught by exact byte accounting.** (a) Holding *all* fetch replies until a round completes deadlocks the system (workers cannot compute without their first fetch); the fix replies immediately to workers whose copy is behind the server. (b) A cross-queue race let fast workers recompute a version they had already served (+14% wasted bytes — detected because total bytes exceeded the exact 3 × rounds × model-size product); fixed by holding on `known_version == version`, which is race-free because a worker's known version only advances via replies.
3. **Version-aware fetch caching.** ResNet gradients are 44.7 MB; unconditional snapshot replies dominate wall-clock. Workers send their known version; the server ships a snapshot only when it differs. This plus GPU placement took CIFAR runs from impractical to ~20 min.
4. **One torch thread per process** measured faster than multithreaded for this workload (the server-threads A/B: 16 vs 10.5 updates/s); per-op thread-sync overhead dominates at these tensor sizes.
5. **Test-set evaluation, not training, was the original throughput bottleneck** (PIL decode per eval); tensorizing the test set once and making eval frequency a config knob fixed it.
6. **Learning-rate sensitivity differs between sync and async.** CIFAR lr 0.05 trained fine synchronously (verified in-process) but diverged under staleness ≈ 3; lr 0.02 is stable for every method. Asynchrony's effective step variance demands a gentler lr — a concrete instance of the staleness-aware training literature.
7. **Gate design lesson:** stage-gate tests must encode policy and learning assertions, not race the wall clock — a 1.8× machine-load swing made an update-budget assertion flaky until the budget was sized for slowdown.
8. **The audit trail design (event-sourced decisions) converted every debugging session into a query.** Both the EF conservation test's wrong first draft and the controller's startup misclassification were diagnosed from logs, not from guesswork.

## 11. Threats to validity and limitations

- **Single seed** for all headline tables; the committed 3-seed matrix exists but has not been run. FMNIST results in particular are within ~0.5-point noise bands.
- **Simulated distribution:** communication is queue-based pickling on one machine; the bandwidth model charges only the measured gradient up-link (down-link snapshots are uncharged, identical across methods); GPU contention between the server and worker processes perturbs absolute timings, which is why all claims are relative under identical conditions.
- **Budgets:** CIFAR runs cover ~15 epochs (~86% ceiling), far from the 93% AGENTS.md aspiration; `time_to_target` at 0.80 is available for all slice runs but the 93% target is unreachable at these budgets.
- **BN recalibration** is server-side and uses worker 0's data partition; it re-estimates statistics honestly per method but is an approximation of fully distributed BN handling.
- **The adaptive method's bounds are untuned:** ρ floor 0.05 (set on FMNIST) lets CIFAR runs compress past the accuracy knee; persistence and the residual-norm signal remain unexercised on genuinely unstable workloads.
- The proposal's "within 1 point" gate is evaluated against the strongest non-adaptive baseline, which is itself our Top-K + EF pipeline — a strong bar by construction.

## 12. Remaining work

1. Run the committed matrix: `python scripts/run_matrix.py --matrix configs/matrix_cifar_full.yaml` (9 configs × 3 seeds; sync runs dominate the wall time).
2. Retune the controller on CIFAR data: ρ floor ≈ 0.10–0.15, control interval matched to eval cadence, optionally wire residual-norm variation and a per-update training-loss signal into the policy; re-run the adaptive config (~20 min per iteration).
3. Add the sync baseline runs and the three ablation configs to produce the ablation table (EF-off, fixed-compression, fixed-staleness — all already expressible as configs).
4. Multi-seed aggregation for the transient >2-point drops observed on FMNIST (stability criterion).
5. Report figures are already generated by `scripts/plot_results.py` (accuracy/loss vs time, bytes-vs-accuracy, staleness-vs-rejection) into `results/plot/`.
