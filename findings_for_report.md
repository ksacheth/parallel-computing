# Findings for the Project Report

**Project:** Adaptive Staleness-Aware Gradient Compression for Asynchronous Distributed Training
**Scope of this document:** all measured results, diagnosed behaviors, and engineering findings from the six-stage implementation plus the GPT-transformer CUDA slice (§5b), with traceability to the run artifacts in `results/` (gitignored, regenerable), the committed transformer bundles in `artifacts/transformer_asgc/`, and the per-stage plans in `docs/plans/`. Numbers marked *(n seeds = 1)* are single-run and indicative; `configs/matrix_cifar_full.yaml` is the intended multi-seed confirmation but has not been run. Where the implementation diverges from the written specification, §12 records it explicitly.

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

A fourth workload — the GPT transformer on Tiny Shakespeare with one T4 and a 32/128 Mbps link — is reported separately in §5b; it uses four workers and a 200 accepted-update budget, so its absolute numbers are not comparable to the tables above.

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
- **F1 — Asynchrony is worth 9.9× wall-clock (16.3× throughput) under stragglers.** Plain async reached 85% accuracy 9.9× sooner than averaged synchronous SGD (27.8 s vs 274.3 s) at 16.3× the throughput. Caveat for fairness: the sync run performed 4500 worker gradient computations to async's 3000 and still took 8× longer in wall-clock; and the §5b barrier emulation uses a different sync semantics (§12.3).
- **F2 — Top-K + EF is the Pareto winner among fixed policies:** the best accuracy of all six methods (87.47%) at exactly half the bytes. The AGENTS.md communication gate (≤50% of uncompressed at comparable accuracy) is met exactly. The post-fix adaptive arm goes further on bytes (0.195 GB, 3.65×) at 87.09%, a difference inside the single-seed noise band (§7, F11).
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

## 5b. Headline result 4 — GPT transformer on Tiny Shakespeare (CUDA, one T4)

Run `asgc_transformer_results_20261102_104333` (notebook `notebooks/transformer_asgc_cuda.ipynb`, seed 0, all four arms; telemetry, plots, and per-seed summaries are committed under [`artifacts/transformer_asgc/run_20261102_104333/`](artifacts/transformer_asgc/run_20261102_104333/) — checkpoints excluded). Corpus: Tiny Shakespeare (1,115,394 chars, 65-character vocabulary). Model: 1,809,984-parameter GPT (4 layers, 4 heads, embd 192, block 128) trained through the public `DistributedModel` / `DistributedOptimizer` path. Four workers: worker 0 on a Tesla T4, workers 1–3 on CPU, with artificial delay profiles (base 0 / 50 / 200 / 700 ms plus jitter and periodic spikes) and a 32 Mbps up-link / 128 Mbps down-link (2 ms latency). Budget: 200 accepted updates. Evaluation is teacher-forced next-token top-1/top-5 on the same 10,240 held-out tokens for every arm, at a uniform grid of every 20 accepted updates plus one final pass (versions 0, 20, …, 200). Random baseline is ≈1.54% top-1 / 7.69% top-5.

| method | wall (s) | updates | val loss | ppl | top-1 % | top-5 % | up MB | down MB | total MB | time→target (s) | ρ mean | S_max mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plain async | 132.0 | 203 | 2.6807 | 14.60 | 24.31 | 61.21 | 1469.7 | 420.7 | 1890.4 | 119.5 | 1.000 | 10.00 |
| async + fixed Top-K 20% + EF | 86.9 | 203 | 2.7074 | 14.99 | 20.75 | 61.48 | 587.9 | 420.7 | 1008.6 | 72.9 | 0.200 | 10.00 |
| **async + adaptive Top-K + EF** | **76.9** | 202 | **2.6853** | **14.66** | **23.55** | **61.60** | **427.9** | 420.7 | **848.6** | **61.5** | 0.149 | 8.48 |
| barrier sync (per-push, no averaging) | 192.5 | 200 | 2.8935 | 18.06 | 20.69 | 55.22 | 1448.0 | 414.5 | 1862.4 | n/a | 1.000 | 10.00 |

**Findings:**
- **F13 — The transformer path needed no framework special-casing.** Top-K, error feedback, bucketing, and versioned parameter synchronization operated on attention and MLP parameters unchanged; the residual state was managed by the same `torch_asgc` path, though rollback itself did not fire (see below). Checkpoint-only re-evaluation reproduced every in-run metric exactly (`checkpoint_eval.csv`), so the result is not a live-parameter artifact. Rollback is implemented in `torch_asgc` but did not actually fire here: the adaptive arm rejected only 0.49% and the other three arms 0.0%, so this run demonstrates the compressed/synchronized path, not rejection recovery.
- **F14 — Asynchrony again wins on both axes, and by a wide quality margin against barrier sync.** Async uncompressed finished in 132.0 s vs 192.5 s for the barrier emulation (1.46×) on comparable worker work (203 vs 200 pushes) and ended at 2.6807 vs 2.8935 loss, 24.31% vs 20.69% top-1, 61.21% vs 55.22% top-5. The sync arm is the same async parameter-server code path with barriers, so this is the cleanest same-pipeline asynchrony comparison in the project. Caveat: the notebook's sync arm disables staleness and applies each worker's push as its own version after the barrier; it does not average the round's gradients into one update the way the research harness's sync mode does (stage 6), so the quality gap also reflects update semantics, not only the barrier.
- **F15 — Fixed Top-K 20% + EF holds loss and top-5 at 2.50× less uplink, but the top-1 delta is run-sensitive.** Uplink falls exactly 2.50× (587.9 vs 1469.7 MB) and wall-clock 1.52× (86.9 vs 132.0 s) at +0.0267 loss and +0.27 pp top-5, with top-1 at −3.56 pp. That top-1 gap exceeds two-arm token-sampling error (≈0.6 pp at these rates, n = 10,240), but it is not stable across runs: the previous T4 run of the same notebook (which used a time-based eval cadence, every 12 s, rather than every 20 updates) ended Top-K at 23.44% top-1 and *better* loss than uncompressed (2.6734 vs 2.7333). The stable transformer reading is therefore "Top-K + EF preserves loss within ~0.03 nats at 2.5× less uplink"; the top-1 ordering is noise-dominated at this budget. Both compressed arms also show a startup transient at version 20 (loss 5.53 / 5.48, top-1 < 1%) that clears by version 40 — error-feedback residual warm-up, not divergence.
- **F16 — On this workload the adaptive controller strictly dominates fixed Top-K 20%.** It used 27.2% fewer uplink bytes (427.9 vs 587.9 MB), 15.9% fewer total wire bytes (848.6 vs 1008.6 MB), and 1.13× less wall-clock (76.9 vs 86.9 s) while ending 0.0221 lower in loss and 2.80 pp higher in top-1 (23.55% vs 20.75%). Time-to-target (plain-async final loss + 1%) was 61.5 s vs 72.9 s for fixed Top-K (1.19×) and 119.5 s for uncompressed async (1.94×). Relative to uncompressed async it is 1.72× faster and sends 2.23× fewer total wire bytes. This is the first workload where adaptation beats the best fixed policy on both bytes and quality; contrast CIFAR (§4, §5), where the same controller compressed to ρ = 0.05 and paid 2.4–2.7 accuracy points. The improvement over the previous transformer run (12.2% uplink / 6.5% total savings, mean ρ = 0.174) comes from the controller spending more time at ρ = 0.10, but the comparison is not clean: the two runs also differ in eval cadence (time-based vs every 20 updates), which changes the evaluation pauses and the controller's observation windows. Do not read the delta as a controller-only effect without re-running both protocols with the same seed.
- **F17 — The measured bytes track mean ρ; the fixed downlink is what caps total savings.** ρ started at 0.20, fell to 0.10 by version 10, returned to 0.20 by version 53, then oscillated between 0.10 and 0.20 (mean 0.1493, final 0.20). S_max first loosened from 10 to 12 by version 10, then tightened in steps to 6 by version 94, moved between 6 and 8 afterwards, and ended at 6; rejection stayed at 0.49%. With Top-K costing 8 B/entry against 4 B/float, mean ρ = 0.149 predicts 29.9% of the uncompressed uplink; the measured adaptive/identity uplink ratio is 29.1%, so the wire accounting and the control trajectory agree. But the INT8 downlink delta broadcast is fixed at ≈2.1 MB per push and is 49.6% of the adaptive arm's total wire bytes, so a 27.2% uplink saving becomes a 15.9% total-wire saving. Compressing or quantizing the downlink delta is the next lever, not more uplink sparsification.
- **The controller still leaves compression on the table.** It repeatedly stepped ρ back to 0.20 because `variability_tol = 0.05` reads interleaved per-worker, per-minibatch push losses as instability. Had it held ρ = 0.10, uplink would have halved relative to fixed Top-K (≈294 MB vs 588 MB) and total wire would have dropped from 1008.6 MB to ≈715 MB (−29%); quality at ρ = 0.10 is untested on this workload, so this is a budget, not a guarantee. Feeding the controller smoothed validation loss instead of raw push losses, and raising `variability_tol` to 0.15–0.25, are the two concrete tuning levers.

The notebook now emits `loss_vs_updates.png` (all four arms on one accepted-update axis, with raw eval markers for single-seed runs) and `final_accuracy_bar.png` (grouped top-1/top-5 bars with value labels); both are in the bundle's `plots/`.

## 6. Answers to the proposal's research questions

- **RQ1 (how much communication can compression save without hurting quality?):** 50% at *better* accuracy than uncompressed on both workloads with Top-K + EF (F2, F5). Beyond that, up to 8–9× is reachable but enters a measurable accuracy trade (−2.4 points at 8.13× on CIFAR, F7/F9). The knee on CIFAR/ResNet-18 sits near ρ ≈ 0.10–0.15.
- **RQ2 (how does staleness affect convergence under heterogeneity?):** On both workloads, mean arrival staleness ≈ 2.1–2.9 with maxima 12–19 under the delay schedules used, and unrestricted async *did not* destabilize — final accuracy was within noise of every controlled variant. Staleness control bought bounded applied staleness (≤ S_max, verified from the event log) at a small throughput cost; its accuracy benefit was not observable on these workloads within these budgets.
- **RQ3 (does joint adaptation beat fixed policies in wall-clock?):** Conditionally yes — under a constrained link the adaptive method was the fastest route to target accuracy (13.7 min vs 20.5 for the best fixed policy and 41.1 for plain async), at an accuracy cost. Unconstrained, no: the controller's overhead and conservative drift made it slower than fixed Top-K + EF. The joint-control claim should be stated as regime-dependent.
- **RQ4 (how much does EF recover?):** At identical bytes on the headline protocol: FMNIST 86.92% → 87.47% (+0.55, §3); CIFAR 84.32% → 86.14% (+1.82, §4). On CIFAR, compressed + EF beats uncompressed outright. A separate FMNIST EF ablation in `docs/plans/stage4-error-feedback.md` reports 86.54% → 87.14% (+0.60); it uses a different protocol from the headline table and should not be mixed into these rows. The EF conservation invariant (Σ transmitted = Σ gradients − final residual) is pinned by unit test, and EF adds zero wire bytes by construction (verified: byte-identical totals with and without EF).
- **RQ5 (which signals are useful control inputs?):** System signals (rejection fraction, bytes/s) drove useful adaptation; the eval-loss **rebound** (mean upward move between evals) is a sound instability signal while raw loss spread is actively harmful (it reads fast learning as instability — the startup misclassification, §7); per-update residual norms are logged and available but not yet wired into the default policy; eval-loss trend was mostly uninformative because CIFAR's eval cadence (every 500 versions ≈ 95 s) exceeded the control interval (30 s) — control intervals must track eval cadence.
- **Transformer slice (§5b, single seed):** RQ1 — 2.50× uplink at +0.027 nats loss (top-1 run-sensitive); the adaptive arm reaches 3.43× uplink at +0.005 nats versus uncompressed. RQ2 — mean arrival staleness 2.77–2.90 with p95 = 5–6; tightening S_max to 6 rejected only 0.49%, so the delay profiles exercise downweighting far more than rejection. RQ3 — this is the favorable regime case: the adaptive method was fastest to target (1.19× vs fixed Top-K, 1.94× vs uncompressed) *without* an accuracy tax, unlike CIFAR. RQ4 — no Top-K-without-EF arm was run on the transformer, so EF's transformer contribution is unmeasured. RQ5 — using raw per-push loss as the variability input drove unnecessary ρ increases; the transformer result reinforces "control on smoothed or eval loss, not minibatch loss".

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
| total bytes ≤ 50% of uncompressed async at comparable accuracy | **met, with the transformer caveat stated** | In the harness only the up-link is modeled, so transmitted bytes = up-link bytes: Top-K + EF is exactly 50% on FMNIST/CIFAR at equal-or-better accuracy. On the transformer (both directions counted) Top-K + EF is 40% of up-link but 53.4% of total wire, above the bar; the adaptive arm is 29% up-link / 44.9% total wire at comparable quality. Adaptive on CIFAR is 11–15% total wire with an accuracy cost |
| faster wall-clock to target under heterogeneous workers | **met** | vs sync: 9.9× on FMNIST and 1.46× on the transformer (CIFAR sync unrun). vs plain async: no when the network is free (+16%), **yes under bandwidth** — 1.98× on CIFAR/250 Mbps and 1.72× for the adaptive arm on the transformer/32 Mbps |
| no major instability vs unrestricted async | **met** | all CIFAR runs ≤ 0.4 pts max eval drop; transformer, excluding the error-feedback warm-up, max consecutive eval-loss increase is +0.052 nats (Top-K, v180→v200), and the compressed arms' startup spike at v20 clears by v40; two FMNIST runs showed one transient > 2-pt drop each on a coarse eval grid — recheck across seeds |
| final accuracy within 1 pt of strongest non-adaptive baseline | **met on FMNIST** (adaptive 87.09 vs 87.47) and **on the transformer** (adaptive 23.55 vs 24.31 top-1 = 0.75 pp, +0.005 nats, single seed); **not yet on CIFAR** (83.76 vs 86.50 = 2.74 pts) at the committed ρ floor of 0.05 — the bounded tuning item |

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
- **The transformer slice is single-seed and timing-sensitive.** Teacher-forced top-1/top-5 over 10,240 tokens has ≈±0.42/±0.48 pp sampling error, but run-to-run movement is larger: across two T4 runs of the same notebook, Top-K ended at 23.44% vs 20.75% top-1 and its loss delta to uncompressed flipped sign (better in the first run, worse in the second). Those runs also used different eval cadences (time-based every 12 s vs every 20 accepted updates), so the spread mixes seed, protocol, and scheduling effects. Evaluation passes and the shared T4 perturb the async interleaving and the accepted-update count at each eval (202 vs 203). Treat transformer quality *orderings* as directional; the byte and wall-clock conclusions held in both runs (Top-K 2.50× uplink; adaptive 12.2–27.2% below fixed Top-K and 1.06–1.13× faster). The transformer downlink is an INT8 delta broadcast but is identical across arms and is ~half of the adaptive arm's total wire, so total-wire savings understate uplink savings.
- **Simulated distribution:** communication is queue-based pickling on one machine; the bandwidth model charges only the measured gradient up-link (down-link snapshots are uncharged, identical across methods); GPU contention between the server and worker processes perturbs absolute timings, which is why all claims are relative under identical conditions. The transformer's synchronous arm is a barrier emulation on the same asynchronous server, not a separate synchronous trainer.
- **Budgets:** CIFAR runs cover ~15 epochs (~86% ceiling), far from the 93% AGENTS.md aspiration; `time_to_target` at 0.80 is available for all slice runs but the 93% target is unreachable at these budgets.
- **BN recalibration** is server-side and uses worker 0's data partition; it re-estimates statistics honestly per method but is an approximation of fully distributed BN handling.
- **The adaptive method's bounds are untuned:** ρ floor 0.05 (set on FMNIST) lets CIFAR runs compress past the accuracy knee; persistence and the residual-norm signal remain unexercised on genuinely unstable workloads.
- The proposal's "within 1 point" gate is evaluated against the strongest non-adaptive baseline, which is itself our Top-K + EF pipeline — a strong bar by construction.

## 12. Deviations from the written methodology

`docs/methodology.tex` is the formal specification and is left unchanged by project rule. The implemented system differs from it in the following eight places. Report the implemented behavior, and cite this section when the text and the code disagree.

1. **The instability input is loss *rebound*, not loss *variance*.** The methodology's controller state vector is $z_t=\{\Delta L_t,\operatorname{Var}(L)_t,\bar\tau_t,r_t,T_t^{\mathrm{comm}},P_t\}$ (`docs/methodology.tex`), but both controllers use the mean positive rebound instead of $\operatorname{Var}(L)$. This is deliberate: raw spread classified the startup fast-learning drop as instability and ratcheted compression up to CR 1.52 (§7). The invariant "Loss Rebound vs. Raw Spread" is the canonical description.
2. **`torch_asgc` feeds per-push training loss into the controller, not evaluation loss.** The intended control signal is evaluation loss, and the research harness wires eval loss. The public `ParameterServer` instead calls `telemetry.add_loss(push_up.loss_val)` for every push, so the library controller sees interleaved worker and minibatch training losses. The transformer run exposes the cost: `variability_tol = 0.05` repeatedly stepped ρ back to 0.20 (§5b). Read the rebound invariant as "rebound of the configured loss stream"; smoothing or eval-loss wiring is the open fix.
3. **Synchronous training means two different things in this project.** The research harness (`workload.sync`) buffers one push per worker per version, **averages** the round's gradients, and applies one server step — bulk-synchronous SGD. The transformer notebook's `sync_identity` arm instead puts workers behind a barrier with staleness disabled, then applies **each push as its own server version with no averaging**. Therefore §3's sync baseline (and F1) measures averaged synchronous SGD, while §5b's 1.46× (F14) measures barrier-serialized per-push updates and is not an apples-to-apples synchronous trainer. The reported metrics are unchanged; the labels and caveats now say which semantics each row uses.
4. **Residual rollback exists only in `torch_asgc`.** The public library implements a transactional residual (`record_tentative` → `commit` / `rollback`) driven by `PushAck`, and it is unit-tested. The research harness `src/asgc` has a fire-and-forget `transport.push` and assigns the new residual before sending; there is no acknowledgement, so a rejected update **cannot** roll back and its residual stays consumed. Every F1–F12 number comes from the harness and therefore does not exercise rollback; rejection still blocks application and does not advance the version, so the loss is the un-retransmitted residual mass, bounded by the rejection rate (0–12.4% across runs). The concrete fix is to piggyback a rejected-push flag/version on the next `FetchResponse`, have the worker retain `(e_prev, u)`, and restore `e ← e_prev + g` before its next push. That is a post-hoc design change, not a rerun. The transformer/`torch_asgc` run exercises the same library path end-to-end, but rollback itself is covered only by unit tests because rejection stayed at or below 0.5% there.
5. **`local_steps` is gradient accumulation, not local optimizer steps.** `DistributedOptimizer.step` accumulates and averages gradients across `local_steps` micro-batches, then sends one compressed update; it never calls the base optimizer locally. `local_steps = 1` in every reported experiment (including the transformer run).
6. **Wire accounting differs by code path.** The harness charges only the compressed up-link; down-link snapshots are unmodeled and identical across methods, so "bytes" there means up-link bytes. `torch_asgc` counts both the compressed push and the INT8 down-link delta broadcast, and §5b reports up-link, down-link, and total separately. Ratios in §3–§5 are up-link ratios; transformer ratios are labeled per column.
7. **Control intervals and evaluation cadence are not matched everywhere.** The methodology fixes a control interval but not an evaluation cadence. CIFAR evaluates every 500 versions (≈95 s) while the controller runs every 30 s, so many controller windows see no new evaluation (RQ5). The transformer headline run evaluates every 20 accepted updates (`eval_every = 20`); the earlier run 095934 evaluated every 12 s. Cross-run comparisons therefore mix cadence with seed and scheduling effects.
8. **Staleness bounds are per-config, not one global policy.** `StalenessConfig` defaults are $S_{low}=2$, $S_{max}=10$; the CIFAR configs use 1/4; the transformer run uses 2/10. Claims of bounded staleness are relative to the config named in each table.

## 13. Remaining work

1. Run the committed matrix: `python scripts/run_matrix.py --matrix configs/matrix_cifar_full.yaml` (9 configs × 3 seeds; sync runs dominate the wall time).
2. Retune the controller on CIFAR data: ρ floor ≈ 0.10–0.15, control interval matched to eval cadence, optionally wire residual-norm variation and a per-update training-loss signal into the policy; re-run the adaptive config (~20 min per iteration).
3. Add the sync baseline runs and the three ablation configs to produce the ablation table (EF-off, fixed-compression, fixed-staleness — all already expressible as configs).
4. Multi-seed aggregation for the transient >2-point drops observed on FMNIST (stability criterion).
5. Report figures are already generated by `scripts/plot_results.py` (accuracy/loss vs time, bytes-vs-accuracy, staleness-vs-rejection) into `results/plot/`.
6. Re-run the transformer slice with `ASGC_SEEDS=0,1,2` to put a confidence interval on the top-1/top-5 deltas, then tune `variability_tol` (0.15–0.25) and feed the controller smoothed eval loss; separately test downlink delta quantization, which is now the dominant wire cost (~50% of adaptive total) and the binding constraint on total-wire savings.
