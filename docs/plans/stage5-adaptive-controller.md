# Stage 5 Implementation Plan — The Adaptive Controller

- **Branch:** `adaptive-controller`
- **Status:** complete — gate green; the controller compressed ρ 0.25 → 0.10 and relaxed S_max 4 → 7 while stable, then held for 8 intervals once pressure was relieved; final CR 3.91 at 86.9% accuracy
- **References:** [AGENTS.md](../../AGENTS.md), [docs/methodology.tex](../methodology.tex) (joint runtime adaptation, z_t and c_{t+1}), [docs/proposal.tex](../proposal.tex) (score R = aL + bV + cC + dT)
- **Builds on:** stages 1–4. The controller consumes only the event stream; every signal it needs (staleness, decisions, bytes, residual norms, eval losses) is already logged.

## Scope

The server periodically adapts **compression strength** and **S_max** together: aggressive when training is stable, safer when instability appears, bounded per interval, every decision logged. Control vector per the methodology: c = (compression, τ_max) — `s_low` and β stay fixed.

**Design calls (reviewed and approved):**
1. **Control channel = fetch reply.** `FetchResponse` carries the current compression setting; workers rebuild their compressor when it changes. No new plumbing; in a real system this is the control-message path.
2. **Prerequisite: settings-free decode.** The quantizer folds `1/s` into the encoded scale so decode never depends on the encoder's bits (else adapted settings would race with in-flight payloads). Top-K decode was already settings-free.
3. **Mode-generic "compression strength":** ρ for Top-K (bytes adapt), bits for quantize (variance adapts; unpacked int8 wire size does not — honestly accounted).

**Explicitly deferred:** adaptive `s_low`/β, tuning of thresholds (stage 6 — the methodology defers them), DGC extras.

## Steps

### Step 1 — quantizer scale-encoding fix (`compression.py`, stage 3 tests)
- [x] Encode stores `scale' = ‖u‖₂ / s`; decode becomes `codes × scale'` — settings-free.
- [x] Update the two affected stage 3 unit tests (error bound reads scale' directly).

*Verify:* compression unit tests green; byte counts unchanged. ✓

### Step 2 — config (`config.py`)
- [x] `ControllerConfig(enabled=False, interval_s, policy, worsen_tol, variability_tol, reject_frac_max, comm_pressure_min, score_weights a–d, rho_bounds, bits_bounds, s_max_bounds, max_delta_rho/bits/s_max)`. Disabled by default — all prior configs unchanged.

*Verify:* parse test. ✓

### Step 3 — `controller.py` (pure functions)
- [x] `WindowStats` + `compute_window_stats(events)`: loss trend/variability from eval events, mean arrival staleness, rejected fraction, bytes/sec, updates/sec, residual-norm variance.
- [x] `ThresholdPolicy`: instability (trend, variability, or rejections over tolerance) → safer (strength ↑, S_max ↓); stable + communication pressure ≥ threshold → aggressive (strength ↓, S_max ↑); otherwise hold.
- [x] `ScorePolicy`: R = a·trend + b·variability + c·comm + d·staleness on normalized terms with a deadband; R > 0 → safer, R < 0 → aggressive.
- [x] All decisions: per-interval deltas bounded, hard bounds clamped.

*Verify:* unit tests on synthetic windows — direction, bounded deltas, clamping, hold, weight effects. ✓

### Step 4 — server + worker integration
- [x] Server keeps an in-memory window of update/eval events; every `interval_s` runs the policy, applies the new (strength, S_max) (staleness via a replaced `StalenessConfig`, strength into the compressor), logs a `ControlEvent` with full window stats and old→new values + reason, resets the window.
- [x] `FetchResponse` gains `compression: CompressionConfig | None`; workers rebuild their compressor when it differs (EF residual unaffected).

*Verify:* gate. ✓

### Step 5 — stage gate (`configs/smoke_fmnist_adaptive.yaml` + `tests/test_controller.py`)
- [x] Gate: Top-K ρ = 0.25 + EF + staleness on, worker 1 delayed 0.3 s, threshold policy, interval 15 s, 2500 updates, eval every 250.
- [x] Assertions: ≥ 5 control events; every (strength, S_max) within hard bounds; consecutive deltas within max_delta; strength actually adapted (range > 0); accuracy > 0.80; loss decreased.

*Verify:* full suite green (35 tests). ✓

### Step 6 — docs
- [x] `AGENTS.md` layout line: controller implemented; only sync SGD and ResNet-18 remain (stage 6).
- [x] Outcome section here.

## Outcome

- **The gate run produced a textbook adaptive trace.** With stable training and 2.5 MB/s of traffic, the controller compressed in bounded steps (ρ 0.25 → 0.20 → 0.15 → 0.10) while relaxing S_max (4 → 5 → 6 → 7) over the first 45 s; once bytes/s fell to ~750 kB/s (below `comm_pressure_min`), it held for all 8 remaining intervals — the anti-oscillation design working as intended, zero thrash.
- **The adaptive run reached CR 3.91 at 86.9% accuracy** (2500 updates, 296 rejections) — better compression than the fixed ρ = 0.25 run's 2.0× at comparable quality (fixed+EF: 87.1%), because EF absorbed the added error of the deeper compression the controller dared to use.
- **A real sign bug was caught by the unit tests before any run:** the score policy's communication term was initially oriented so high pressure pushed *safer* instead of aggressive, contradicting the methodology. Fixed to `comm = 1 − bytes/s ÷ threshold` and pinned by the direction test.
- **All five controller signals now flow from the event stream alone** — including residual-norm variance, logged since stage 4 for exactly this purpose — and every decision is auditable via `control` events (stats + old→new + reason).

## Definition of Done

- `pytest tests/ -q` green across all five gates.
- The adaptive gate demonstrates real adaptation: multiple logged, bounded, in-bounds decisions with the strength actually moving, while still converging.
- Every controller decision fully auditable from the event log (stats + old/new + reason).
- Conventional commits per step, pushed to `origin/adaptive-controller`.

## Risks

- **Threshold defaults are guesses** until stage 6 tuning — the methodology explicitly defers them; the hold region and bounded deltas prevent thrash.
- **Loss signal coarseness** — windows without two evals report zero trend and lean on staleness/rejection/bytes signals; interval ≥ eval cadence documented.
- **Suite runtime** — fifth gate; adaptive gate sized at 2500 updates.
- **In-flight payload race on adapted settings** — eliminated by the settings-free decode fix (step 1).
