# Stage 4 Implementation Plan — Error Feedback

- **Branch:** `error-feedback`
- **Status:** complete — gate green; EF recovers +0.6 points accuracy at byte-identical cost (CR 2.000 in both runs)
- **References:** [AGENTS.md](../../AGENTS.md), [docs/methodology.tex](../methodology.tex) (error-feedback section: u = g + e, ĝ = D(C(u)), e ← u − ĝ), Karimireddy et al. 2019 in [docs/proposal.tex](../proposal.tex)
- **Builds on:** [stage 3](stage3-gradient-compression.md) — the `decode(payload, like)` protocol was shaped for exactly this stage.

## Scope

Workers carry their compression error forward: `u = g + e`, compress `u`, then `e ← u − ĝ` using the reconstruction the server will apply. Expected effect: close the ~0.8-point accuracy gap the stage 3 gate showed for Top-K ρ = 0.25, at **zero extra communication**.

**Design decision (reviewed and approved):** the worker decodes its own payload locally instead of the server acking the reconstruction. Our compressors are pure functions of (payload, shapes), so worker-side decode is exactly the server-side reconstruction — no extra message, honest byte metrics. The `Compressor` protocol gains a purity note to keep this true.

**Orthogonality note:** EF composes cleanly with staleness. If the server rejects an update, the worker has already folded the compression error into `e`; the rejected gradient itself is dropped by policy — EF carries compression loss, not policy loss. Standard semantics, documented.

**Explicitly deferred:** DGC extras (momentum correction, warm-up) — vanilla EF-SGD per the methodology. Quantize+EF runs as a stage 6 config.

## Steps

### Step 1 — config (`config.py`)
- [x] `ErrorFeedbackConfig(enabled: bool = False)` wired into `ExperimentConfig` and the loader; default off preserves every existing config.
- [x] Note that under identity compression EF is a mathematical no-op (ĝ = u ⇒ residual stays zero) — harmless, and pinned by a test.

*Verify:* parse test. ✓

### Step 2 — worker EF state (`worker.py`)
- [x] Init `residual = [zeros_like(p) for p in model.parameters()]` when enabled.
- [x] Per step: `u = g + e` → encode → decode locally against own parameters → `e ← u − ĝ`; EF work sits inside the worker's compute window.
- [x] Protocol purity note in `compression.py`: decode must remain a pure function of (payload, like).

*Verify:* conservation test plus gate. ✓

### Step 3 — metrics (`transport.py`, `metrics.py`, `server.py`)
- [x] `PushUpdate` and `UpdateEvent` gain `residual_norm: float` (0.0 when disabled) — the controller's gradient-norm-variation signal, landed early for stage 5.

*Verify:* appears in the gate's event log. ✓ (mean ‖e‖ ≈ 0.30, max 1.58 across the gate run)

### Step 4 — conservation tests (`tests/test_error_feedback.py`)
- [x] Telescoping invariant over 50 Top-K steps: **Σ ĝ = Σ g − e_final** — total received mass equals total computed mass minus the residual.
- [x] Identity compressor keeps e exactly zero (EF no-op proof).
- [x] EF never changes `encoded_bytes`.

*Verify:* unit tests green. ✓

### Step 5 — stage gate (`configs/smoke_fmnist_topk_ef.yaml` + test)
- [x] Same as the stage 3 gate (Top-K ρ = 0.25, staleness disabled, 3000 updates) with `error_feedback.enabled: true`.
- [x] Assertions: compression_ratio ≥ 1.8 (same bytes as no-EF), accuracy > 0.84 (versus > 0.75 no-EF; expected ≈ 87%), loss decreased, residual norms present in the log.
- [x] Stage 3 no-EF gate stays unchanged as the regression baseline; a same-seed A/B is reported manually in the Outcome, not asserted (flaky cross-run comparison).

*Verify:* full suite green (27 tests). ✓

### Step 6 — docs
- [x] `AGENTS.md` layout line: EF implemented; controller and ResNet-18 remain.
- [x] Outcome section appended here.

## Outcome

- **Measured effect of EF (same seed, 3000 updates, Top-K ρ = 0.25):** accuracy 86.54% → 87.14% (+0.6 points), loss 0.380 → 0.359, bytes byte-for-byte identical (332,016,000, CR 2.000). Against the ≈87.3% uncompressed reference at the same budget, EF-on-compression now sits within ~0.2 points at half the bytes — the proposal's core tradeoff, working.
- **The first draft of the conservation test asserted the wrong invariant** — Σd = Σu − e_final, which fails by simple algebra (intermediate u's don't telescope). The correct telescoping is d_t = g_t + e_{t−1} − e_t ⇒ **Σd = Σg − e_final**. The implementation was correct; the test now pins the right identity, and the test failure was a useful reminder that invariants deserve a second look when they fail with small residuals rather than large ones.
- **`residual_norm` is live in the event stream** (mean 0.30, max 1.58 in the gate run), giving stage 5's controller its gradient-variation signal without any schema churn later.
- **Suite now runs 27 tests in ~17–25 minutes** (four gates); still acceptable, revisit at stage 6.

## Definition of Done

- `pytest tests/ -q` green: conservation tests, the new gate, and all three prior gates.
- The EF gate shows CR ≈ 2.0 with accuracy measurably above the no-EF gate's threshold band.
- `residual_norm` present in event logs when EF is on.
- Conventional commits per step, pushed to `origin/error-feedback`.

## Risks

- **Worker-side decode drifting from server-side decode** — only possible if a compressor becomes stateful; the purity note plus the shared decode path guard it.
- **EF + rejection semantics** — intentional and documented above.
- **Suite runtime** — a fourth gate pushes the suite toward ~25 minutes; accepted for now, revisited with the stage 6 experiment matrix.
- **Residual memory** — one extra model copy per worker (220 KB here, ~44 MB per ResNet-18 worker later); noted, not a problem.
