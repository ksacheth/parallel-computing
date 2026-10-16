# Stage 3 Implementation Plan — Top-K Sparsification and Quantization with Communication Logging

- **Branch:** `gradient-compression`
- **Status:** complete — gate green (Top-K ρ = 0.25 compresses at the predicted ratio and converges loosely without error feedback)
- **References:** [AGENTS.md](../../AGENTS.md) (testing policy, settled decisions), [docs/methodology.tex](../methodology.tex) (compression section, CR metric), [docs/proposal.tex](../proposal.tex) (QSGD / Deep Gradient Compression background)
- **Builds on:** [stage 1](stage1-parameter-server.md) (the `Compressor` protocol was designed so this stage is additive) and [stage 2](stage2-staleness-control.md).

## Scope

Two real compressors land behind the stage 1 protocol — Top-K sparsification with active fraction ρ = k/d, and QSGD-style quantization to a configurable bit width — plus the effective compression ratio in the run metrics. The protocol is reworked from "payload = list of tensors" to "payload = one piece per parameter" because sparse and quantized payloads carry indices/codes, not dense values.

**Wire formats (accounted exactly by `encoded_bytes`):**
- Dense (identity): float32 values.
- Top-K: float32 values + int32 flat indices → 8 bytes per kept entry; k = max(1, round(ρ·numel)) per tensor.
- Quantize: per-tensor float32 L2 scale + int8 codes on s = 2^(bits−1) − 1 levels with stochastic rounding (unbiased). Sub-byte packing is explicitly out of scope; accounting counts the int8 storage.

**Known and accepted consequence:** this stage runs compressors *without* error feedback (stage 4). Top-K without EF is expected to lose accuracy — that is the research narrative (stage 3 shows the cost, stage 4 recovers it), so the gate asserts loose convergence, not parity with identity.

**Baseline payoff:** "async + fixed Top-K" and "async + fixed quantization" become pure configs — four of the six proposal baselines now exist without new machinery.

## Steps

### Step 1 — payload pieces + protocol rework (`compression.py`, `server.py`, `worker.py`)
- [x] Piece types: `DensePiece(values)`, `SparsePiece(indices int32, values float32)`, `QuantizedPiece(codes int8, scale float32)`.
- [x] `decode(payload, like)` gains a `like` argument (server/worker pass their parameters) so sparse decode can restore shapes without shipping them.
- [x] `make_compressor(compression_config)` replaces the mode-string factory; identity migrates to `DensePiece`.
- [x] Update stage 1 tests to the new protocol.

*Verify:* identity roundtrip still exact; byte accounting unchanged for dense payloads. ✓

### Step 2 — `TopKCompressor(rho)`
- [x] Per-tensor top-k by absolute value, k = max(1, round(ρ·numel)); signed values at int32 flat indices.
- [x] Decode scatters values into zeros of the `like` shape (int32 → int64 upcast at decode).
- [x] `encoded_bytes` = Σ (4·k + 4·k); ρ = 0.25 on a d-element tensor yields exactly CR 2.0.

*Verify:* unit tests — exact selected set on a hand-built vector, zeros elsewhere on decode, byte formula, ρ bounds validation, ρ = 1 keeps everything. ✓

### Step 3 — `QuantizeCompressor(bits)`
- [x] s = 2^(bits−1) − 1 levels; per-tensor L2 scale; codes = stochastic-round(s·u/‖u‖₂) clamped to ±s, stored int8; zero tensors encode to zeros (no NaN).
- [x] Decode = codes/s·scale reshaped to `like` shape.
- [x] `encoded_bytes` = Σ (numel + 4).

*Verify:* unit tests — codes in range, elementwise error bounded by scale/s, zero-tensor safety, byte formula, bits bounds validation; seeded RNG via the worker's existing seeding. ✓

### Step 4 — config + compression-ratio metrics (`config.py`, `metrics.py`, `server.py`)
- [x] `CompressionConfig` gains `rho: float = 0.25` and `bits: int = 8`.
- [x] `RunEndEvent` gains `raw_bytes` (uncompressed equivalent: parameter bytes × arrivals) and `compression_ratio = raw_bytes / total_bytes` — the methodology's CR = B_raw/B_cmp including index overhead.
- [x] Run summary prints the ratio.

*Verify:* identity runs report CR 1.0; the gate run reports the expected Top-K ratio. ✓

### Step 5 — stage gate (`configs/smoke_fmnist_compression.yaml` + test)
- [x] Gate: Top-K ρ = 0.25, staleness disabled (isolate compression), 3000 updates.
- [x] Assertions: `compression_ratio ≥ 1.8`, test loss decreased, accuracy > 0.75 (loose — no error feedback yet by design).

*Verify:* full `pytest` green; stage 1 and 2 gates unchanged.

### Step 6 — docs
- [x] `AGENTS.md` layout line: compression modes implemented; EF residual, controller, ResNet-18 still placeholders.
- [x] Outcome section in this plan.

## Definition of Done

- `pytest tests/ -q` green: previous suites unchanged plus new unit tests and the compression gate.
- The gate reports CR ≈ 2.0 for Top-K ρ = 0.25 and converges loosely without error feedback.
- Unit tests pin both wire formats' byte accounting exactly.
- [x] Conventional commits per step, pushed to `origin/gradient-compression`.

## Outcome

- **The wire-format math held exactly.** The gate run reports CR = 2.000: Top-K ρ = 0.25 sends 332 MB where the uncompressed run sends 664 MB — halved communication for the same 3000-update workload, with index overhead honestly included.
- **Top-K without error feedback degrades far less than feared on this workload**: 86.5% final accuracy at 3000 updates versus ≈87.3% uncompressed — about 0.8 points, with the loss curve still decreasing. Stage 4 (error feedback) should close most of that gap and is what makes aggressive ρ viable.
- **`decode(payload, like)` proved the right protocol shape**: sparse decode restores shapes from the receiver's own parameters instead of shipping them, which keeps the wire format minimal and will let stage 4's worker-side residual reuse the same call.
- **Baseline count after this stage: four of six** exist as pure configs (plain async, fixed staleness, fixed Top-K, fixed quantization); only synchronous SGD and the full adaptive method need new machinery.

## Risks

- **Top-K without EF degrades accuracy** — expected and intended as the stage 3→4 narrative; gate thresholds set loose accordingly, and the gate config documents that EF (stage 4) is the recovery mechanism.
- **Stochastic rounding uses the global RNG** — workers already seed deterministically per worker id, so runs stay reproducible.
- **Index overhead can dominate at high ρ** — accounted honestly (int32); at ρ = 0.25 the ratio is 2.0, at ρ = 0.1 it is 5.0.
- **Suite runtime grows** (~20 min with three gates) — gates stay budget-sized per the stage 2 lesson; revisit if it becomes a burden.
