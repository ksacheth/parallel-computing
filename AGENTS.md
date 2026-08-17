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
