# ASGC Domain Context

Adaptive Staleness-Aware Gradient Compression (ASGC) for asynchronous distributed deep learning, balancing communication efficiency, straggler resilience, and model convergence.

## Language

### Core Distribution & Staleness

**Server Version**:
The monotonically increasing counter incremented by the parameter server each time an accepted worker update is applied to the global model parameters.
_Avoid_: Global clock, global step, iteration count

**Worker Version**:
The snapshot version of the global model parameters that a worker used to compute a mini-batch gradient.
_Avoid_: Local clock, fetch version, base version

**Staleness**:
The integer age difference between the server version upon update arrival and the worker version the update was computed from.
_Avoid_: Delay, lag, skew

**Bounded Staleness**:
The admission and weighting policy that partitions incoming worker updates into accepted, downweighted, or rejected categories based on staleness thresholds.
_Avoid_: Staleness bounding, lag filter, bounded delay

**Staleness Attenuation**:
The multiplicative weighting factor applied to scale down stale updates within the soft staleness window.
_Avoid_: Staleness damping, lag penalty, gradient discounting

**Soft Staleness Bound**:
The maximum staleness threshold at or below which an incoming worker update is applied at full weight.
_Avoid_: Lower threshold, slack limit, safe bound

**Hard Staleness Bound**:
The maximum staleness threshold beyond which an incoming worker update is rejected and discarded by the server.
_Avoid_: Staleness ceiling, upper cutoff, drop threshold

**Straggler**:
A distributed worker process that computes or communicates significantly slower than peer workers due to system or hardware heterogeneity.
_Avoid_: Slow worker, delayed node, outlier worker

### Compression & Error Feedback

**Gradient Compression**:
The transformation applied to a gradient tensor before transmission to reduce its wire byte footprint.
_Avoid_: Gradient reduction, network throttling, payload shrinking

**Top-K Sparsification**:
The gradient compression operator that transmits only the components with the largest absolute magnitudes along with their indices.
_Avoid_: Magnitude pruning, thresholding, sparse masking

**Active Fraction**:
The proportion of parameter indices retained during Top-K sparsification relative to the total parameter count.
_Avoid_: Density, sparsity rate, compression factor

**Quantization**:
The gradient compression operator that maps floating-point gradient elements to discrete low-bit integer levels.
_Avoid_: Discretization, bit packing, precision reduction

**Error Feedback**:
The worker-side mechanism that accumulates uncompressed gradient residual mass into subsequent mini-batch updates to ensure zero asymptotic information loss.
_Avoid_: Residual memory, error compensation, gradient accumulation

**Residual**:
The worker-local tensor tracking the difference between pre-compression accumulated gradients and transmitted compressed updates.
_Avoid_: Error buffer, velocity, gradient leftover

**Residual Rollback**:
The worker-side restoration of pre-transmission residual state when an update is rejected by the server, preserving gradient mass across dropped updates.
_Avoid_: Residual undo, residual recovery, error unrolling

### Adaptive Control & Telemetry

**Adaptive Controller**:
The server-side control module that periodically adjusts compression active fraction and staleness bounds based on live training telemetry.
_Avoid_: Hyperparameter scheduler, auto-tuner, dynamic coordinator

**Loss Rebound**:
The average upward variation in evaluation loss between consecutive evaluation checkpoints, used as the primary detector for training instability.
_Avoid_: Loss spike, loss variance, instability metric

**Control Interval**:
The evaluation-version period between successive controller evaluation and parameter adjustment steps.
_Avoid_: Tuning frequency, controller cadence, observation window

### Network & Synchronization

**Downlink Delta**:
A compressed parameter server broadcast transmitting only the parameter differences accumulated since a worker's last fetch.
_Avoid_: Model diff, weight patch, broadcast delta

**Gradient Bucketing**:
The grouping of individual parameter gradients into contiguous memory buffers prior to compression and transmission.
_Avoid_: Tensor fusing, buffer batching, gradient chunking

**Micro-Local Steps**:
The execution of multiple local forward and backward mini-batch optimization steps on a worker between parameter server synchronization rounds.
_Avoid_: Local iterations, inner steps, multi-batch accumulation
