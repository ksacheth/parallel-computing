# Slide 2: Dataset & Workload Setup — Technical Explanation and 5-Minute Spoken Script

**Project Title:** Adaptive Staleness-Aware Gradient Compression for Asynchronous Distributed Training  
**Course:** Parallel Computing (Dept. of Information Technology, NITK Surathkal)  
**Slide Reference:** Slide 2 — `Dataset` (Primary Benchmark, Auxiliary Workload, Distributed Setup, and Snapshot)

---

## Part 1: Comprehensive Technical Explanation of Slide 2

This slide addresses a fundamental question in distributed machine learning and parallel computing: **how do we construct a rigorous, reproducible, and realistic experimental testbed that stresses communication bandwidth and worker synchronization?**

In distributed data-parallel training, datasets are not merely learning tasks—they dictate the **communication-to-computation ratio ($T_{\text{comm}} / T_{\text{comp}}$)**, the memory footprint, and the gradient vector dimensionality that must traverse the physical or emulated interconnect at every iteration.

```
+--------------------------------------------------------------------------------------------------+
|                                  DISTRIBUTED WORKLOAD ARCHITECTURE                               |
|                                                                                                  |
|   +------------------------------------+             +---------------------------------------+   |
|   |    PRIMARY BENCHMARK (CIFAR-10)    |             |      AUXILIARY TESTBED (FMNIST)       |   |
|   |  - 60,000 images (50k train / 10k) |             |  - 70,000 images (60k train / 10k)   |   |
|   |  - 32x32x3 RGB (3,072 features)    |             |  - 28x28 grayscale (784 features)    |   |
|   |  - ResNet-18 (11.17M parameters)   |             |  - 2-Layer CNN (55k parameters)      |   |
|   |  - Raw Gradient: ~44.7 MB / step   |             |  - Raw Gradient: ~221 KB / step      |   |
|   |  --> Primary Communication Stress  |             |  --> Rapid Verification & Unit Tests |   |
|   +-----------------+------------------+             +-------------------+-------------------+   |
|                     |                                                    |                       |
|                     +------------------------+---------------------------+                       |
|                                              |                                                   |
|                                              v                                                   |
|                         +-----------------------------------------+                              |
|                         |  DISTRIBUTED ASYNCHRONOUS PARTITIONING  |                              |
|                         |  - Mini-batch B = 128 per worker       |                              |
|                         |  - Zero barrier synchronization         |                              |
|                         |  - g_i = \nabla L(\theta_{v_i}; B_i)    |                              |
|                         +--------------------+--------------------+                              |
|                                              |                                                   |
|                                              v                                                   |
|                         +-----------------------------------------+                              |
|                         |     CONTROLLED SYSTEM HETEROGENEITY     |                              |
|                         |  - Worker 0: Baseline (+0.0s)           |                              |
|                         |  - Worker 1: Straggler (+0.5s delay)    |                              |
|                         |  - Worker 2: Severe Straggler (+1.5s)   |                              |
|                         |  - Bandwidth Cap: 250 Mbps Up-Link      |                              |
|                         +-----------------------------------------+                              |
+--------------------------------------------------------------------------------------------------+
```

---

### 1. Primary Benchmark: CIFAR-10 & ResNet-18

#### A. Dataset Characteristics
- **Instance Count:** 60,000 $32 \times 32$ color images partitioned into 50,000 training examples and 10,000 validation/test examples.
- **Class Balance:** Exactly 10 mutually exclusive, balanced classes (6,000 images per class): *Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck*.
- **Feature Space:** $32 \times 32 \times 3$ channels $= 3,072$ numerical input features per instance. Preprocessing applies channel-wise mean and standard deviation normalization ($\mu, \sigma$), alongside standard data augmentations (random horizontal flip and 4-pixel random crop padding).

#### B. Workload Model: ResNet-18
- **Model Topology:** A deep convolutional residual network (He et al., 2016) tailored for CIFAR-10 (using a $3 \times 3$ initial convolution instead of the $7 \times 7$ kernel used for ImageNet, and omitting the max-pooling layer to preserve low-resolution spatial feature maps).
- **Parameter Scale:** $11,173,962$ trainable parameters ($\approx 11.17\text{ million}$).
- **Gradient Payload Footprint:** 
  $$\text{Payload} = 11,173,962 \text{ float32 values} \times 4 \text{ bytes} \approx 44,695,848 \text{ bytes} \approx 44.70\text{ MB}.$$
- **Why This Matters for Parallel Computing:** 
  In data-parallel distributed training, each worker must transmit this $44.7\text{ MB}$ gradient vector to the parameter server after every single forward-backward pass. In a 3-worker cluster processing 6,000 asynchronous updates, uncompressed communication transmits over **$268\text{ Gigabytes}$** of raw gradient data. Over standard or constrained interconnects (e.g., 250 Mbps up-link), gradient transmission time dwarfs local backpropagation time ($T_{\text{comm}} \gg T_{\text{comp}}$). This makes CIFAR-10 with ResNet-18 an ideal stress workload for testing gradient compression (Top-$K$ sparsification and quantization).

---

### 2. Auxiliary Workload: Fashion-MNIST & 2-Layer CNN

#### A. Dataset Characteristics
- **Instance Count:** 70,000 $28 \times 28$ grayscale images (60,000 training instances and 10,000 test instances across 10 clothing categories).
- **Feature Space:** $28 \times 28 \times 1 = 784$ input features per instance, normalized to $[0, 1]$.

#### B. Workload Model: Small 2-Layer CNN
- **Topology:** Two $3 \times 3$ convolutional layers followed by max-pooling, ReLU activations, and a fully connected linear classification head.
- **Parameter Scale:** $55,338$ trainable parameters.
- **Gradient Payload Footprint:**
  $$\text{Payload} = 55,338 \times 4 \text{ bytes} \approx 221.35\text{ KB}.$$
- **Engineering Role in the Project:**
  Fashion-MNIST serves as our low-latency, rapid-iteration debugging testbed. Because each gradient is only $\approx 221\text{ KB}$, training runs complete in tens of seconds rather than hours. This allowed us to execute thousands of sanity checks, verify the error-feedback conservation invariant ($\sum \text{transmitted} = \sum \text{gradients} - \text{final residual}$), validate race-free version counter increments on the parameter server, and diagnose controller instability before launching intensive CIFAR-10 training runs.

---

### 3. Distributed Mini-Batch Partitioning & Asynchronous Mechanics

- **Mini-Batch Sizing ($B = 128$):** Each worker independently samples a mini-batch of size $B = 128$ from its assigned data slice.
- **Zero Synchronization Barrier:** Unlike synchronous SGD (which halts all workers at a barrier after each iteration until the slowest worker finishes), our workers operate completely asynchronously.
- **Stochastic Gradient Generation:**
  Worker $i$ computes:
  $$g_i = \nabla_\theta \mathcal{L}(\theta_{v_i}; B_i)$$
  where $\theta_{v_i}$ is the local model weights fetched from the parameter server at server version $v_i$.
- **Independent Execution:** Faster workers immediately fetch the latest available parameter weights, compute gradients on a fresh batch, and transmit updates without waiting for their peers.

---

### 4. Emulated Heterogeneity & Straggler Simulation

A core failure mode in real-world distributed clusters is **worker heterogeneity**—caused by differences in GPU/CPU compute capability, multi-tenant resource contention, thermal throttling, or uneven network hops. In our single-machine simulated distributed framework (where workers execute as isolated spawned OS processes exchanging messages over transport queues), we explicitly inject controlled heterogeneity:

1. **Injected Compute Delays:**
   - **Worker 0:** Baseline speed ($+0.0\text{s}$ injected delay). Emulates a high-performance compute node.
   - **Worker 1:** Mild straggler ($+0.5\text{s}$ delay per iteration).
   - **Worker 2:** Severe straggler ($+1.5\text{s}$ delay per iteration). Emulates an overloaded or edge node.
2. **Network Bandwidth Throttling (250 Mbps):**
   - Each update transmission incurs an emulated transmission delay:
     $$T_{\text{comm}} = \frac{\text{payload\_bytes}}{\text{bandwidth}}$$
   - This charges workers proportionally to their serialized gradient message size, directly rewarding aggressive compression under bandwidth congestion.
3. **The Consequence — Staleness Generation ($\tau$):**
   Because Worker 2 is delayed by $1.5\text{s}$, by the time its gradient arrives at the parameter server, Worker 0 has already submitted multiple updates. The parameter server version has advanced from $v_i$ to $v_s$. The update staleness:
   $$\tau_i = v_s - v_i$$
   frequently reaches values between $5$ and $15$. This controlled environment provides the exact empirical stress test required to validate our server's **tri-zone staleness policy** (accept, downweight, reject).

---

### 5. Visual Snapshot Analysis

The right-hand panel displays a visual grid of CIFAR-10 sample images across all 10 target classes (4 samples per class). This visualization confirms:
- **Intra-Class Variance:** High variance in camera angles, object scale, backgrounds, and lighting (e.g., ships in different sea states, various dog breeds).
- **Input Resolution & Complexity:** The low resolution ($32 \times 32$) makes feature extraction non-trivial, requiring deep residual convolutional representations (ResNet-18) rather than simple linear classifiers.
- **Supervised Classification Integrity:** Balanced representation across classes ensures that gradients are not skewed toward majority classes, making optimization dynamics strictly dependent on learning rate, compression, and update staleness.

---
---

## Part 2: 5-Minute Spoken Presentation Script

*Spoken delivery guide:*
- **Total Duration:** 5 minutes (0:00 to 5:00)
- **Target Pace:** 130–140 words per minute (Total words: ~680–720 words)
- **Stage Directions:** Formatting in brackets `[like this]` indicates body language, pacing, vocal emphasis, and slide interactions.

---

### [0:00 – 1:00] Minute 1: The Parallel Computing Motivation & Problem Framing

`[Stage direction: Stand tall, make direct eye contact with the evaluators, open hand gesture toward the screen.]`

Good morning, respected professors and fellow classmates. 

Moving into the core of our methodology, this slide presents our **experimental testbed, dataset design, and distributed workload partitioning**.

`[Pacing: Deliberate and engaging]`

When evaluating a distributed machine learning framework in a parallel computing course, it is crucial to recognize that a dataset is not just a collection of images for classification. **In parallel computing, a dataset and model architecture define a communication workload.** They determine the exact communication-to-computation ratio, the volume of data traversing the network, and the severity of worker synchronization bottlenecks.

To rigorously benchmark our proposed adaptive framework against stragglers and communication bottlenecks, we needed an experimental setup with three key properties:
First, a workload heavy enough that gradient communication becomes a genuine bottleneck;
Second, a lightweight benchmark for rapid testing and invariant validation; and
Third, a controlled way to simulate hardware heterogeneity and network congestion.

Let’s look at how we established these two workloads.

---

### [1:00 – 2:00] Minute 2: Primary Benchmark — CIFAR-10 and the 44.7 MB Gradient Bottleneck

`[Stage direction: Step toward the top-left card on the slide, point to "Primary Benchmark: CIFAR-10".]`

Our primary benchmark is **CIFAR-10**, paired with the **ResNet-18** deep residual architecture.

`[Tone: Technical and precise]`

As shown in the top card, CIFAR-10 consists of 60,000 $32 \times 32$ RGB images evenly distributed across 10 balanced classes—50,000 images for training and 10,000 for evaluation. Each individual image represents 3,072 numerical input features after per-channel standardization.

Now, why did we choose ResNet-18 as our primary workload model? 

`[Pause for 1 second for rhetorical emphasis]`

ResNet-18 contains **11.17 million trainable parameters**. In data-parallel distributed training, every worker computes the loss gradient across its mini-batch and must serialize and push those gradients back to the parameter server. 

At 4 bytes per 32-bit float, an 11.17-million parameter model produces an **uncompressed gradient vector of approximately 44.7 Megabytes per single step**. 

In an asynchronous training run of 6,000 updates, an uncompressed system transmits over **268 Gigabytes of gradient data**. On shared networks or commodity bandwidth, transmitting 44.7 Megabytes per worker update creates severe network saturation. This provides the exact communication pressure required to evaluate whether our Top-$K$ sparsification and quantization techniques can successfully reduce communication by 50% to 90% without destroying accuracy.

---

### [2:00 – 3:00] Minute 3: Auxiliary Workload — Fashion-MNIST as an Engineering Sandbox

`[Stage direction: Shift focus to the bottom-left card, gesturing downward.]`

Now, looking at the second box on the lower left: before deploying multi-hour ResNet-18 runs, we developed an auxiliary workload using **Fashion-MNIST** and a custom **2-layer convolutional network**.

Fashion-MNIST consists of 70,000 $28 \times 28$ grayscale images across 10 clothing categories, with 784 input features. Coupled with our lightweight 2-layer CNN, the model has just **55,338 parameters**, translating to a gradient payload of only **221 Kilobytes**.

`[Tone: Warm, practical, sharing engineering insight]`

From an engineering and systems standpoint, this auxiliary benchmark was invaluable. Because runs complete in under 40 seconds, it allowed us to:
- Verify our mathematical error-feedback conservation invariant—confirming that no gradient information is lost when sparsifying updates;
- Prove that our server-side version counter increments without race conditions; and
- Catch controller edge cases—such as distinguishing rapid early optimization from genuine instability—before committing long GPU cycles to CIFAR-10.

---

### [3:00 – 4:00] Minute 4: Distributed Partitioning & Simulating Real-World Stragglers

`[Stage direction: Emphasize the lower bullet points on the left card: Mini-batch Partitioning and Emulated Heterogeneity.]`

Let us now focus on the parallel computing execution model.

We partition data at **mini-batch granularity with a batch size of $B = 128$** across three worker processes. Crucially, our system operates with **zero synchronization barriers**. 

`[Tone: Firm and confident]`

In traditional synchronous SGD, the system is only as fast as its slowest worker. Fast workers sit completely idle waiting at a barrier. In our asynchronous architecture, each worker independently computes its stochastic gradient $g_i = \nabla \mathcal{L}(\theta_{v_i}; B_i)$ using its local copy of model version $v_i$, transmits its update, and immediately moves to the next mini-batch.

To evaluate how our parameter server handles stragglers, we introduce **controlled artificial heterogeneity**:
- **Worker 0** operates at full baseline speed;
- **Worker 1** has a $0.5\text{-second}$ synthetic compute delay per iteration; and
- **Worker 2** has a severe $1.5\text{-second}$ compute delay.
- Furthermore, we enforce an up-link bandwidth cap of **250 Mbps**, charging each push message a delay directly proportional to its transmitted byte size.

What happens as a result? 

While Worker 2 is delayed computing a gradient, the parameter server version advances as faster workers commit updates. When Worker 2 finally submits its gradient, its staleness $\tau = v_s - v_i$ often reaches 5 to 15 versions old! 

This artificial delay schedule reproduces real-world cloud conditions, directly testing whether our parameter server can downweight or reject stale gradients without collapsing system throughput.

---

### [4:00 – 5:00] Minute 5: Snapshot & Transition to Methodology Block Diagram

`[Stage direction: Turn and gesture toward the right-hand panel displaying the CIFAR-10 image grid.]`

Finally, looking at the right panel of the slide, we have provided a visual snapshot of the CIFAR-10 dataset across all 10 balanced classes—from airplanes and automobiles to ships and trucks.

`[Pacing: Smooth conclusion leading into the next slide]`

Notice the significant variation in lighting, orientation, and complex backgrounds at just $32 \times 32$ pixels. This visual complexity confirms that the optimization landscape is non-convex and non-trivial. Stale or heavily compressed gradients could easily destabilize training if left unmanaged.

To summarize:
1. **CIFAR-10 with ResNet-18** supplies the heavy 44.7 MB gradient payload that stresses network communication;
2. **Fashion-MNIST** provides the rapid verification testbed for algorithmic correctness; and
3. Our **injected delay schedule and 250 Mbps bandwidth constraint** simulate the real-world stragglers that produce stale updates.

With this rigorous data and distributed setup established, how does our system actually process, compress, and regulate these gradients? 

In the next slide, we will walk through the **Methodology Block Diagram**, explaining our 5-phase pipeline, error-feedback mechanics, and runtime adaptive controller.

Thank you. `[Nod toward the evaluators, transition smoothly to Slide 3.]`

---
---

## Part 3: Anticipated Questions & Defense Strategy

When presenting this slide, evaluators typically probe the distributed data setup. Below are the key anticipated questions and precise defense answers:

### Q1: "Why did you use simulated heterogeneity on a single machine rather than running across 3 physical machines?"
**Defense:**
> "That was a deliberate design decision grounded in experimental control. In a physical multi-machine cluster, background operating system tasks, switch buffering, and temperature variations introduce uncontrollable stochastic noise. By running workers as isolated spawned processes with explicit delay injection ($+0.5\text{s}, +1.5\text{s}$) and exact bandwidth charging (250 Mbps), we ensure that every single baseline—synchronous SGD, plain async, and our adaptive method—experiences the exact same delay schedule and seed. This guarantees that observed convergence gains stem strictly from our algorithmic strategy rather than random network jitter."

### Q2: "How do you ensure asynchronous workers don't repeatedly train on the exact same mini-batches?"
**Defense:**
> "Data partitioning is handled at mini-batch granularity through worker-indexed non-overlapping subsets. Each worker independently advances through its dedicated partition of the dataset. Because the batch size is $B = 128$, workers sample distinct gradient trajectories, preventing redundant computation while maintaining statistically unbiased stochastic gradient estimates across the global training distribution."

### Q3: "Why is CIFAR-10 sufficient? Why not ImageNet?"
**Defense:**
> "CIFAR-10 with ResNet-18 has 11.17 million parameters, generating a 44.7 MB gradient vector per iteration. In an asynchronous cluster running 6,000 updates, that already totals over 268 GB of transmitted wire traffic. In parallel computing research, the primary objective is to stress the $T_{\text{comm}} / T_{\text{comp}}$ ratio. Because ResNet-18 on CIFAR-10 has a relatively short forward-backward compute time compared to its 44.7 MB transfer time, it produces an exceptionally high communication bottleneck—making it an ideal benchmark to demonstrate the impact of gradient compression and staleness control without requiring weeks of cloud compute."
