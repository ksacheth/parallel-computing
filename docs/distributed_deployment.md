# Multi-machine deployment: cloud parameter server + Colab workers

This guide runs `torch_asgc` as a real distributed system. One **Parameter Server** runs on a
cloud VM (here, an Oracle Cloud Ampere A1 ARM instance). Any number of **workers** connect to it
over the public internet from separate Google Colab sessions, which can use different Google accounts
and different runtimes. Mixing T4 GPU and CPU runtimes gives natural worker heterogeneity, the
straggler setting ASGC targets.

```
 Colab #0 (T4, account A) ─┐
 Colab #1 (T4, account B) ─┼── TCP 29500, HMAC-authenticated ──> Oracle ARM VM: scripts/ps_server.py
 Colab #2 (CPU, account C) ┘                                       results/distributed/<run>/events.jsonl
```

| Piece | File |
|---|---|
| Server entry point | `scripts/ps_server.py` |
| Worker entry point | `scripts/ps_worker.py` |
| Shared task registry (model + data) | `scripts/remote_tasks.py` |
| Colab worker notebook | `notebooks/colab_remote_worker.ipynb` |
| VM setup + systemd unit | `deploy/oracle/setup.sh`, `deploy/oracle/asgc-ps.service`, `deploy/oracle/ps.env.example` |

## How it works

- The server owns the authoritative weights and the optimizer. It also picks the codec (`--compression`,
  `--rho` / `--bits`) and the staleness policy (`--s-low`, `--s-max`, `--beta`). Workers learn the codec
  from the first fetch response, so a worker can never encode with a different codec than the server decodes.
- Each worker trains on a disjoint shard (`--rank` of `--num-shards`). It pushes Top-K/QSGD gradients
  with error feedback and pulls INT8 parameter deltas. If a push is rejected as too stale, the
  worker rolls back its residual.
- When `--max-versions` or `--max-wall-time-s` is reached (or the service is stopped), the server
  sets `stop=True` on fetch responses. Workers exit on their next step. The server then writes a
  final eval, `checkpoint.pt`, and `summary.json`.
- Every push is logged to `events.jsonl` with worker id, staleness, decision, weight, wire bytes,
  loss, and the current compression strength and `S_max`. Periodic `eval` events record test
  accuracy and loss. BatchNorm statistics are recalibrated first, because the server never runs
  training forwards.

## Security

The port is open to the internet, so:

- Every connection must pass the HMAC challenge/response handshake from `multiprocessing.connection`
  (keyed by `ASGC_AUTHKEY`) **before** any payload is unpickled. `ps_server.py` refuses to listen on a
  non-loopback address with the library's default key.
- A failed handshake (wrong key, port scanner) is handled on its own thread and never stops other
  workers from connecting.
- Traffic is authenticated but **not encrypted**. Gradients and weights cross the internet in the clear.
  That is fine for public datasets. Use a VPN or SSH tunnel for anything sensitive.

## 1. Server (one time)

On the VM, from a checkout of this repository:

```bash
bash deploy/oracle/setup.sh
```

This creates `/home/ubuntu/asgc/{venv,data,results,ps.env}`, installs CPU PyTorch, downloads
Fashion-MNIST and CIFAR-10, generates a random `ASGC_AUTHKEY` in `ps.env` (mode 600), opens the port
in iptables (persisted), and installs the `asgc-ps` systemd unit.

**Oracle Cloud also filters traffic at the VCN level.** In the OCI console, open Networking > Virtual
Cloud Networks > your VCN > Security Lists > Default Security List > Add Ingress Rules and add:
source CIDR `0.0.0.0/0`, IP protocol TCP, destination port `29500`. Without this rule, connections
time out even though the VM firewall allows them.

## 2. Start a run

Edit `ASGC_SERVER_ARGS` in `/home/ubuntu/asgc/ps.env`, then:

```bash
sudo systemctl restart asgc-ps
journalctl -u asgc-ps -f          # live status every 10 s: versions/s, connected workers, decisions
grep ASGC_AUTHKEY /home/ubuntu/asgc/ps.env   # the secret to give to workers
```

Useful presets:

| Goal | `ASGC_SERVER_ARGS` |
|---|---|
| Fashion-MNIST, fixed Top-K + EF | `--task fmnist --compression topk --rho 0.1 --eval-interval-s 30 --max-wall-time-s 1800` |
| Fashion-MNIST, adaptive controller | `--task fmnist --compression topk --rho 0.1 --adaptive --eval-interval-s 30 --max-wall-time-s 1800` |
| Uncompressed async baseline | `--task fmnist --compression identity --eval-interval-s 30 --max-wall-time-s 1800` |
| CIFAR-10 ResNet-18 | `--task cifar10 --compression topk --rho 0.1 --eval-interval-s 120 --eval-samples 2000 --max-wall-time-s 3600` |
| Transformer, char-level GPT on Tiny Shakespeare (10.7M params) | `--task gpt --compression topk --rho 0.1 --eval-interval-s 60 --eval-samples 64 --max-wall-time-s 3600` |
| Small transformer from the notebooks (1.8M params) | `--task gpt-mini --compression topk --rho 0.2 --eval-interval-s 30 --eval-samples 128 --max-wall-time-s 1800` |

On a 2-core ARM VM, a ResNet-18 eval over the full 10k test set takes about a minute and competes
with the update loop for CPU. Use `--eval-samples 2000` for periodic evals. The final eval at the
end of the run uses the same subset.

If the service crashes mid-run, add `--resume /home/ubuntu/asgc/results/distributed/<run>/checkpoint.pt`
to continue from the last checkpoint (written every `--checkpoint-interval-s`, default 300 s).

## 3. Workers (each Colab session)

1. Open `notebooks/colab_remote_worker.ipynb` in Colab (File > Open notebook > GitHub, or upload it).
2. Add a Colab secret named `ASGC_AUTHKEY` with the server's key (key icon in the sidebar). Otherwise
   the notebook prompts for it.
3. Set `TASK` to match the server, give each session a unique `RANK` (0, 1, 2, ...), and set
   `NUM_SHARDS` to the total number of workers.
4. Run all cells. The connectivity check prints the RTT to the server. The train cell streams
   worker logs and exits when the server ends the run.

**Runtime choice.** `DEVICE = "auto"` picks a CUDA GPU, then a TPU (via `torch_xla`), then CPU.
T4 GPUs make the best fast workers. A TPU works but waits on the network every step, because each
step is a push plus a pull, and its first step spends time compiling. For the 55k-parameter
Fashion-MNIST CNN, a TPU is often no faster than a CPU. If `torch_xla` or the TPU is unavailable,
the worker falls back to CPU and logs a warning.

The same worker runs anywhere outside Colab too:

```bash
export ASGC_AUTHKEY=<secret>
python scripts/ps_worker.py --server tcp://<server-ip>:29500 --task fmnist --rank 0 --num-shards 3
```

## 4. Collect results

```bash
scp -r oracle-arm:/home/ubuntu/asgc/results/distributed/<run> results/distributed/
```

`events.jsonl` has `start`, `push`, `eval`, `stop_requested`, and `summary` events. `summary.json` has
total versions, accept/downweight/reject counts, uplink bytes, and the final accuracy.

## Performance notes

- **The step time is set by round-trip latency, not compute.** In a measured run with 3 Colab T4 workers
  in the US and the server in Oracle ap-hyderabad-1, the Fashion-MNIST CNN ran at about 1 step/s per
  worker (3.2 updates/s in total) with the server nearly idle. Two intercontinental round trips per step
  took about 1 s, while the GPU compute took milliseconds. A larger model (`--task gpt`) or
  `--local-steps 4` (averaging 4 batches per push) puts the GPU to work without adding round trips.

- Each worker step costs two round trips (push + ack, fetch + delta). For the 55k-parameter SmallCNN,
  latency dominates. For ResNet-18 (11.2M parameters), each step moves about 9 MB up (Top-K at
  rho = 0.1) and about 11 MB down (INT8 delta), so the network is the bottleneck. That is the regime
  where compression and `--local-steps` pay off.
- The server applies updates one at a time on CPU. With many fast GPU workers on ResNet-18, the
  2-core VM's apply rate becomes the limit. Watch `upd/s` in the status line.
