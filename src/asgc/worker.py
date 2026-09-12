"""Worker loop: fetch model, compute gradient, compress, push update.

The error-feedback residual is a stage 4 addition; stage 1 pushes the plain
compressed gradient. Payload tensors are aligned with ``model.parameters()``
order, matching the server's application loop.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from asgc.compression import make_compressor
from asgc.config import ExperimentConfig, seed_everything
from asgc.data import cycle, worker_loader
from asgc.delays import apply_compute_delay
from asgc.models.cnn_mnist import build_model
from asgc.transport import PushUpdate, Transport


def worker_main(config: ExperimentConfig, worker_id: int, transport: Transport) -> None:
    torch.set_num_threads(1)  # one thread per process; N processes times N threads oversubscribes the CPU
    seed_everything(config.run.seed + 1000 + worker_id)
    model = build_model(config.workload.model)
    model.train()
    compressor = make_compressor(config.compression.mode)
    batches = cycle(worker_loader(config, worker_id))
    criterion = nn.CrossEntropyLoss()

    while True:
        fetch_start = time.perf_counter()
        response = transport.fetch(worker_id)
        fetch_s = time.perf_counter() - fetch_start
        if response.stop:
            break
        model.load_state_dict(response.params)

        x, y = next(batches)
        compute_start = time.perf_counter()
        apply_compute_delay(config.heterogeneity, worker_id)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
        compute_s = time.perf_counter() - compute_start

        payload = compressor.encode(grads)
        transport.push(PushUpdate(
            worker_id=worker_id,
            version=response.version,
            payload=payload,
            payload_bytes=compressor.encoded_bytes(payload),
            fetch_s=fetch_s,
            compute_s=compute_s,
        ))


if __name__ == "__main__":
    raise SystemExit("worker_main is started by scripts/run_experiment.py")
