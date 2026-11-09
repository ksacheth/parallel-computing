"""Run one ASGC worker that trains against a remote parameter server over TCP.

Example (on any machine that can reach the server, e.g. a Colab runtime):

    export ASGC_AUTHKEY=<same secret as the server>
    python scripts/ps_worker.py --server tcp://203.0.113.10:29500 --task fmnist --rank 0 --num-shards 3

The compression codec and its strength come from the server, so a worker only
chooses local concerns: its data shard, batch size, device, and error feedback.
"""
from __future__ import annotations

import argparse
import logging
import socket
import time
import uuid

import torch
import torch.nn as nn

from remote_tasks import get_task, make_model, train_loader
from torch_asgc import CompressionConfig, DistributedModel, DistributedOptimizer, connect

log = logging.getLogger("asgc.worker")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASGC remote worker.")
    p.add_argument("--server", required=True, help="tcp://host:port of the parameter server")
    p.add_argument("--authkey", default=None, help="shared secret (default: $ASGC_AUTHKEY)")
    p.add_argument("--worker-id", default=None, help="unique name (default: hostname + random suffix)")
    p.add_argument("--task", default="fmnist", help="must match the server's --task")
    p.add_argument("--rank", type=int, default=0, help="this worker's data shard index")
    p.add_argument("--num-shards", type=int, default=1, help="total shards (usually the worker count)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--no-error-feedback", action="store_true")
    p.add_argument("--local-steps", type=int, default=1, help="micro-batches averaged per push")
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N pushes (0 = until server stops)")
    p.add_argument("--amp", action="store_true", help="fp16 autocast + GradScaler (CUDA only)")
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default="data")
    p.add_argument("--loader-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--connect-retries", type=int, default=60, help="retry every 5 s while the server boots")
    p.add_argument("--reply-timeout-s", type=float, default=600.0)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def batches(loader):
    """Endless stream over the shard; reshuffles every pass."""
    while True:
        yield from loader


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task = get_task(args.task)
        self.device = pick_device(args.device)
        self.worker_id = args.worker_id or f"{socket.gethostname()[:12]}-{uuid.uuid4().hex[:4]}"
        log.info("worker %s connecting to %s (device=%s)", self.worker_id, args.server, self.device)
        self.client = connect(
            args.server,
            worker_id=self.worker_id,
            authkey=args.authkey,
            connect_retries=args.connect_retries,
            reply_timeout=args.reply_timeout_s,
        )
        self.net = make_model(self.task).to(self.device)
        first = self.client.sync_model(self.net)  # pull authoritative weights before the first step
        if first.compression_mode is None:
            raise SystemExit("server did not report its compression mode; update torch_asgc on the server")
        compression = CompressionConfig(mode=first.compression_mode).with_strength(first.compression_setting)
        log.info("joined at server version %d; codec=%s strength=%.3g", first.version,
                 compression.mode, compression.strength)
        self.model = DistributedModel(self.net, client=self.client)
        self.optimizer = DistributedOptimizer(
            base_optimizer=torch.optim.SGD(self.net.parameters(), lr=self.task.lr),  # server applies the step
            model=self.model,
            client=self.client,
            compression=compression,
            error_feedback=not args.no_error_feedback,
            local_steps=args.local_steps,
            max_grad_norm=args.max_grad_norm,
        )
        use_amp = args.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.autocast = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        self.optimizer.zero_grad()
        with self.autocast():
            loss = nn.functional.cross_entropy(self.model(x), y)
        loss_val = float(loss.detach())
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.optimizer.step(loss_val=loss_val, scaler=self.scaler)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step(loss_val=loss_val)
        return loss_val

    def run(self) -> None:
        args = self.args
        loader = train_loader(
            self.task, args.data_root, args.rank, args.num_shards,
            args.batch_size or self.task.batch_size, args.loader_workers,
        )
        log.info("shard %d/%d: %d batches per pass", args.rank, args.num_shards, len(loader))
        self.net.train()
        step, start, window_start, loss_sum = 0, time.time(), time.time(), 0.0
        for x, y in batches(loader):
            if self.client.stop_requested:
                log.info("server requested stop")
                break
            loss_sum += self.train_step(x, y)
            step += 1
            if step % args.log_every == 0:
                rate = args.log_every / (time.time() - window_start)
                log.info("step %d  server_version=%d  loss=%.4f  %.2f steps/s  strength=%.3g",
                         step, self.client.known_version, loss_sum / args.log_every, rate,
                         self.optimizer.compression_config.strength)
                window_start, loss_sum = time.time(), 0.0
            if args.max_steps and step >= args.max_steps:
                log.info("reached --max-steps %d", args.max_steps)
                break
        log.info("finished %d local steps in %.0fs", step, time.time() - start)

    def close(self) -> None:
        self.client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = Worker(parse_args())
    try:
        worker.run()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
