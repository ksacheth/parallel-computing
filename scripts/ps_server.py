"""Run a standalone ASGC parameter server that remote workers connect to over TCP.

Example (on the server machine):

    export ASGC_AUTHKEY="$(openssl rand -hex 32)"
    python scripts/ps_server.py --task fmnist --port 29500 --compression topk --rho 0.1

Workers on other machines then run ``scripts/ps_worker.py --server tcp://<ip>:29500``
with the same ``ASGC_AUTHKEY``. Telemetry lands in ``results/<run>/events.jsonl``.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import signal
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from remote_tasks import bn_calibration_tensors, eval_tensors, evaluate, get_task, make_model
from torch_asgc import AdaptiveConfig, CompressionConfig, ParameterServer, StalenessConfig
from torch_asgc.transport.tcp import DEFAULT_AUTHKEY, resolve_authkey

log = logging.getLogger("asgc.ps")


# Controller limits. "transformer" reproduces the tuned settings of the transformer
# notebooks (findings F16/F17): rho in [0.10, 0.20], S_max in [6, 12], 2 s interval.
ADAPTIVE_PRESETS: dict[str, dict] = {
    "default": {"interval_s": 10.0},
    "transformer": {
        "interval_s": 2.0,
        "mode": "threshold",
        "rho_bounds": (0.10, 0.20),
        "max_delta_rho": 0.05,
        "s_max_bounds": (6, 12),
        "max_delta_s_max": 1,
        "worsen_tol": 0.05,
        "variability_tol": 0.05,
        "reject_frac_max": 0.25,
        "comm_pressure_min": 200_000.0,
        "instability_persistence": 2,
    },
}


def adaptive_config(args: argparse.Namespace) -> AdaptiveConfig | None:
    if not args.adaptive:
        return None
    settings = dict(ADAPTIVE_PRESETS[args.adaptive_preset])
    if args.control_interval_s is not None:
        settings["interval_s"] = args.control_interval_s
    return AdaptiveConfig(**settings)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASGC parameter server for multi-machine training.")
    p.add_argument("--task", default="fmnist", help="fmnist (SmallCNN), cifar10 (ResNet-18), gpt (10.7M char GPT), or gpt-mini (1.8M)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--authkey", default=None, help="shared secret (default: $ASGC_AUTHKEY)")
    p.add_argument("--compression", default="topk", choices=["topk", "quantize", "identity"])
    p.add_argument("--rho", type=float, default=0.1, help="Top-K active fraction")
    p.add_argument("--bits", type=int, default=8, help="QSGD bits")
    p.add_argument("--adaptive", action="store_true", help="enable the adaptive controller")
    p.add_argument("--control-interval-s", type=float, default=None, help="default: preset value")
    p.add_argument("--adaptive-preset", default="default", choices=sorted(ADAPTIVE_PRESETS),
                   help="controller limits; 'transformer' = tuned settings behind finding F16")
    p.add_argument("--no-staleness", action="store_true", help="accept every update at full weight")
    p.add_argument("--s-low", type=int, default=2)
    p.add_argument("--s-max", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=None, help="server lr (default: task default)")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD tasks only")
    p.add_argument("--weight-decay", type=float, default=None, help="default: task default")
    p.add_argument("--max-versions", type=int, default=0, help="stop after N applied updates (0 = no limit)")
    p.add_argument("--max-wall-time-s", type=float, default=0, help="stop N s after the first push (0 = no limit)")
    p.add_argument("--drain-s", type=float, default=30.0, help="keep serving after stop so workers see it")
    p.add_argument("--eval-interval-s", type=float, default=60.0, help="0 disables periodic eval")
    p.add_argument("--eval-samples", type=int, default=0, help="test examples per eval (0 = all)")
    p.add_argument("--bn-samples", type=int, default=1024, help="train images for BatchNorm recalibration")
    p.add_argument("--checkpoint-interval-s", type=float, default=300.0)
    p.add_argument("--resume", default=None, help="checkpoint.pt to continue from")
    p.add_argument("--run-name", default=None)
    p.add_argument("--results-dir", default="results/distributed")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=0, help="0 = PyTorch default")
    return p.parse_args()


class EventLog:
    """Append-only JSONL telemetry, one object per line (same spirit as asgc.metrics)."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("a", buffering=1)
        self._lock = threading.Lock()
        self.t0 = time.time()

    def write(self, kind: str, **fields: object) -> None:
        now = time.time()
        record = {"event": kind, "wall": round(now, 3), "t": round(now - self.t0, 3), **fields}
        with self._lock:
            self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._file.close()


class Evaluator:
    """Evaluates parameter snapshots on a background thread so pushes keep flowing."""

    def __init__(self, model: torch.nn.Module, eval_set: tuple[torch.Tensor, torch.Tensor], bn_x, events: EventLog) -> None:
        self._model = copy.deepcopy(model)
        self._eval_set, self._bn_x = eval_set, bn_x
        self._events = events
        self._thread: threading.Thread | None = None
        self.last: dict[str, float] | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, state: dict[str, torch.Tensor], version: int) -> None:
        self._thread = threading.Thread(target=self.run, args=(state, version), daemon=True)
        self._thread.start()

    def run(self, state: dict[str, torch.Tensor], version: int) -> None:
        start = time.perf_counter()
        self._model.load_state_dict(state)
        acc, loss = evaluate(self._model, self._eval_set, self._bn_x)
        self.last = {"version": version, "accuracy": acc, "loss": loss}
        self._events.write("eval", version=version, accuracy=acc, loss=loss,
                           eval_s=round(time.perf_counter() - start, 2))
        log.info("eval  version=%d  acc=%.4f  loss=%.4f", version, acc, loss)

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()


class PushStats:
    """Rolling counters for the periodic status line and the final summary."""

    def __init__(self, events: EventLog) -> None:
        self.events = events
        self.decisions: Counter[str] = Counter()
        self.bytes_total = 0
        self.first_push: float | None = None
        self.workers: set[str] = set()

    def __call__(self, event: dict) -> None:
        if self.first_push is None:
            self.first_push = time.time()
            log.info("first push received; training clock started")
        self.decisions[event["decision"]] += 1
        self.bytes_total += event["bytes"]
        self.workers.add(event["worker_id"])
        self.events.write("push", **event)

    def train_time(self) -> float:
        return 0.0 if self.first_push is None else time.time() - self.first_push


def build_server(args: argparse.Namespace, model: torch.nn.Module, on_push) -> ParameterServer:
    task = get_task(args.task)
    lr = args.lr if args.lr is not None else task.lr
    weight_decay = args.weight_decay if args.weight_decay is not None else task.weight_decay
    if task.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=weight_decay)
    return ParameterServer(
        model=model,
        optimizer=optimizer,
        bind_address=(args.host, args.port),
        authkey=args.authkey,
        staleness_config=StalenessConfig(
            enabled=not args.no_staleness, s_low=args.s_low, s_max=args.s_max, beta=args.beta
        ),
        compression=CompressionConfig(mode=args.compression, rho=args.rho, bits=args.bits),
        adaptive_config=adaptive_config(args),
        on_push=on_push,
    )


def save_checkpoint(path: Path, server: ParameterServer) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "version": server.version,
            "model": server.model.state_dict(),
            "optimizer": server.server_optimizer.optimizer.state_dict(),
            "compression_strength": server.current_strength,
            "s_max": server.staleness_config.s_max,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: str, server: ParameterServer) -> None:
    ckpt = torch.load(path, map_location="cpu")
    server.model.load_state_dict(ckpt["model"])
    server.server_optimizer.optimizer.load_state_dict(ckpt["optimizer"])
    server.version = int(ckpt["version"])
    server.current_strength = ckpt["compression_strength"]
    server.staleness_config.s_max = int(ckpt["s_max"])
    log.info("resumed from %s at version %d", path, server.version)


def guard_authkey(args: argparse.Namespace) -> None:
    args.authkey = resolve_authkey(args.authkey)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if args.authkey == DEFAULT_AUTHKEY and not loopback:
        raise SystemExit(
            "refusing to listen on a public interface with the default authkey; "
            "set ASGC_AUTHKEY (e.g. `export ASGC_AUTHKEY=$(openssl rand -hex 32)`)"
        )


def stop_reason(args: argparse.Namespace, server: ParameterServer, stats: PushStats) -> str | None:
    if args.max_versions and server.version >= args.max_versions:
        return f"reached {args.max_versions} versions"
    if args.max_wall_time_s and stats.train_time() >= args.max_wall_time_s:
        return f"reached {args.max_wall_time_s:.0f}s wall time"
    return None


def log_status(server: ParameterServer, stats: PushStats, last_version: int, interval: float) -> None:
    rate = (server.version - last_version) / interval
    connected = server.transport.connected_workers
    log.info(
        "status version=%d  %.1f upd/s  connected=%d %s  accepted=%d down=%d rejected=%d  uplink=%.1f MB  strength=%.3g s_max=%d",
        server.version, rate, len(connected), connected, stats.decisions["accepted"],
        stats.decisions["downweighted"], stats.decisions["rejected"], stats.bytes_total / 1e6,
        server.current_strength, server.staleness_config.s_max,
    )


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class ServerLoop:
    """Polls pushes/fetches and runs the periodic status, eval, checkpoint and stop duties."""

    def __init__(self, args: argparse.Namespace, server: ParameterServer, stats: PushStats, run_dir: Path) -> None:
        self.args, self.server, self.stats, self.run_dir = args, server, stats, run_dir
        self.evaluator: Evaluator | None = None
        self.stopping = threading.Event()
        self.stop_at: float | None = None
        self.last_status = self.last_eval = self.last_ckpt = time.time()
        self.last_version = server.version

    def run(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.stopping.set())
        while self.keep_serving():
            if not self.server.process_one():
                time.sleep(0.001)
            now = time.time()
            self.maybe_status(now)
            self.maybe_eval(now)
            self.maybe_checkpoint(now)
            self.maybe_request_stop(now)

    def keep_serving(self) -> bool:
        """Before stop: always. After stop: until workers leave or the drain window ends."""
        if self.stop_at is None:
            return True
        return time.time() < self.stop_at and bool(self.server.transport.connected_workers)

    def maybe_status(self, now: float) -> None:
        if now - self.last_status >= 10:
            log_status(self.server, self.stats, self.last_version, now - self.last_status)
            self.last_status, self.last_version = now, self.server.version

    def maybe_eval(self, now: float) -> None:
        due = now - self.last_eval >= self.args.eval_interval_s
        ready = self.evaluator is not None and not self.evaluator.busy and self.stats.first_push is not None
        if due and ready:
            self.evaluator.submit(snapshot(self.server.model), self.server.version)
            self.last_eval = now

    def maybe_checkpoint(self, now: float) -> None:
        if now - self.last_ckpt >= self.args.checkpoint_interval_s:
            save_checkpoint(self.run_dir / "checkpoint.pt", self.server)
            self.last_ckpt = now

    def maybe_request_stop(self, now: float) -> None:
        if self.stop_at is not None:
            return
        reason = "signal" if self.stopping.is_set() else stop_reason(self.args, self.server, self.stats)
        if reason:
            log.info("stopping (%s); draining up to %.0fs so workers receive stop", reason, self.args.drain_s)
            self.server.stop_training = True
            self.stop_at = now + self.args.drain_s
            self.stats.events.write("stop_requested", reason=reason, version=self.server.version)

    def finish(self) -> None:
        server, stats = self.server, self.stats
        server.close()
        save_checkpoint(self.run_dir / "checkpoint.pt", server)
        summary = {
            "version": server.version,
            "train_time_s": round(stats.train_time(), 1),
            "decisions": dict(stats.decisions),
            "uplink_bytes": stats.bytes_total,
            "workers": sorted(stats.workers),
        }
        if self.evaluator:
            self.evaluator.wait()
            self.evaluator.run(snapshot(server.model), server.version)
            summary["final_eval"] = self.evaluator.last
        stats.events.write("summary", **summary)
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        stats.events.close()
        log.info("done: %s", json.dumps(summary))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    guard_authkey(args)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    task = get_task(args.task)
    run_name = args.run_name or f"{args.task}_{args.compression}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(run_dir / "events.jsonl")
    stats = PushStats(events)

    model = make_model(task, seed=args.seed)
    server = build_server(args, model, on_push=stats)
    if args.resume:
        load_checkpoint(args.resume, server)

    config = {k: v for k, v in vars(args).items() if k != "authkey"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    events.write("start", config=config, params=sum(p.numel() for p in model.parameters()))

    loop = ServerLoop(args, server, stats, run_dir)
    if args.eval_interval_s > 0:
        log.info("loading %s eval data into %s", task.dataset, args.data_root)
        eval_set = eval_tensors(task, args.data_root, args.eval_samples)
        bn_x = bn_calibration_tensors(task, args.data_root, args.bn_samples)
        loop.evaluator = Evaluator(model, eval_set, bn_x, events)

    log.info("ASGC parameter server listening on tcp://%s:%d  run=%s", args.host, args.port, run_dir)
    loop.run()
    loop.finish()


if __name__ == "__main__":
    main()
