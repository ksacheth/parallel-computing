"""Parameter-server loop: authoritative model, version counter, update application.

Stage 1 accepts every arriving update at full weight. The staleness policy
arrives in stage 2 behind the ``decision`` field of the update event, so the
event schema is already final.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asgc.compression import make_compressor
from asgc.config import ExperimentConfig
from asgc.data import test_loader
from asgc.metrics import (
    EvalEvent,
    EventWriter,
    RunEndEvent,
    RunStartEvent,
    UpdateEvent,
)
from asgc.models.cnn_mnist import build_model
from asgc.staleness import decide
from asgc.transport import STOP_VERSION, FetchResponse, Transport

FETCH_POLL_S = 0.0
PUSH_POLL_S = 0.002
SHUTDOWN_GRACE_S = 2.0


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total, loss_sum / total


def server_main(config: ExperimentConfig, transport: Transport) -> None:
    torch.set_num_threads(1)  # one thread per process; measured faster than multi-thread for this workload
    torch.manual_seed(config.run.seed)
    device = torch.device("cpu")
    model = build_model(config.workload.model).to(device)
    compressor = make_compressor(config.compression)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    loader = test_loader(config)
    writer = EventWriter(config.run.results_dir, config.run.name)

    t0 = time.perf_counter()
    version = 0
    applied = 0
    rejected = 0
    total_bytes = 0
    stalenesses: list[int] = []
    last_eval_version = -1
    final_acc = 0.0
    final_loss = 0.0

    writer.write(RunStartEvent(
        event="run_start",
        config_name=config.run.name,
        num_workers=config.workload.num_workers,
        dataset=config.workload.dataset,
        model=config.workload.model,
        batch_size=config.workload.batch_size,
        compute_delay=dict(config.heterogeneity.compute_delay),
        comm_delay_ms=config.heterogeneity.comm_delay_ms,
        started_at=time.time(),
    ))

    def run_eval() -> None:
        nonlocal last_eval_version, final_acc, final_loss
        acc, loss = _evaluate(model, loader, device)
        final_acc, final_loss = acc, loss
        last_eval_version = version
        elapsed = time.perf_counter() - t0
        writer.write(EvalEvent(event="eval", version=version, test_accuracy=acc, test_loss=loss, elapsed_s=elapsed))
        print(f"[eval] version={version} accuracy={acc:.4f} loss={loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    run_eval()  # model quality at version 0, before any update

    # Server-side step decay: every method sees the same schedule by version.
    lr_milestone = (
        int(config.workload.max_updates * config.workload.lr_decay_frac)
        if config.workload.lr_decay_frac > 0
        else None
    )

    while applied < config.workload.max_updates and (time.perf_counter() - t0) < config.workload.max_wall_time_s:
        # Answer every pending fetch before considering pushes: workers must
        # never starve behind update processing.
        while True:
            worker_id = transport.next_fetch_request(timeout=FETCH_POLL_S)
            if worker_id is None:
                break
            snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
            transport.reply(worker_id, FetchResponse(params=snapshot, version=version))

        update = transport.next_push(timeout=PUSH_POLL_S)
        if update is None:
            continue

        elapsed = time.perf_counter() - t0
        tau = version - update.version
        stalenesses.append(tau)
        decision = decide(tau, config.staleness)
        if decision.accept:
            decoded = compressor.decode(update.payload, list(model.parameters()))
            lr = config.workload.lr
            if lr_milestone is not None and version >= lr_milestone:
                lr *= config.workload.lr_gamma
            with torch.no_grad():
                for p, g in zip(model.parameters(), decoded):
                    p.add_(g.to(p.dtype), alpha=-lr * decision.weight)
            version += 1
            applied += 1
            kind = "downweighted" if decision.weight < 1.0 else "accepted"
        else:
            rejected += 1
            kind = "rejected"
        total_bytes += update.payload_bytes
        writer.write(UpdateEvent(
            event="update",
            worker_id=update.worker_id,
            version=version,
            staleness=tau,
            payload_bytes=update.payload_bytes,
            fetch_s=update.fetch_s,
            compute_s=update.compute_s,
            decision=kind,
            weight=decision.weight,
            elapsed_s=elapsed,
        ))

        if version - last_eval_version >= config.workload.eval_interval:
            run_eval()

    # Workers blocked on a fetch must hear a stop before this process exits.
    shutdown_start = time.perf_counter()
    while time.perf_counter() - shutdown_start < SHUTDOWN_GRACE_S:
        worker_id = transport.next_fetch_request(timeout=0.05)
        if worker_id is not None:
            transport.reply(worker_id, FetchResponse(params=None, version=STOP_VERSION, stop=True))

    elapsed = time.perf_counter() - t0
    raw_bytes = param_bytes * len(stalenesses)
    writer.write(RunEndEvent(
        event="run_end",
        applied_updates=applied,
        rejected_updates=rejected,
        rejected_frac=rejected / len(stalenesses) if stalenesses else 0.0,
        total_bytes=total_bytes,
        raw_bytes=raw_bytes,
        compression_ratio=raw_bytes / total_bytes if total_bytes else 1.0,
        updates_per_sec=applied / elapsed if elapsed > 0 else 0.0,
        elapsed_s=elapsed,
        mean_staleness=sum(stalenesses) / len(stalenesses) if stalenesses else 0.0,
        max_staleness=max(stalenesses) if stalenesses else 0,
        final_test_accuracy=final_acc,
        final_test_loss=final_loss,
    ))
    writer.close()
    print(f"[server] run complete: {applied} applied ({rejected} rejected), {total_bytes} bytes, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    raise SystemExit("server_main is started by scripts/run_experiment.py")
