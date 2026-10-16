"""Parameter-server loop: authoritative model, version counter, update application.

Two topologies share this loop. Async (default): every arriving update is
measured and applied immediately under the staleness policy. Sync
(``workload.sync``): pushes are buffered per version and the mean of all
worker gradients is applied once per round, after which held fetch replies
are released — true bulk-synchronous behavior with the workers' fetches
blocking until the round completes.

Stage 1 accepts every arriving update at full weight; the staleness policy
arrives in stage 2 behind the ``decision`` field of the update event, so the
event schema is already final.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asgc.compression import make_compressor
from asgc.config import ExperimentConfig, resolve_device
from asgc.controller import compute_window_stats, make_policy
from asgc.data import test_loader, worker_loader
from asgc.metrics import (
    ControlEvent,
    EvalEvent,
    EventWriter,
    RunEndEvent,
    RunStartEvent,
    UpdateEvent,
)
from asgc.models.cnn_mnist import build_model
from asgc.staleness import decide
from asgc.transport import FetchRequest, FetchResponse, Transport, STOP_VERSION

FETCH_POLL_S = 0.0
PUSH_POLL_S = 0.002
SHUTDOWN_GRACE_S = 2.0


def _bn_recalibrate(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 64) -> None:
    """Refresh BatchNorm running statistics against the current server weights.

    The server never runs training forwards, so its BN running stats would
    otherwise stay at initialization and every eval of a BN model would be
    meaningless (accuracy pinned near chance). A short no-grad forward sweep
    re-estimates the stats honestly, identically for every method.
    """
    if not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        return
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= batches:
                break
            model(x.to(device))


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
    device = resolve_device(config.run.device)
    model = build_model(config.workload.model).to(device)
    compressor = make_compressor(config.compression)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    loader = test_loader(config)
    stats_loader = worker_loader(config, worker_id=0)  # BN recalibration batches
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

    controller = config.controller
    policy = make_policy(controller, config.compression.mode) if controller.enabled else None
    current_strength = (
        config.compression.rho if config.compression.mode == "topk" else float(config.compression.bits)
    )
    compression_for_workers = config.compression
    effective_staleness = config.staleness
    window: list[dict] = []
    next_control_t = t0 + controller.interval_s if policy is not None else float("inf")

    sync = config.workload.sync
    num_workers = config.workload.num_workers
    held_fetches: list[FetchRequest] = []
    rounds: dict[int, dict[int, object]] = {}  # sync: version -> worker_id -> PushUpdate

    writer.write(RunStartEvent(
        event="run_start",
        config_name=config.run.name,
        num_workers=num_workers,
        dataset=config.workload.dataset,
        model=config.workload.model,
        batch_size=config.workload.batch_size,
        compute_delay=dict(config.heterogeneity.compute_delay),
        comm_delay_ms=config.heterogeneity.comm_delay_ms,
        bandwidth_mbps=config.heterogeneity.bandwidth_mbps,
        started_at=time.time(),
    ))

    def run_eval() -> None:
        nonlocal last_eval_version, final_acc, final_loss
        _bn_recalibrate(model, stats_loader, device)
        acc, loss = _evaluate(model, loader, device)
        final_acc, final_loss = acc, loss
        last_eval_version = version
        elapsed = time.perf_counter() - t0
        eval_event = EvalEvent(event="eval", version=version, test_accuracy=acc, test_loss=loss, elapsed_s=elapsed)
        writer.write(eval_event)
        window.append(asdict(eval_event))
        print(f"[eval] version={version} accuracy={acc:.4f} loss={loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    run_eval()  # model quality at version 0, before any update

    # Server-side step decay: every method sees the same schedule by version.
    lr_milestone = (
        int(config.workload.max_updates * config.workload.lr_decay_frac)
        if config.workload.lr_decay_frac > 0
        else None
    )

    def snapshot_response(req: FetchRequest) -> FetchResponse:
        """Full snapshot only when the worker's copy is stale; a params-less
        reply otherwise so the worker keeps its local copy (ResNet gradients
        make full snapshots far too costly to ship unconditionally)."""
        if req.known_version == version:
            return FetchResponse(params=None, version=version, compression=compression_for_workers)
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return FetchResponse(params=snapshot, version=version, compression=compression_for_workers)

    def release_held_fetches() -> None:
        nonlocal held_fetches
        if not held_fetches:
            return
        snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for req in held_fetches:
            transport.reply(req.worker_id, FetchResponse(
                params=snapshot, version=version, compression=compression_for_workers,
            ))
        held_fetches = []

    while applied < config.workload.max_updates and (time.perf_counter() - t0) < config.workload.max_wall_time_s:
        if policy is not None and time.perf_counter() >= next_control_t:
            stats = compute_window_stats(window)
            decision = policy.decide(stats, current_strength, effective_staleness.s_max)
            control_event = ControlEvent(
                event="control",
                elapsed_s=time.perf_counter() - t0,
                loss_trend=stats.loss_trend,
                loss_variability=stats.loss_variability,
                mean_staleness=stats.mean_staleness,
                rejected_frac=stats.rejected_frac,
                bytes_per_s=stats.bytes_per_s,
                updates_per_s=stats.updates_per_s,
                residual_norm_var=stats.residual_norm_var,
                old_compression=current_strength,
                new_compression=decision.value,
                old_s_max=effective_staleness.s_max,
                new_s_max=decision.s_max,
                reason=decision.reason,
            )
            writer.write(control_event)
            print(
                f"[control] strength={current_strength:.3f}->{decision.value:.3f} "
                f"s_max={effective_staleness.s_max}->{decision.s_max} ({decision.reason})",
                flush=True,
            )
            current_strength = decision.value
            effective_staleness = replace(effective_staleness, s_max=decision.s_max)
            compression_for_workers = replace(
                config.compression,
                rho=current_strength if config.compression.mode == "topk" else config.compression.rho,
                bits=int(current_strength) if config.compression.mode == "quantize" else config.compression.bits,
            )
            window = []
            next_control_t += controller.interval_s

        # In sync mode a worker's post-push fetch is held until its round
        # completes (prevents duplicate compute for the same version), but
        # fetches from workers still catching up are answered immediately —
        # otherwise no round could ever complete. A worker whose known version
        # equals the server's holds the current model whether or not its push
        # has been processed yet, so known_version == version is race-free.
        # Async mode never starves.
        while True:
            req = transport.next_fetch_request(timeout=FETCH_POLL_S)
            if req is None:
                break
            if sync and req.known_version >= version:
                held_fetches.append(req)
            else:
                transport.reply(req.worker_id, snapshot_response(req))

        update = transport.next_push(timeout=PUSH_POLL_S)
        if update is None:
            continue

        elapsed = time.perf_counter() - t0
        lr = config.workload.lr
        if lr_milestone is not None and version >= lr_milestone:
            lr *= config.workload.lr_gamma

        if sync:
            round_entries = rounds.setdefault(update.version, {})
            round_entries[update.worker_id] = update
            total_bytes += update.payload_bytes
            stalenesses.append(0)
            if len(round_entries) < num_workers:
                continue
            # round complete: apply the mean of all worker gradients once
            pieces = list(round_entries.values())
            params_list = list(model.parameters())
            decoded_pushes = [compressor.decode(pu.payload, params_list) for pu in pieces]
            with torch.no_grad():
                for pi, p in enumerate(params_list):
                    mean_grad = torch.stack([d[pi].to(p) for d in decoded_pushes]).mean(dim=0)
                    p.add_(mean_grad, alpha=-lr)
            version += 1
            applied += 1
            rounds.pop(update.version)
            for pu in pieces:
                update_event = UpdateEvent(
                    event="update",
                    worker_id=pu.worker_id,
                    version=version,
                    staleness=0,
                    payload_bytes=pu.payload_bytes,
                    fetch_s=pu.fetch_s,
                    compute_s=pu.compute_s,
                    decision="accepted",
                    weight=1.0 / num_workers,
                    residual_norm=pu.residual_norm,
                    elapsed_s=elapsed,
                )
                writer.write(update_event)
                window.append(asdict(update_event))
            release_held_fetches()
            if version - last_eval_version >= config.workload.eval_interval:
                run_eval()
            continue

        tau = version - update.version
        stalenesses.append(tau)
        decision = decide(tau, effective_staleness)
        if decision.accept:
            decoded = compressor.decode(update.payload, list(model.parameters()))
            with torch.no_grad():
                for p, g in zip(model.parameters(), decoded):
                    p.add_(g.to(p), alpha=-lr * decision.weight)
            version += 1
            applied += 1
            kind = "downweighted" if decision.weight < 1.0 else "accepted"
        else:
            rejected += 1
            kind = "rejected"
        total_bytes += update.payload_bytes
        update_event = UpdateEvent(
            event="update",
            worker_id=update.worker_id,
            version=version,
            staleness=tau,
            payload_bytes=update.payload_bytes,
            fetch_s=update.fetch_s,
            compute_s=update.compute_s,
            decision=kind,
            weight=decision.weight,
            residual_norm=update.residual_norm,
            elapsed_s=elapsed,
        )
        writer.write(update_event)
        window.append(asdict(update_event))

        if version - last_eval_version >= config.workload.eval_interval:
            run_eval()

    # Workers blocked on a fetch must hear a stop before this process exits:
    # both late async fetches and sync fetches still held for an incomplete round.
    shutdown_start = time.perf_counter()
    while time.perf_counter() - shutdown_start < SHUTDOWN_GRACE_S:
        req = transport.next_fetch_request(timeout=0.05)
        if req is not None:
            transport.reply(req.worker_id, FetchResponse(params=None, version=STOP_VERSION, stop=True))
        for held in held_fetches:
            transport.reply(held.worker_id, FetchResponse(params=None, version=STOP_VERSION, stop=True))
        held_fetches = []

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
