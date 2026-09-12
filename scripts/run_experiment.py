"""Run one experiment: python scripts/run_experiment.py --config configs/smoke_fmnist.yaml"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asgc.config import load_config, seed_everything
from asgc.data import prepare_datasets
from asgc.metrics import read_events, summarize
from asgc.server import server_main
from asgc.transport import Transport
from asgc.worker import worker_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one asynchronous training experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.run.seed)
    prepare_datasets(config)  # download from the parent so children never race

    ctx = mp.get_context("spawn")
    transport = Transport(ctx, comm_delay_s=config.heterogeneity.comm_delay_ms / 1000.0)
    for worker_id in range(config.workload.num_workers):
        transport.register_worker(worker_id)

    server = ctx.Process(target=server_main, args=(config, transport), name="parameter-server")
    workers = [
        ctx.Process(target=worker_main, args=(config, worker_id, transport), name=f"worker-{worker_id}")
        for worker_id in range(config.workload.num_workers)
    ]
    for process in [*workers, server]:
        process.start()

    server.join()
    for process in workers:
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()

    if server.exitcode != 0:
        raise SystemExit(f"parameter server exited with code {server.exitcode}")

    events = read_events(Path(config.run.results_dir) / config.run.name / "events.jsonl")
    summary = summarize(events)
    print(f"run '{config.run.name}' complete")
    print(f"  applied updates : {summary['applied_updates']}")
    print(f"  final accuracy  : {summary['final_test_accuracy']:.4f}")
    print(f"  final loss      : {summary['final_test_loss']:.4f}")
    print(f"  total bytes     : {summary['total_bytes']}")
    print(f"  updates/sec     : {summary['updates_per_sec']:.2f}")
    print(f"  mean staleness  : {summary['mean_staleness']:.3f} (max {summary['max_staleness']})")
    print(f"  elapsed         : {summary['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
