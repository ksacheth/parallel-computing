"""Run a matrix of configs x seeds sequentially: the overnight driver.

Usage: python scripts/run_matrix.py --matrix configs/matrix_cifar_full.yaml
Derives one YAML per (config, seed) into results/matrix/ (name and seed
overridden), then invokes run_experiment.py on each, continuing past failures.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configs x seeds sequentially.")
    parser.add_argument("--matrix", required=True, help="Matrix YAML listing configs and seeds")
    args = parser.parse_args()

    with open(args.matrix, "r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f)
    config_paths = matrix["configs"]
    seeds = matrix["seeds"]

    out_dir = ROOT / "results" / "matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for seed in seeds:
        for config_path in config_paths:
            with open(ROOT / config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            raw["run"]["seed"] = seed
            raw["run"]["name"] = f"{raw['run']['name']}_s{seed}"
            derived = out_dir / f"{raw['run']['name']}.yaml"
            with open(derived, "w", encoding="utf-8") as f:
                yaml.safe_dump(raw, f, sort_keys=False)

            print(f"[matrix] === {raw['run']['name']} ===", flush=True)
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config", str(derived)],
                cwd=ROOT,
            )
            minutes = (time.perf_counter() - started) / 60
            status = "ok" if completed.returncode == 0 else f"FAILED ({completed.returncode})"
            print(f"[matrix] {raw['run']['name']}: {status} in {minutes:.1f} min", flush=True)
            if completed.returncode != 0:
                failures.append(raw["run"]["name"])

    print(f"[matrix] done: {len(failures)} failures", flush=True)
    for name in failures:
        print(f"[matrix]   failed: {name}", flush=True)


if __name__ == "__main__":
    main()
