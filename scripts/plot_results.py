"""Aggregate and plot experiment results.

Usage:
  python scripts/plot_results.py --runs fmnist_sync_sgd fmnist_async_plain ... [--target 0.85]

Prints the cross-run summary table, writes results/plot/summary.csv, and
renders the proposal's comparison figures when matplotlib is available.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_run(name: str, results_dir: Path) -> dict | None:
    path = results_dir / name / "events.jsonl"
    if not path.exists():
        return None
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_end = next((e for e in events if e["event"] == "run_end"), None)
    evals = [e for e in events if e["event"] == "eval"]
    updates = [e for e in events if e["event"] == "update"]
    run_start = next((e for e in events if e["event"] == "run_start"), {})
    return {
        "name": name,
        "events": events,
        "run_end": run_end,
        "evals": evals,
        "updates": updates,
        "dataset": run_start.get("dataset", "?"),
        "model": run_start.get("model", "?"),
        "delays": run_start.get("compute_delay", {}),
    }


def time_to_target(evals: list[dict], target: float) -> float | None:
    for e in evals:
        if e["test_accuracy"] >= target:
            return e["elapsed_s"]
    return None


def summarize(runs: list[dict], target: float | None) -> list[dict]:
    rows = []
    for run in runs:
        end = run["run_end"] or {}
        evals = run["evals"]
        accs = [e["test_accuracy"] for e in evals]
        drops = [max(0.0, a - b) for a, b in zip(accs, accs[1:])]
        rows.append({
            "run": run["name"],
            "dataset": run["dataset"],
            "final_accuracy": end.get("final_test_accuracy"),
            "final_loss": end.get("final_test_loss"),
            "applied": end.get("applied_updates"),
            "rejected_frac": end.get("rejected_frac"),
            "total_gb": round(end.get("total_bytes", 0) / 1e9, 3),
            "compression_ratio": end.get("compression_ratio"),
            "updates_per_sec": end.get("updates_per_sec"),
            "mean_staleness": end.get("mean_staleness"),
            "max_staleness": end.get("max_staleness"),
            "max_acc_drop": round(max(drops), 4) if drops else None,
            "wall_s": round(end.get("elapsed_s", 0), 1),
            "time_to_target_s": (
                round(t, 1) if (target is not None and (t := time_to_target(evals, target)) is not None) else None
            ),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    widths = {k: max(len(k), *(len(f"{r[k]}") if r[k] is not None else 4 for r in rows)) for k in keys}
    header = " | ".join(k.rjust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(("-" if r[k] is None else f"{r[k]}").rjust(widths[k]) for k in keys))


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary written to {out}")


def plot(runs: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        if run["evals"]:
            ax.plot([e["elapsed_s"] for e in run["evals"]], [e["test_accuracy"] for e in run["evals"]], label=run["name"])
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs wall-clock time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_vs_time.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        if run["evals"]:
            ax.plot([e["elapsed_s"] for e in run["evals"]], [e["test_loss"] for e in run["evals"]], label=run["name"])
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("test loss")
    ax.set_title("Loss vs wall-clock time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_time.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        pts = []
        cumulative = 0
        update_iter = iter(run["updates"])
        current = next(update_iter, None)
        for e in run["evals"]:
            while current is not None and current["elapsed_s"] <= e["elapsed_s"]:
                cumulative += current["payload_bytes"]
                current = next(update_iter, None)
            pts.append((cumulative / 1e9, e["test_accuracy"]))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=3, label=run["name"])
    ax.set_xlabel("communication (GB)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Bytes vs accuracy (lower-left is better)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bytes_vs_accuracy.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        taus = [u["staleness"] for u in run["updates"]]
        decisions = [1.0 if u["decision"] == "rejected" else 0.0 for u in run["updates"]]
        if any(t > 0 for t in taus) or any(decisions):
            ax.scatter(taus, decisions, s=4, alpha=0.15, label=run["name"])
    ax.set_xlabel("arrival staleness")
    ax.set_ylabel("rejected (1) / kept (0)")
    ax.set_title("Staleness vs rejection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "staleness_vs_rejection.png", dpi=150)

    print(f"figures written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run names under the results dir")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--target", type=float, default=None, help="Target accuracy for time-to-target")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = []
    for name in args.runs:
        run = load_run(name, results_dir)
        if run is None:
            print(f"skipping {name}: no events.jsonl")
            continue
        runs.append(run)

    rows = summarize(runs, args.target)
    print_table(rows)
    if rows:
        write_csv(rows, results_dir / "plot" / "summary.csv")
        plot(runs, results_dir / "plot")


if __name__ == "__main__":
    main()
