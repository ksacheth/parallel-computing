"""Compare multi-machine runs recorded by ps_server.py.

Usage:
    python scripts/compare_remote_runs.py results/distributed/gpt_topk_* results/distributed/gpt_identity_* \
        [--target-loss 2.5] [--out results/distributed/comparison.md]

Reads each run's config.json and events.jsonl and prints one Markdown table:
throughput, wall-clock to the final version, uplink bytes, staleness decisions,
final evaluation, and the controller trajectory (mean/final rho, S_max range).
Uplink bytes are the exact encoded payload sizes; the INT8 downlink is the same
per fetch for every arm and is not logged by the server.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    return {"name": run_dir.name, "config": config, "events": events}


def arm_label(config: dict) -> str:
    if config.get("adaptive"):
        return f"ASGC adaptive (start rho={config['rho']}, preset={config.get('adaptive_preset', 'default')})"
    if config["compression"] == "identity":
        return "No compression"
    if config["compression"] == "topk":
        return f"Fixed Top-K {config['rho']:.0%} + EF"
    return f"Fixed QSGD {config['bits']}-bit + EF"


def time_to_loss(evals: list[dict], t0: float, target: float | None) -> float | None:
    if target is None:
        return None
    hit = next((e for e in evals if e["loss"] <= target), None)
    return None if hit is None else hit["t"] - t0


def split_events(events: list[dict]) -> dict[str, list[dict]]:
    """Group events by kind: push, eval, stop_requested, summary, ..."""
    grouped: dict[str, list[dict]] = {}
    for event in events:
        grouped.setdefault(event["event"], []).append(event)
    return grouped


def summarize(run: dict, target_loss: float | None) -> dict:
    grouped = split_events(run["events"])
    pushes, evals = grouped["push"], grouped.get("eval", [])
    summary = (grouped.get("summary") or [{}])[-1]
    t0 = pushes[0]["t"]
    stops = grouped.get("stop_requested")
    t_end = stops[0]["t"] if stops else pushes[-1]["t"]
    taus = sorted(p["staleness"] for p in pushes)
    rhos = [p["compression_setting"] for p in pushes]
    s_maxes = [p["s_max"] for p in pushes]
    decisions = {k: sum(p["decision"] == k for p in pushes) for k in ("accepted", "downweighted", "rejected")}
    applied = decisions["accepted"] + decisions["downweighted"]
    uplink = sum(p["bytes"] for p in pushes)
    final = summary.get("final_eval") or (evals[-1] if evals else {})
    return {
        "arm": arm_label(run["config"]),
        "run": run["name"],
        "versions": summary.get("version", pushes[-1]["server_version"]),
        "wall_s": t_end - t0,
        "upd_per_s": applied / max(t_end - t0, 1e-9),
        "uplink_mb": uplink / 1e6,
        "uplink_mb_per_update": uplink / 1e6 / max(applied, 1),
        "decisions": decisions,
        "mean_tau": statistics.fmean(taus),
        "p95_tau": taus[int(0.95 * (len(taus) - 1))],
        "final_loss": final.get("loss"),
        "final_acc": final.get("accuracy"),
        "time_to_target_s": time_to_loss(evals, t0, target_loss),
        "rho_mean": statistics.fmean(rhos),
        "rho_final": rhos[-1],
        "s_max_range": (min(s_maxes), max(s_maxes)),
        "workers": len({p["worker_id"] for p in pushes}),
    }


def fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def markdown(rows: list[dict], target_loss: float | None) -> str:
    target_col = f"Time to loss <= {target_loss} (s)" if target_loss is not None else None
    header = ["Arm", "Versions", "Wall-clock (s)", "Updates/s", "Uplink (MB)", "MB per applied update",
              "Accepted / down / rejected", "Mean / p95 staleness", "Final loss", "Final acc",
              "rho mean (final)", "S_max range"]
    if target_col:
        header.insert(9, target_col)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        d = r["decisions"]
        cells = [
            r["arm"], str(r["versions"]), fmt(r["wall_s"], ".1f"), fmt(r["upd_per_s"], ".2f"),
            fmt(r["uplink_mb"], ",.0f"), fmt(r["uplink_mb_per_update"], ".2f"),
            f"{d['accepted']} / {d['downweighted']} / {d['rejected']}",
            f"{r['mean_tau']:.2f} / {r['p95_tau']}", fmt(r["final_loss"], ".4f"),
            fmt(r["final_acc"], ".2%"), f"{r['rho_mean']:.3f} ({r['rho_final']:.2f})",
            f"{r['s_max_range'][0]}-{r['s_max_range'][1]}",
        ]
        if target_col:
            cells.insert(9, fmt(r["time_to_target_s"], ".1f"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ps_server.py runs.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories containing events.jsonl")
    parser.add_argument("--target-loss", type=float, default=None, help="report wall-clock to this eval loss")
    parser.add_argument("--out", type=Path, default=None, help="also write the table to this file")
    args = parser.parse_args()
    rows = [summarize(load_run(d), args.target_loss) for d in args.runs]
    table = markdown(rows, args.target_loss)
    print(table)
    print("\nRuns: " + ", ".join(f"{r['arm']} = {r['run']} ({r['workers']} workers)" for r in rows))
    if args.out:
        args.out.write_text(table + "\n")


if __name__ == "__main__":
    main()
