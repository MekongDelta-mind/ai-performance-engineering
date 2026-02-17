#!/usr/bin/env python3
"""
Summarize paired FP4 smoke runs for two nodes and generate a comparison plot.

Expected input naming pattern:
  <run_prefix>_r<round>_<node>_cluster_perf_fp4_smoke.json
  <run_prefix>_r<round>_<node>_cluster_perf_fp4_smoke_clock_lock.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


ROUND_RE = re.compile(r"_r(\d+)_")


def _extract_round(path: Path) -> int:
    match = ROUND_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse round from filename: {path.name}")
    return int(match.group(1))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _device0_current_sm_mhz(clock_data: dict) -> float | None:
    for lock in clock_data.get("locks", []):
        if int(lock.get("device", -1)) == 0:
            clocks = lock.get("clocks", {})
            return float(clocks["current_sm_mhz"]) if "current_sm_mhz" in clocks else None
    return None


def _collect_node_rounds(base_dir: Path, run_prefix: str, node: str) -> dict[int, dict]:
    pattern = f"{run_prefix}_r*_{node}_cluster_perf_fp4_smoke.json"
    files = sorted(base_dir.glob(pattern), key=_extract_round)
    out: dict[int, dict] = {}

    for smoke_path in files:
        round_id = _extract_round(smoke_path)
        clock_path = smoke_path.with_name(smoke_path.stem + "_clock_lock.json")
        smoke = _load_json(smoke_path)
        clock = _load_json(clock_path) if clock_path.exists() else {}

        deepgemm = smoke["results"]["deepgemm_fp8_fp4"]["avg_tflops"]
        torch_bf16 = smoke["results"]["torch_bf16_baseline"]["avg_tflops"]
        speedup = smoke["results"]["deepgemm_over_torch_bf16_speedup"]

        out[round_id] = {
            "smoke_json": str(smoke_path),
            "clock_lock_json": str(clock_path) if clock_path.exists() else None,
            "status": smoke.get("status", "unknown"),
            "deepgemm_avg_tflops": float(deepgemm),
            "torch_bf16_avg_tflops": float(torch_bf16),
            "speedup_over_torch_bf16": float(speedup),
            "device0_current_sm_mhz": _device0_current_sm_mhz(clock),
            "clock_lock_returncode": clock.get("returncode"),
        }
    return out


def _build_summary(node1_rounds: dict[int, dict], node2_rounds: dict[int, dict]) -> dict:
    common_rounds = sorted(set(node1_rounds.keys()) & set(node2_rounds.keys()))
    if not common_rounds:
        raise SystemExit("ERROR: No paired rounds found between node1 and node2.")

    rounds = []
    node1_tflops = []
    node2_tflops = []
    node1_speedup = []
    node2_speedup = []

    for r in common_rounds:
        n1 = node1_rounds[r]
        n2 = node2_rounds[r]
        n1_t = n1["deepgemm_avg_tflops"]
        n2_t = n2["deepgemm_avg_tflops"]
        delta_pct = ((n2_t - n1_t) / n1_t) * 100.0 if n1_t else None

        rounds.append(
            {
                "round": r,
                "node1": n1,
                "node2": n2,
                "deepgemm_delta_pct_node2_vs_node1": delta_pct,
            }
        )
        node1_tflops.append(n1_t)
        node2_tflops.append(n2_t)
        node1_speedup.append(n1["speedup_over_torch_bf16"])
        node2_speedup.append(n2["speedup_over_torch_bf16"])

    mean_node1 = mean(node1_tflops)
    mean_node2 = mean(node2_tflops)

    return {
        "paired_round_count": len(common_rounds),
        "rounds": rounds,
        "aggregate": {
            "deepgemm_avg_tflops": {
                "node1_mean": mean_node1,
                "node2_mean": mean_node2,
                "node1_median": median(node1_tflops),
                "node2_median": median(node2_tflops),
                "delta_pct_node2_vs_node1_mean": ((mean_node2 - mean_node1) / mean_node1) * 100.0
                if mean_node1
                else None,
            },
            "speedup_over_torch_bf16": {
                "node1_mean": mean(node1_speedup),
                "node2_mean": mean(node2_speedup),
                "node1_median": median(node1_speedup),
                "node2_median": median(node2_speedup),
            },
        },
    }


def _plot(summary: dict, out_png: Path, title: str) -> None:
    rounds = [entry["round"] for entry in summary["rounds"]]
    n1_tflops = [entry["node1"]["deepgemm_avg_tflops"] for entry in summary["rounds"]]
    n2_tflops = [entry["node2"]["deepgemm_avg_tflops"] for entry in summary["rounds"]]
    n1_speedup = [entry["node1"]["speedup_over_torch_bf16"] for entry in summary["rounds"]]
    n2_speedup = [entry["node2"]["speedup_over_torch_bf16"] for entry in summary["rounds"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(rounds, n1_tflops, marker="o", linewidth=1.5, label="node1")
    ax1.plot(rounds, n2_tflops, marker="o", linewidth=1.5, label="node2")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("DeepGEMM FP8xFP4 avg TFLOPS")
    ax1.set_title("DeepGEMM Throughput by Round")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    ax2.plot(rounds, n1_speedup, marker="o", linewidth=1.5, label="node1")
    ax2.plot(rounds, n2_speedup, marker="o", linewidth=1.5, label="node2")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Speedup vs torch BF16")
    ax2.set_title("Relative Speedup by Round")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize paired FP4 smoke runs across two nodes.")
    ap.add_argument(
        "--run-prefix",
        required=True,
        help="Run prefix before _r<round> (example: 2026-02-08_fp4_smoke_node_compare)",
    )
    ap.add_argument(
        "--structured-dir",
        default="results/structured",
        help="Directory containing structured artifacts (default: results/structured)",
    )
    ap.add_argument(
        "--node1-label",
        default="node1",
        help="Node label for first node in filename pattern (default: node1)",
    )
    ap.add_argument(
        "--node2-label",
        default="node2",
        help="Node label for second node in filename pattern (default: node2)",
    )
    ap.add_argument("--output-json", required=True, help="Output summary JSON path")
    ap.add_argument("--output-png", required=True, help="Output comparison PNG path")
    ap.add_argument("--title", default=None, help="Optional plot title")
    args = ap.parse_args()

    base_dir = Path(args.structured_dir)
    node1_rounds = _collect_node_rounds(base_dir, args.run_prefix, args.node1_label)
    node2_rounds = _collect_node_rounds(base_dir, args.run_prefix, args.node2_label)

    summary = _build_summary(node1_rounds, node2_rounds)
    summary["run_prefix"] = args.run_prefix
    summary["node_labels"] = {"node1": args.node1_label, "node2": args.node2_label}

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    title = args.title or f"FP4 smoke paired comparison ({args.node1_label} vs {args.node2_label})"
    _plot(summary, Path(args.output_png), title)

    print(f"Saved: {out_json}")
    print(f"Saved: {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
