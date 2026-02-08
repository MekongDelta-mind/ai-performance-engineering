#!/usr/bin/env python3
"""Plot control-plane collective benchmark output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _method_label(method: str) -> str:
    mapping = {
        "all_gather_object": "all_gather_object",
        "all_gather_tensor": "all_gather(tensor)",
        "all_reduce_tensor": "all_reduce(tensor)",
    }
    return mapping.get(method, method)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot all-gather control-plane benchmark")
    parser.add_argument("--input", required=True, help="Input JSON from scripts/allgather_control_plane_bench.py")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Optional title override")
    args = parser.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    methods = data.get("methods", {})
    order = ["all_gather_object", "all_gather_tensor", "all_reduce_tensor"]
    labels = []
    mean_ms = []
    p99_ms = []
    for method in order:
        entry = methods.get(method)
        if not entry:
            continue
        labels.append(_method_label(method))
        mean_ms.append(float(entry.get("latency_ms", {}).get("mean", 0.0)))
        p99_ms.append(float(entry.get("latency_ms", {}).get("p99", 0.0)))

    title = args.title or (
        "Control-Plane Collective Latency "
        f"({data.get('world_size', '?')} ranks, {data.get('iters', '?')} iters)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    bars0 = axes[0].bar(labels, mean_ms, color=["#d62728", "#1f77b4", "#2ca02c"])  # red, blue, green
    axes[0].set_title("Mean Latency")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars0, mean_ms):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    bars1 = axes[1].bar(labels, p99_ms, color=["#d62728", "#1f77b4", "#2ca02c"])  # red, blue, green
    axes[1].set_title("P99 Latency")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars1, p99_ms):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    comp = data.get("comparison", {})
    speedup_ag = comp.get("all_gather_object_over_all_gather_tensor_speedup")
    speedup_ar = comp.get("all_gather_object_over_all_reduce_tensor_speedup")
    notes = []
    if speedup_ag is not None:
        notes.append(f"obj / gather(tensor): {float(speedup_ag):.2f}x")
    if speedup_ar is not None:
        notes.append(f"obj / reduce(tensor): {float(speedup_ar):.2f}x")
    if notes:
        fig.text(0.5, 0.01, " | ".join(notes), ha="center", va="bottom", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
