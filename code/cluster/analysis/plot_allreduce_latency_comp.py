#!/usr/bin/env python3
"""
Plot all-reduce latency comparison output.

Shows:
1) Mean bus bandwidth for one-large vs many-small
2) Mean duration for one-large vs many-small
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def main() -> int:
    apply_plot_style()
    parser = argparse.ArgumentParser(description="Plot all-reduce latency comparison")
    parser.add_argument("--input", required=True, help="Input JSON from scripts/allreduce_latency_comp.py")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Optional plot title")
    args = parser.parse_args()

    in_path = Path(args.input)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    cases = data.get("cases", {})
    one = cases.get("one_large", {})
    many = cases.get("many_small", {})
    comp = data.get("comparison", {})

    one_bw = float(one.get("busbw_gbps", {}).get("mean", 0.0))
    many_bw = float(many.get("busbw_gbps", {}).get("mean", 0.0))
    one_d = float(one.get("duration_ms", {}).get("mean", 0.0))
    many_d = float(many.get("duration_ms", {}).get("mean", 0.0))
    ratio_bw = comp.get("bandwidth_ratio_large_over_small")
    ratio_d = comp.get("duration_ratio_small_over_large")
    chunks = data.get("chunks", "?")
    payload = data.get("payload_target_gib", "?")
    ranks = data.get("world_size", "?")

    title = args.title or f"All-Reduce Latency Comparison ({payload} GiB total, {chunks} chunks, {ranks} ranks)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    labels = ["1x large", f"{chunks}x small"]
    bw_vals = [one_bw, many_bw]
    d_vals = [one_d, many_d]

    bars0 = axes[0].bar(labels, bw_vals, color=["#2ca02c", "#1f77b4"])
    axes[0].set_ylabel("Bus Bandwidth (GBps)")
    axes[0].set_title("Bandwidth")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars0, bw_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    if ratio_bw is not None:
        axes[0].text(0.5, 0.96, f"large/small: {ratio_bw:.3f}x", transform=axes[0].transAxes, ha="center", va="top", fontsize=9)

    bars1 = axes[1].bar(labels, d_vals, color=["#2ca02c", "#1f77b4"])
    axes[1].set_ylabel("Duration (ms)")
    axes[1].set_title("Duration")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, val in zip(bars1, d_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    if ratio_d is not None:
        axes[1].text(0.5, 0.96, f"small/large: {ratio_d:.3f}x", transform=axes[1].transAxes, ha="center", va="top", fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
