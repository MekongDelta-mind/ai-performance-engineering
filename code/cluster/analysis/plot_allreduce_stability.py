#!/usr/bin/env python3
"""
Plot all-reduce stability profiling results.

Generates:
1. Per-iteration bandwidth time series (shows jitter/instability)
2. Bandwidth histogram (shows distribution shape - bimodal = bad)

Usage:
  python analysis/plot_allreduce_stability.py \
    --input results/structured/allreduce_stability.json \
    --output docs/figures/allreduce_stability.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot all-reduce stability results")
    ap.add_argument("--input", required=True, help="Input JSON from allreduce_stability_bench.py")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--title", default=None, help="Plot title (auto-generated if omitted)")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_iter = data.get("per_iteration", {})
    busbw = per_iter.get("busbw_gbps", [])
    summary = data.get("summary", {})

    if not busbw:
        raise SystemExit("ERROR: No per-iteration data in input")

    payload_gib = data.get("payload_gib", "?")
    world_size = data.get("world_size", "?")
    cv = summary.get("busbw_cv_pct", 0)
    jitter = summary.get("jitter_assessment", "unknown")
    mean_bw = summary.get("busbw_mean_gbps", 0)
    p50_bw = summary.get("busbw_p50_gbps", 0)
    p99_bw = summary.get("busbw_p99_gbps", 0)
    p01_bw = summary.get("busbw_p01_gbps", 0)

    title = args.title or f"All-Reduce Stability ({payload_gib} GiB, {world_size} ranks)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: time series
    iterations = list(range(1, len(busbw) + 1))
    ax1.plot(iterations, busbw, linewidth=0.8, alpha=0.7, color="#1f77b4")
    ax1.axhline(y=mean_bw, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_bw:.1f} GBps")
    ax1.axhline(y=p50_bw, color="green", linestyle=":", linewidth=1, label=f"P50: {p50_bw:.1f} GBps")

    # Shade P01-P99 band
    ax1.axhspan(p01_bw, p99_bw, alpha=0.1, color="blue", label=f"P01-P99: [{p01_bw:.1f}, {p99_bw:.1f}]")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Bus Bandwidth (GBps)")
    ax1.set_title(f"Per-Iteration Bandwidth\nCV={cv:.2f}%, Jitter={jitter}")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Right: histogram
    ax2.hist(busbw, bins=50, color="#1f77b4", edgecolor="black", linewidth=0.3, alpha=0.8)
    ax2.axvline(x=mean_bw, color="red", linestyle="--", linewidth=1, label=f"Mean: {mean_bw:.1f}")
    ax2.axvline(x=p50_bw, color="green", linestyle=":", linewidth=1, label=f"P50: {p50_bw:.1f}")

    ax2.set_xlabel("Bus Bandwidth (GBps)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Bandwidth Distribution\n(bimodal shape = routing/congestion issue)")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
