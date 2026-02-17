#!/usr/bin/env python3
"""
Plot NCCL algorithm comparison results.

Overlays bandwidth vs message size curves for each tested algorithm
(Ring, Tree, NVLS, auto) to show which algorithm wins at each size.

Usage:
  python analysis/plot_nccl_algo_comparison.py \
    --inputs results/structured/*_nccl_algo_*.json \
    --output docs/figures/nccl_algo_comparison.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


ALGO_COLORS = {
    "ring": "#1f77b4",
    "tree": "#ff7f0e",
    "nvls": "#2ca02c",
    "auto": "#d62728",
    "collnetdirect": "#9467bd",
    "collnetchain": "#8c564b",
}


def _extract_algo(path: Path) -> Optional[str]:
    """Extract algorithm name from filename like *_nccl_algo_ring.json."""
    m = re.search(r"nccl_algo_(\w+)\.json$", path.name)
    if m:
        return m.group(1)
    return None


def main() -> int:
    apply_plot_style()
    ap = argparse.ArgumentParser(description="Plot NCCL algorithm comparison")
    ap.add_argument("--inputs", nargs="+", required=True, help="NCCL algo JSON files")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--title", default="NCCL Algorithm Comparison", help="Plot title")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for input_path in sorted(args.inputs):
        p = Path(input_path)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping")
            continue

        algo = _extract_algo(p)
        if algo is None:
            # Skip comparison summary files
            if "comparison" in p.name:
                continue
            algo = p.stem
        elif algo.lower() == "comparison":
            # Skip top-level comparison summary files that do not contain per-size results.
            continue

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            print(f"WARNING: No results in {p}, skipping")
            continue

        # Extract message sizes and bandwidth
        sizes = []
        busbw = []
        algbw = []
        for r in results:
            size = r.get("size_bytes", 0)
            bw = r.get("busbw_gbps", 0)
            abw = r.get("algbw_gbps", 0)
            if size > 0:
                sizes.append(size)
                busbw.append(bw)
                algbw.append(abw)

        if not sizes:
            continue

        color = ALGO_COLORS.get(algo, "#333333")
        label = algo.upper()
        peak = max(busbw) if busbw else 0

        # Left: bus bandwidth vs message size
        ax1.plot(
            sizes,
            busbw,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            label=f"{label} (peak={peak:.1f} GB/s)",
            alpha=0.85,
        )

        # Right: algorithm bandwidth vs message size
        ax2.plot(
            sizes,
            algbw,
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color,
            label=f"{label}",
            alpha=0.85,
        )

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Message Size (bytes)")
    ax1.set_ylabel("Bus Bandwidth (GB/s)")
    ax1.set_title("Bus Bandwidth by Algorithm")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Message Size (bytes)")
    ax2.set_ylabel("Algorithm Bandwidth (GB/s)")
    ax2.set_title("Algorithm Bandwidth by Algorithm")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(args.title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
