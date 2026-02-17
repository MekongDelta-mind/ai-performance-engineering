#!/usr/bin/env python3
"""
Plot MAMF (Maximum Achievable Matmul FLOPS) results.

Generates two views:
1. Heatmap of TFLOPS across shapes (if 2D range was scanned)
2. Bar chart comparing MAMF across GPUs (straggler detection)

Usage:
  python analysis/plot_mamf.py \\
    --inputs results/structured/*_mamf.csv \\
    --output docs/figures/mamf_comparison.png

  python analysis/plot_mamf.py \\
    --summary-inputs results/structured/*_mamf_summary.json \\
    --output docs/figures/mamf_straggler.png \\
    --mode straggler
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from plot_style import apply_plot_style
import numpy as np


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def plot_straggler(summary_paths: List[Path], output: Path, title: str) -> None:
    """Bar chart of MAMF TFLOPS per GPU for straggler detection."""
    labels = []
    mamf_values = []
    shapes = []

    for p in sorted(summary_paths):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        label = data.get("label", p.stem)
        mamf = data.get("mamf_tflops", 0)
        best = data.get("best_shape", {})
        shape_str = f"{best.get('m', '?')}x{best.get('k', '?')}x{best.get('n', '?')}"
        labels.append(label)
        mamf_values.append(mamf)
        shapes.append(shape_str)

    if not labels:
        raise SystemExit("ERROR: No summary data found")

    fig, ax = plt.subplots(figsize=(max(7.5, 0.8 * len(labels)), 5))

    colors = []
    min_val = min(mamf_values)
    max_val = max(mamf_values)
    spread_pct = ((max_val - min_val) / max_val * 100) if max_val > 0 else 0

    for v in mamf_values:
        if v == min_val and spread_pct > 2:
            colors.append("#d62728")  # Red for straggler
        elif v == max_val:
            colors.append("#2ca02c")  # Green for fastest
        else:
            colors.append("#1f77b4")  # Blue for others

    bars = ax.bar(range(len(labels)), mamf_values, color=colors)

    # Annotate bars with best shape
    for i, (bar, shape) in enumerate(zip(bars, shapes)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            f"{mamf_values[i]:.1f}\n{shape}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAMF (TFLOPS)")
    ax.set_title(f"{title}\nSpread: {spread_pct:.1f}% (min={min_val:.1f}, max={max_val:.1f})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Fastest"),
        Patch(facecolor="#1f77b4", label="Normal"),
    ]
    if spread_pct > 2:
        legend_elements.append(Patch(facecolor="#d62728", label=f"Straggler ({spread_pct:.1f}% slower)"))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved: {output}")


def plot_shape_sweep(csv_paths: List[Path], output: Path, title: str) -> None:
    """Line plot showing TFLOPS vs M dimension (or heatmap if 2D)."""
    all_rows: List[Dict[str, str]] = []
    for p in csv_paths:
        all_rows.extend(_read_csv(p))

    if not all_rows:
        raise SystemExit("ERROR: No CSV data found")

    # Group by label
    by_label: Dict[str, List[Tuple[int, float]]] = {}
    for row in all_rows:
        label = (row.get("label") or "").strip()
        m = _to_float(row.get("m"))
        tflops = _to_float(row.get("max_tflops"))
        if m is None or tflops is None:
            continue
        if label not in by_label:
            by_label[label] = []
        by_label[label].append((int(m), tflops))

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, points in sorted(by_label.items()):
        points.sort(key=lambda x: x[0])
        ms = [p[0] for p in points]
        tflops = [p[1] for p in points]
        ax.plot(ms, tflops, marker="o", markersize=3, label=label, alpha=0.8)

    ax.set_xlabel("M dimension")
    ax.set_ylabel("Max TFLOPS")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> int:
    apply_plot_style()
    ap = argparse.ArgumentParser(description="Plot MAMF results")
    ap.add_argument("--inputs", nargs="*", default=[], help="MAMF CSV file(s)")
    ap.add_argument("--summary-inputs", nargs="*", default=[], help="MAMF summary JSON file(s)")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--title", default="MAMF (Maximum Achievable Matmul FLOPS)", help="Plot title")
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "straggler", "sweep"],
        help="Plot mode (default: auto)",
    )
    args = ap.parse_args()

    csv_paths = [Path(p) for p in args.inputs if Path(p).exists()]
    summary_paths = [Path(p) for p in args.summary_inputs if Path(p).exists()]

    mode = args.mode
    if mode == "auto":
        mode = "straggler" if summary_paths else "sweep"

    if mode == "straggler":
        if not summary_paths:
            raise SystemExit("ERROR: --summary-inputs required for straggler mode")
        plot_straggler(summary_paths, Path(args.output), args.title)
    else:
        if not csv_paths:
            raise SystemExit("ERROR: --inputs required for sweep mode")
        plot_shape_sweep(csv_paths, Path(args.output), args.title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
