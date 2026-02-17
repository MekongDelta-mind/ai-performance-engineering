#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot nvbandwidth SUM metrics from CSV.")
    p.add_argument("--input", required=True, help="Input CSV from run_nvbandwidth_bundle.sh")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--top-k", type=int, default=12, help="Number of highest-SUM tests to plot (default: 12)")
    p.add_argument("--title", default="nvbandwidth SUM metrics (GB/s)", help="Plot title")
    return p.parse_args()


def main() -> int:
    apply_plot_style()
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    latest_by_test: dict[str, float] = {}
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test = str((row.get("test") or "")).strip()
            value = str((row.get("sum_gbps") or "")).strip()
            if not test or not value:
                continue
            latest_by_test[test] = float(value)

    if not latest_by_test:
        raise SystemExit("No SUM values found in CSV.")

    items = sorted(latest_by_test.items(), key=lambda x: x[1], reverse=True)[: max(1, args.top_k)]
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig_h = max(4.0, 0.45 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = list(range(len(labels)))
    ax.barh(y, values, color="#2a78c2")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("SUM bandwidth (GB/s)")
    ax.set_title(args.title)
    ax.grid(axis="x", alpha=0.25)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
