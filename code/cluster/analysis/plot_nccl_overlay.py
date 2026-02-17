#!/usr/bin/env python3
"""
Overlay plot for nccl-tests structured JSON outputs.

This is intentionally simple: compare bandwidth curves across runs to spot
regime shifts (e.g., bimodal performance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def _load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise SystemExit(f"{path}: expected non-empty JSON key 'results'")
    return results


def _sorted_points(results: list[dict], key: str) -> tuple[list[float], list[float]]:
    pts = []
    for r in results:
        size = r.get("size_bytes")
        val = r.get(key)
        if not isinstance(size, (int, float)) or not isinstance(val, (int, float)):
            continue
        if size <= 0:
            continue
        pts.append((float(size), float(val)))
    pts.sort(key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


def main() -> None:
    apply_plot_style()
    p = argparse.ArgumentParser(
        description="Overlay NCCL busbw/algbw curves from structured nccl-tests JSON files."
    )
    p.add_argument(
        "--inputs",
        required=True,
        help="Comma-separated list of structured JSON paths (each must contain key 'results').",
    )
    p.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels matching --inputs (optional; defaults to file stem).",
    )
    p.add_argument(
        "--metric",
        default="busbw_gbps",
        choices=["busbw_gbps", "algbw_gbps"],
        help="Metric to plot (default: busbw_gbps).",
    )
    p.add_argument("--title", default="NCCL bandwidth overlay", help="Plot title.")
    p.add_argument("--out", required=True, help="Output PNG path.")
    args = p.parse_args()

    input_paths = [Path(tok.strip()) for tok in args.inputs.split(",") if tok.strip()]
    if not input_paths:
        raise SystemExit("No --inputs provided")

    labels = [tok.strip() for tok in args.labels.split(",") if tok.strip()] if args.labels else []
    if labels and len(labels) != len(input_paths):
        raise SystemExit(f"--labels count ({len(labels)}) must match --inputs count ({len(input_paths)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, path in enumerate(input_paths):
        results = _load_results(path)
        xs_bytes, ys = _sorted_points(results, args.metric)
        if not xs_bytes:
            continue

        # Log scale in bytes keeps nccl-tests' power-of-two sizes nicely spaced.
        label = labels[i] if labels else path.stem
        ax.plot(xs_bytes, ys, marker="o", linewidth=2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Message size (bytes, log scale)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(args.title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

