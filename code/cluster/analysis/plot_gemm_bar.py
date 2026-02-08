#!/usr/bin/env python3
"""
Bar plot for GEMM CSV outputs (e.g. per-GPU isolation runs).

This is intentionally minimal and uses matplotlib + stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_RE_NODE_GPU = re.compile(r"^(?P<node>node\\d+)_gpu(?P<gpu>\\d+)$")


def _label_sort_key(label: str) -> Tuple[int, int, str]:
    m = _RE_NODE_GPU.match(label)
    if m:
        node = m.group("node")
        gpu = int(m.group("gpu"))
        node_id = int(node.replace("node", ""))
        return (0, node_id * 100 + gpu, label)
    return (1, 0, label)


def _to_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Bar plot GEMM TFLOPS by label.")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more GEMM CSV paths")
    ap.add_argument("--output", required=True, help="Output PNG path")
    ap.add_argument("--filter-m", type=int, default=0, help="Only include rows where M=N=K equals this (0=all)")
    ap.add_argument(
        "--value",
        default="avg_tflops",
        choices=["avg_tflops", "p50_tflops", "p99_tflops"],
        help="Which metric to plot",
    )
    ap.add_argument("--title", default="GEMM TFLOPS by label", help="Plot title")
    args = ap.parse_args()

    all_rows: List[Dict[str, str]] = []
    for raw in args.inputs:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"ERROR: input not found: {p}")
        all_rows.extend(_read_rows(p))

    rows: List[Tuple[str, float]] = []
    for r in all_rows:
        label = (r.get("label") or "").strip()
        if not label:
            continue
        m = _to_int(r.get("m"))
        n = _to_int(r.get("n"))
        k = _to_int(r.get("k"))
        if args.filter_m:
            if m != args.filter_m or n != args.filter_m or k != args.filter_m:
                continue
        v = _to_float(r.get(args.value))
        if v is None:
            continue
        rows.append((label, v))

    if not rows:
        raise SystemExit("ERROR: no rows to plot (check --filter-m and --value)")

    rows.sort(key=lambda t: _label_sort_key(t[0]))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * len(rows)), 4.2))
    ax.bar(range(len(rows)), vals, color="#1f77b4")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TFLOPS (bf16)")
    ax.set_title(args.title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

