#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def main() -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Plot GPU STREAM-style bandwidth results.")
    p.add_argument("--input", required=True, help="Structured JSON from run_gpu_stream_bench.sh")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="", help="Optional chart title")
    args = p.parse_args()

    inp = Path(args.input)
    payload = json.loads(inp.read_text(encoding="utf-8"))
    rows = payload.get("operations") or []
    if not rows:
        raise SystemExit(f"No operations in {inp}")

    ops = [str(r.get("operation", "")) for r in rows]
    bws = [float(r.get("bandwidth_gbps", 0.0)) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ops, bws, color="#1F77B4")
    ax.set_ylabel("Bandwidth (GB/s)")
    default_title = (
        f"GPU STREAM-like bandwidth ({payload.get('label', 'unknown')}, "
        f"{payload.get('dtype', 'dtype')}, {payload.get('size_mb', '?')}MB)"
    )
    ax.set_title(args.title or default_title)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, bws):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
