#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    p = argparse.ArgumentParser(description="Plot NUMA memory memcpy bandwidth probe results.")
    p.add_argument("--input", required=True, help="Structured JSON from run_numa_mem_bw.sh")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="")
    args = p.parse_args()

    inp = Path(args.input)
    d = json.loads(inp.read_text(encoding="utf-8"))
    results = d.get("results") or []

    # Filter only successful measurements.
    rows = []
    for r in results:
        if r.get("error"):
            continue
        try:
            node = int(r.get("node"))
            bw = float(r.get("bw_gbps"))
        except Exception:
            continue
        rows.append((node, bw))

    if not rows:
        raise SystemExit("No successful NUMA bandwidth rows to plot.")

    rows.sort(key=lambda x: x[0])
    nodes = [n for n, _ in rows]
    bws = [bw for _, bw in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(n) for n in nodes], bws)
    title = args.title or f"NUMA memcpy BW ({d.get('label','')})"
    ax.set_title(title)
    ax.set_xlabel("NUMA node")
    ax.set_ylabel("Memcpy throughput (GB/s)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

