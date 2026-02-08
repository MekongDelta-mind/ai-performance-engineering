#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_records(records: List[dict], test: str) -> Dict[Tuple[str, str], List[dict]]:
    out: Dict[Tuple[str, str], List[dict]] = {}
    for r in records:
        if r.get("test") != test:
            continue
        host_mem = str(r.get("host_mem", ""))
        direction = str(r.get("direction", ""))
        out.setdefault((host_mem, direction), []).append(r)
    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda x: int(x.get("size_bytes", 0) or 0))
    return out


def _plot_bw(meta: dict, out_path: Path) -> None:
    records = meta.get("records") or []
    groups = _group_records(records, "bw")
    if not groups:
        raise SystemExit("No bandwidth records found in input JSON.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for (host_mem, direction), rows in sorted(groups.items()):
        xs = [int(r["size_bytes"]) / (1024.0 * 1024.0) for r in rows]
        ys = [float(r.get("bw_gbps") or 0.0) for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{direction} {host_mem}")

    title = f"CPU<->GPU memcpy bandwidth ({meta.get('device_name','')})"
    ax.set_title(title)
    ax.set_xlabel("Size (MiB)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_lat(meta: dict, out_path: Path) -> None:
    records = meta.get("records") or []
    groups = _group_records(records, "lat")
    if not groups:
        raise SystemExit("No latency records found in input JSON.")

    fig, ax = plt.subplots(figsize=(10, 6))
    for (host_mem, direction), rows in sorted(groups.items()):
        xs = [int(r["size_bytes"]) for r in rows]
        ys = [float(r.get("lat_us") or 0.0) for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{direction} {host_mem}")

    title = f"CPU<->GPU memcpy latency ({meta.get('device_name','')})"
    ax.set_title(title)
    ax.set_xlabel("Size (bytes)")
    ax.set_ylabel("Avg latency (us)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot CPU<->GPU memcpy (C2C) results.")
    p.add_argument("--input", required=True, help="Structured JSON from run_c2c_memcpy_bench.sh")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", required=True, help="Prefix for figure filenames")
    args = p.parse_args()

    inp = Path(args.input)
    meta = _load(inp)

    out_dir = Path(args.out_dir)
    run_id = args.run_id
    _plot_bw(meta, out_dir / f"{run_id}_c2c_memcpy_bw.png")
    _plot_lat(meta, out_dir / f"{run_id}_c2c_memcpy_lat.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

