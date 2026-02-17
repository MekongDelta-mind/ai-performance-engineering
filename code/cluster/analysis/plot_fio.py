#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def main() -> int:
    apply_plot_style()
    p = argparse.ArgumentParser(description="Plot fio summary JSON from scripts/run_fio_bench.sh")
    p.add_argument("--input", required=True, help="Path to structured fio summary JSON")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--title", default="fio Storage Benchmark", help="Plot title")
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text())
    res = data.get("results", {})

    seq_write = float(res.get("seq_write", {}).get("bw_mb_s", 0.0) or 0.0)
    seq_read = float(res.get("seq_read", {}).get("bw_mb_s", 0.0) or 0.0)
    rand_read_iops = float(res.get("rand_read", {}).get("iops", 0.0) or 0.0)
    rand_write_iops = float(res.get("rand_write", {}).get("iops", 0.0) or 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_bw, ax_iops = axes

    ax_bw.bar(["seq_write", "seq_read"], [seq_write, seq_read])
    ax_bw.set_ylabel("MB/s")
    ax_bw.set_title("Sequential")
    ax_bw.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax_iops.bar(["rand_read", "rand_write"], [rand_read_iops, rand_write_iops])
    ax_iops.set_ylabel("IOPS")
    ax_iops.set_title("Random (4K)")
    ax_iops.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

