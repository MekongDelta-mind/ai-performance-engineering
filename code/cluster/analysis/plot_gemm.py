#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise SystemExit("No rows found in GEMM CSV.")
    return rows


def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def plot_tflops(rows, out_path: Path, title: str):
    rows = sorted(rows, key=lambda r: int(r.get("m", 0)))
    sizes = [int(r["m"]) for r in rows]
    avg = [to_float(r.get("avg_tflops")) for r in rows]
    p50 = [to_float(r.get("p50_tflops")) for r in rows]
    p99 = [to_float(r.get("p99_tflops")) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, avg, marker="o", label="avg")
    ax.plot(sizes, p50, marker="o", label="p50")
    ax.plot(sizes, p99, marker="o", label="p99")

    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel("TFLOPS (bf16)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot GEMM microbench results.")
    parser.add_argument("--input", required=True, help="Path to GEMM CSV")
    parser.add_argument("--out-dir", required=True, help="Directory for output figures")
    parser.add_argument("--run-id", default="run", help="Run id prefix for file names")
    args = parser.parse_args()

    data_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_path)
    out_path = out_dir / f"{args.run_id}_gemm_tflops.png"
    plot_tflops(rows, out_path, "BF16 GEMM throughput")


if __name__ == "__main__":
    main()
