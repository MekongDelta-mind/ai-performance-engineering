#!/usr/bin/env python3
"""Plot transient node2_gpu2 GEMM collapse and recovery across three runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    (
        "r2_anomaly",
        ROOT / "results/structured/2026-02-08_ssh_key_full_suite_r2_node2_gemm_gpu_sanity.csv",
        "r2 anomaly",
    ),
    (
        "diag_recovery",
        ROOT / "results/structured/2026-02-08_node2_gpu2_diag_pre_reset_node2_gemm_gpu_sanity.csv",
        "isolated rerun",
    ),
    (
        "clean_suite",
        ROOT / "results/structured/2026-02-08_ssh_key_full_suite_clean_node2_gemm_gpu_sanity.csv",
        "clean suite",
    ),
]

FIG_OUT = ROOT / "docs/figures/2026-02-08_node2_gpu2_transient_gemm_tflops.png"
JSON_OUT = ROOT / "results/structured/2026-02-08_node2_gpu2_transient_gemm_tflops.json"


def _load(csv_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip()
            out[label] = float(row["avg_tflops"])
    return out


def main() -> None:
    apply_plot_style()
    labels = ["node2_gpu0", "node2_gpu1", "node2_gpu2", "node2_gpu3"]
    loaded = []
    for key, path, pretty in RUNS:
        vals = _load(path)
        loaded.append((key, pretty, path, vals))

    payload = {"sources": {}, "avg_tflops": {}, "node2_gpu2": {}, "drop_and_recovery": {}}
    for key, pretty, path, vals in loaded:
        payload["sources"][key] = str(path)
        payload["avg_tflops"][key] = {k: vals[k] for k in labels}
        payload["node2_gpu2"][key] = vals["node2_gpu2"]

    baseline = payload["node2_gpu2"]["r2_anomaly"]
    recovered = payload["node2_gpu2"]["diag_recovery"]
    clean = payload["node2_gpu2"]["clean_suite"]
    payload["drop_and_recovery"] = {
        "r2_to_diag_gain_tflops": recovered - baseline,
        "r2_to_diag_gain_x": recovered / baseline,
        "r2_to_clean_gain_tflops": clean - baseline,
        "r2_to_clean_gain_x": clean / baseline,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    for idx, (key, pretty, _path, vals) in enumerate(loaded):
        y = [vals[g] for g in labels]
        ax.bar(x + (idx - 1) * width, y, width=width, label=pretty, color=colors[idx], alpha=0.9)

    # highlight gpu2 lane
    ax.axvspan(1.5, 2.5, color="#f2f2f2", alpha=0.4)
    ax.text(
        2.0,
        max(payload["node2_gpu2"].values()) * 1.02,
        "node2_gpu2 transient lane",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average TFLOPS")
    ax.set_title("Transient node2_gpu2 GEMM collapse recovered without reset")
    ax.legend(loc="upper left")

    ax.text(
        0.02,
        0.98,
        f"gpu2: {baseline:.1f} -> {recovered:.1f} -> {clean:.1f} TFLOPS",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=160)
    plt.close(fig)
    print(f"Wrote: {FIG_OUT}")
    print(f"Wrote: {JSON_OUT}")


if __name__ == "__main__":
    main()
