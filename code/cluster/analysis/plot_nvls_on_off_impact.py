#!/usr/bin/env python3
"""Plot NCCL all-reduce bus bandwidth impact of NVLS on vs off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


ROOT = Path(__file__).resolve().parents[1]
ON_SUMMARY = ROOT / "results/structured/2026-02-08_032814_cloud_eval_full_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json"
OFF_SUMMARY = ROOT / "results/structured/2026-02-08_031531_health_suite_extended_nvls0_node1node2_cluster_health_suite_summary.json"
OUT_FIG = ROOT / "docs/figures/2026-02-08_nvls_on_off_allreduce_busbw.png"
OUT_JSON = ROOT / "results/structured/2026-02-08_nvls_on_off_allreduce_busbw.json"


def _extract(summary_path: Path) -> tuple[float, int]:
    data = json.loads(summary_path.read_text())
    max_busbw = data["nccl"]["all_reduce_perf"]["max_busbw"]
    return float(max_busbw["busbw_gbps"]), int(max_busbw["size_bytes"])


def main() -> None:
    apply_plot_style()
    on_bw, on_size = _extract(ON_SUMMARY)
    off_bw, off_size = _extract(OFF_SUMMARY)
    if on_size != off_size:
        raise ValueError(f"Message size mismatch: on={on_size}, off={off_size}")

    drop_pct = (on_bw - off_bw) / on_bw * 100.0
    payload = {
        "nvls_on_summary": str(ON_SUMMARY),
        "nvls_off_summary": str(OFF_SUMMARY),
        "size_bytes": on_size,
        "nvls_on_busbw_gbps": round(on_bw, 2),
        "nvls_off_busbw_gbps": round(off_bw, 2),
        "drop_percent": round(drop_pct, 2),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["NVLS on", "NVLS off"]
    vals = [on_bw, off_bw]
    colors = ["#1f77b4", "#d62728"]
    bars = ax.bar(labels, vals, color=colors)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value:.2f}", ha="center", va="bottom")
    ax.set_ylabel("All-reduce max bus bandwidth (GB/s)")
    ax.set_title(f"NCCL all-reduce @ {on_size // (1024**3)} GiB message size")
    ax.text(0.5, max(vals) * 0.60, f"Drop with NVLS off: {drop_pct:.2f}%", ha="center", va="center", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.20)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=160)
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()
