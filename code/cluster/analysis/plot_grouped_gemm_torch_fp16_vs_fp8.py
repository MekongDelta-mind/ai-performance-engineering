#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from plot_style import apply_plot_style


def _shape_label(s: Dict[str, Any]) -> str:
    return f"G={s['groups']} M={s['m']} N={s['n']} K={s['k']}"


def main() -> int:
    apply_plot_style()
    ap = argparse.ArgumentParser(description="Plot grouped GEMM summary JSON (torch fp16/fp8 + DeepGEMM when available).")
    ap.add_argument(
        "--summary-json",
        required=True,
        help="Input summary JSON from summarize_grouped_gemm_torch_fp16_vs_fp8.py",
    )
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument(
        "--title",
        default="Grouped GEMM: Torch FP16 vs Torch FP8 vs DeepGEMM FP8xFP4",
        help="Figure title (subtitle will include DeepGEMM status when available)",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.summary_json).read_text())
    rows: List[Dict[str, Any]] = data.get("shapes") or []
    if not rows:
        raise SystemExit("No shapes in summary JSON")

    # Sort shapes by fp16/fp8 ratio descending to make the worst cases obvious.
    def _has_deepgemm(r: Dict[str, Any]) -> bool:
        d = r.get("deepgemm_fp8") or {}
        return (d.get("status") == "ok") and (d.get("tflops") is not None)

    deepgemm_present = any(_has_deepgemm(r) for r in rows)

    def ratio_key(r: Dict[str, Any]) -> float:
        ratios = r.get("ratios") or {}
        if deepgemm_present:
            v = ratios.get("deepgemm_over_torch_fp8")
        else:
            v = ratios.get("fp16_over_fp8")
        return float(v) if v is not None else -1.0

    rows = sorted(rows, key=ratio_key, reverse=True)

    fp16 = []
    fp8 = []
    deep = []
    ratio = []
    labels = []
    for r in rows:
        s = r["shape"]
        labels.append(_shape_label(s))
        fp16.append(float((r.get("torch_fp16") or {}).get("tflops") or 0.0))
        fp8.append(float((r.get("torch_fp8") or {}).get("tflops") or 0.0))
        d = r.get("deepgemm_fp8") or {}
        deep.append(float(d.get("tflops") or 0.0) if d.get("status") == "ok" else 0.0)
        ratios = r.get("ratios") or {}
        if deepgemm_present:
            ratio.append(float((ratios.get("deepgemm_over_torch_fp8") or 0.0)))
        else:
            ratio.append(float((ratios.get("fp16_over_fp8") or 0.0)))

    n = len(rows)
    x = list(range(n))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax0.plot(x, fp16, marker="o", markersize=3, linewidth=1.2, label="torch_fp16", color="#4C78A8")
    ax0.plot(x, fp8, marker="o", markersize=3, linewidth=1.2, label="torch_fp8 (loop baseline)", color="#F58518")
    if deepgemm_present:
        ax0.plot(x, deep, marker="o", markersize=3, linewidth=1.2, label="deepgemm_fp8xfp4", color="#E45756")
    ax0.set_ylabel("TFLOPS")
    ax0.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax0.legend(loc="upper right", frameon=False)

    # Ratio bars.
    ax1.bar(x, ratio, color="#54A24B")
    ax1.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax1.axhline(1.5, color="#E45756", linewidth=1, linestyle=":")
    ax1.axhline(2.0, color="#E45756", linewidth=1, linestyle=":")
    ax1.set_ylabel("DeepGEMM/FP8" if deepgemm_present else "FP16/FP8")
    ax1.set_xlabel("Shape index (sorted by ratio)")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Put a few labels to orient readers without making the x-axis unreadable.
    step = max(1, n // 8)
    xticks = list(range(0, n, step))
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([labels[i] for i in xticks], rotation=20, ha="right")

    # Title includes DeepGEMM status when known.
    deepgemm = data.get("deepgemm") or {}
    d_reason = deepgemm.get("unsupported_reason")
    subtitle = None
    if d_reason:
        subtitle = f"DeepGEMM unsupported: {d_reason}"
    elif deepgemm_present:
        subtitle = "Ratio bars: DeepGEMM TFLOPS / Torch FP8 TFLOPS"
    else:
        subtitle = "Ratio bars: Torch FP16 TFLOPS / Torch FP8 TFLOPS"

    if subtitle:
        fig.suptitle(f"{args.title}\n{subtitle}")
    else:
        fig.suptitle(args.title)

    fig.tight_layout(rect=[0, 0.0, 1, 0.93])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
