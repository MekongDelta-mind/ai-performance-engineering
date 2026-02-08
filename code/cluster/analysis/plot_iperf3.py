#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot iperf3 OOB TCP throughput from a cluster health suite summary JSON."
    )
    ap.add_argument("--summary", required=True, help="Path to *_cluster_health_suite_summary.json")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="", help="Optional plot title")
    ap.add_argument(
        "--skip-if-missing",
        action="store_true",
        help="Exit 0 (skip) if iperf3 fwd/rev metrics are missing in the summary.",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON path for a small structured iperf3 summary payload.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    ip = summary.get("iperf3") or {}
    fwd = _to_float((ip.get("fwd") or {}).get("gbps"))
    rev = _to_float((ip.get("rev") or {}).get("gbps"))

    if fwd is None and rev is None:
        if args.skip_if_missing:
            print(f"plot_iperf3: no iperf3.*.gbps found in {summary_path}; skipping.")
            return 0
        raise SystemExit(f"No iperf3.*.gbps found in {summary_path}")

    labels = []
    vals = []
    if fwd is not None:
        labels.append("fwd")
        vals.append(fwd)
    if rev is not None:
        labels.append("rev")
        vals.append(rev)

    title = args.title.strip()
    if not title:
        oob_if = summary.get("oob_if") or "<unknown>"
        hosts = summary.get("hosts") or []
        host_str = ",".join(hosts) if hosts else "<unknown>"
        title = f"iperf3 OOB TCP throughput ({host_str}, {oob_if})"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(labels, vals, color=["#4C78A8", "#F58518"][: len(vals)])
    ax.set_ylabel("Gbps")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ymax = max(vals) * 1.20 if vals else 1.0
    ax.set_ylim(0, ymax)

    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + ymax * 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if args.out_json:
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": summary.get("run_id"),
            "hosts": summary.get("hosts"),
            "oob_if": summary.get("oob_if"),
            "iperf3": {"fwd_gbps": fwd, "rev_gbps": rev},
            "source_summary_json": str(summary_path),
        }
        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

