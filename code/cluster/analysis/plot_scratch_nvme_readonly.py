#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _node_label(node: str) -> str:
    return node


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot scratch NVMe read-only fio summary JSON")
    ap.add_argument("--summary-json", required=True, help="Input summary JSON from summarize_scratch_nvme_readonly.py")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--title", default="Scratch NVMe Read-Only Performance", help="Figure title")
    args = ap.parse_args()

    data = json.loads(Path(args.summary_json).read_text())
    measurements = data.get("measurements") or {}
    estimates = data.get("estimates") or {}

    nodes = sorted(measurements.keys())
    if not nodes:
        raise SystemExit("No measurements in summary JSON")

    # For each node, pick representative device from estimates (highest seqread bw).
    seq_single = []
    rr_iops_single = []
    seq_raid_cons = []
    seq_raid_lin = []
    rr_raid_cons = []
    rr_raid_lin = []
    xticks = []

    for node in nodes:
        node_meas = measurements.get(node) or {}
        rep = (estimates.get(node) or {}).get("representative_device")
        if not rep or rep not in node_meas:
            # Fall back to first device with seqread.
            rep = None
            for dev, tests in node_meas.items():
                if "seqread" in tests and "randread4k" in tests:
                    rep = dev
                    break
        if not rep:
            continue

        seq = node_meas[rep]["seqread"]
        rr = node_meas[rep]["randread4k"]

        seq_single.append(float(seq["bw_gib_s"]))
        rr_iops_single.append(float(rr["iops"]) / 1e6)  # MIOPS

        raid = (estimates.get(node) or {}).get("raid0") or {}
        seq_raid_cons.append(float(raid.get("seqread_gib_s_conservative") or 0.0))
        seq_raid_lin.append(float(raid.get("seqread_gib_s_linear") or 0.0))
        rr_raid_cons.append(float(raid.get("randread4k_iops_conservative") or 0.0) / 1e6)
        rr_raid_lin.append(float(raid.get("randread4k_iops_linear") or 0.0) / 1e6)

        xticks.append(_node_label(node))

    if not xticks:
        raise SystemExit("No plottable nodes (missing seqread/randread4k pairs)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_seq, ax_rr = axes

    x = list(range(len(xticks)))
    w = 0.35

    # Single-device bars.
    ax_seq.bar([i - w / 2 for i in x], seq_single, width=w, label="single NVMe (sample)", color="#4C78A8")
    ax_rr.bar([i - w / 2 for i in x], rr_iops_single, width=w, label="single NVMe (sample)", color="#4C78A8")

    # RAID0 estimate bars: conservative value, with an errorbar up to linear scaling.
    seq_err_up = [max(0.0, lin - cons) for lin, cons in zip(seq_raid_lin, seq_raid_cons)]
    rr_err_up = [max(0.0, lin - cons) for lin, cons in zip(rr_raid_lin, rr_raid_cons)]

    ax_seq.bar(
        [i + w / 2 for i in x],
        seq_raid_cons,
        width=w,
        label="RAID0 8x est (0.7x..1.0x linear)",
        color="#F58518",
        yerr=[[0.0 for _ in seq_err_up], seq_err_up],
        capsize=4,
    )
    ax_rr.bar(
        [i + w / 2 for i in x],
        rr_raid_cons,
        width=w,
        label="RAID0 8x est (0.7x..1.0x linear)",
        color="#F58518",
        yerr=[[0.0 for _ in rr_err_up], rr_err_up],
        capsize=4,
    )

    ax_seq.set_xticks(x)
    ax_seq.set_xticklabels(xticks)
    ax_seq.set_ylabel("GiB/s (1 MiB reads)")
    ax_seq.set_title("Sequential Read (Read-only probe)")
    ax_seq.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax_rr.set_xticks(x)
    ax_rr.set_xticklabels(xticks)
    ax_rr.set_ylabel("MIOPS (4 KiB)")
    ax_rr.set_title("Random Read (Read-only probe)")
    ax_rr.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(args.title)
    handles, labels = ax_seq.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

