#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load_fio_read_metrics(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    jobs = data.get("jobs") or []
    if not jobs:
        raise ValueError(f"fio json missing jobs: {path}")

    job = jobs[0]
    read = job.get("read") or {}
    bw_bytes = float(read.get("bw_bytes") or 0.0)
    iops = float(read.get("iops") or 0.0)

    p99_ms: Optional[float] = None
    clat = read.get("clat_ns") or {}
    pct = clat.get("percentile") or {}
    p99_ns = pct.get("99.000000")
    if p99_ns is not None:
        p99_ms = float(p99_ns) / 1e6

    return {
        "file": str(path),
        "bw_bytes_s": bw_bytes,
        "bw_gib_s": bw_bytes / (1024.0**3),
        "iops": iops,
        "p99_clat_ms": p99_ms,
    }


def _parse_name(run_id: str, name: str) -> Optional[Tuple[str, str, str]]:
    # Example:
    #   2026-02-08_042500_scratch_nvme_readonly_node1_nvme0n1_seqread.json
    pat = re.compile(
        re.escape(run_id)
        + r"_(?P<node>node\d+)_(?P<dev>nvme\d+n\d+)_(?P<test>seqread|randread4k)\.json$"
    )
    m = pat.search(name)
    if not m:
        return None
    return m.group("node"), m.group("dev"), m.group("test")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize read-only fio-on-raw-block-device scratch NVMe probes.")
    ap.add_argument("--run-id", required=True, help="RUN_ID prefix used in results/structured filenames")
    ap.add_argument(
        "--structured-dir",
        default="results/structured",
        help="Directory containing fio JSON outputs (default: results/structured)",
    )
    ap.add_argument("--out-json", required=True, help="Write summary JSON to this path")
    ap.add_argument(
        "--raid0-drives",
        type=int,
        default=8,
        help="How many equal-size scratch NVMe drives would be used in RAID0 (default: 8)",
    )
    ap.add_argument(
        "--raid0-efficiency",
        type=float,
        default=0.7,
        help="Conservative fraction of linear scaling for RAID0 estimates (default: 0.7)",
    )
    args = ap.parse_args()

    structured_dir = Path(args.structured_dir)
    run_id = args.run_id

    measurements: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for path in sorted(structured_dir.glob(f"{run_id}_*.json")):
        parsed = _parse_name(run_id, path.name)
        if parsed is None:
            continue
        node, dev, test = parsed
        metrics = _load_fio_read_metrics(path)
        measurements.setdefault(node, {}).setdefault(dev, {})[test] = metrics

    # Pick a "large scratch" representative device per node as the one with the highest seqread bw.
    estimates: Dict[str, Any] = {}
    for node, devs in measurements.items():
        best_dev = None
        best_bw = -1.0
        for dev, tests in devs.items():
            seq = tests.get("seqread")
            if not seq:
                continue
            bw = float(seq.get("bw_gib_s") or 0.0)
            if bw > best_bw:
                best_bw = bw
                best_dev = dev

        if best_dev is None:
            continue

        seq = devs[best_dev].get("seqread")
        rnd = devs[best_dev].get("randread4k")
        if not seq or not rnd:
            continue

        linear_seq_gib = float(seq["bw_gib_s"]) * args.raid0_drives
        linear_iops = float(rnd["iops"]) * args.raid0_drives
        conservative_seq_gib = linear_seq_gib * float(args.raid0_efficiency)
        conservative_iops = linear_iops * float(args.raid0_efficiency)

        estimates[node] = {
            "representative_device": best_dev,
            "raid0": {
                "drives": args.raid0_drives,
                "efficiency": args.raid0_efficiency,
                "seqread_gib_s_linear": linear_seq_gib,
                "seqread_gib_s_conservative": conservative_seq_gib,
                "randread4k_iops_linear": linear_iops,
                "randread4k_iops_conservative": conservative_iops,
            },
        }

    out = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {
            "fio": "direct=1, read-only, raw block device",
            "seqread": {"rw": "read", "bs": "1M", "iodepth": 32, "numjobs": 1, "runtime_s": 30},
            "randread4k": {"rw": "randread", "bs": "4k", "iodepth": 128, "numjobs": 4, "runtime_s": 30},
        },
        "measurements": measurements,
        "estimates": estimates,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

