#!/usr/bin/env python3
"""
Plot simple GPU telemetry captured via `nvidia-smi --query-gpu ... --format=csv -l 1`.

This intentionally stays lightweight and expects the CSV format emitted by nvidia-smi,
including the leading spaces after commas in the header fields.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def _parse_float(val: str) -> Optional[float]:
    v = val.strip()
    if not v:
        return None
    for suffix in ("MHz", "W", "%"):
        if v.endswith(suffix):
            v = v[: -len(suffix)].strip()
    try:
        return float(v)
    except ValueError:
        return None


def _parse_ts(val: str) -> Optional[datetime]:
    v = val.strip()
    if not v:
        return None
    # Example: 2026/02/07 22:25:30.912
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            pass
    return None


@dataclass
class Series:
    label: str
    t_s: List[float]
    sm_mhz: List[float]
    mem_mhz: List[float]
    power_w: List[float]
    temp_c: List[float]
    util_gpu_pct: List[float]
    sw_power_cap_active: List[int]


def _read_telemetry_csv(path: Path, label: str) -> Series:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # nvidia-smi uses headers like " index" with a leading space.
            rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})

    t0: Optional[datetime] = None
    t_s: List[float] = []
    sm_mhz: List[float] = []
    mem_mhz: List[float] = []
    power_w: List[float] = []
    temp_c: List[float] = []
    util_gpu_pct: List[float] = []
    sw_power_cap_active: List[int] = []

    for r in rows:
        ts = _parse_ts(r.get("timestamp", ""))
        if ts is None:
            # Skip malformed rows.
            continue
        if t0 is None:
            t0 = ts
        dt = (ts - t0).total_seconds()

        sm = _parse_float(r.get("clocks.current.sm [MHz]", ""))
        mem = _parse_float(r.get("clocks.current.memory [MHz]", ""))
        pw = _parse_float(r.get("power.draw [W]", ""))
        tc = _parse_float(r.get("temperature.gpu", ""))
        util = _parse_float(r.get("utilization.gpu [%]", ""))
        cap = 1 if r.get("clocks_event_reasons.sw_power_cap", "").strip().lower() == "active" else 0

        # Keep the series aligned by only appending when we have the core numeric fields.
        if sm is None or mem is None or pw is None or tc is None or util is None:
            continue
        t_s.append(dt)
        sm_mhz.append(sm)
        mem_mhz.append(mem)
        power_w.append(pw)
        temp_c.append(tc)
        util_gpu_pct.append(util)
        sw_power_cap_active.append(cap)

    return Series(
        label=label,
        t_s=t_s,
        sm_mhz=sm_mhz,
        mem_mhz=mem_mhz,
        power_w=power_w,
        temp_c=temp_c,
        util_gpu_pct=util_gpu_pct,
        sw_power_cap_active=sw_power_cap_active,
    )


def _auto_label(path: Path) -> str:
    # Try to produce a readable label like node1_gpu0 from the filename.
    name = path.stem
    for token in ("_telemetry", "_nvidia_smi", "_query"):
        name = name.replace(token, "")
    return name


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Telemetry CSV path (repeatable).",
    )
    p.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label for the corresponding --csv (repeatable). If omitted, derived from filename.",
    )
    p.add_argument("--out", required=True, help="Output PNG path.")
    p.add_argument("--title", default="GPU Telemetry", help="Plot title.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    csv_paths = [Path(p) for p in args.csv]
    labels: List[str] = list(args.label or [])

    if labels and len(labels) != len(csv_paths):
        raise SystemExit("--label count must match --csv count (or omit --label entirely).")

    if not labels:
        labels = [_auto_label(p) for p in csv_paths]

    series = [_read_telemetry_csv(p, lbl) for p, lbl in zip(csv_paths, labels)]
    if not series or all(len(s.t_s) == 0 for s in series):
        raise SystemExit("No telemetry samples parsed from inputs.")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax_clk, ax_pwr, ax_tmp, ax_cap = axes

    for s in series:
        ax_clk.plot(s.t_s, s.sm_mhz, label=f"{s.label} SM")
        ax_pwr.plot(s.t_s, s.power_w, label=s.label)
        ax_tmp.plot(s.t_s, s.temp_c, label=s.label)
        ax_cap.plot(s.t_s, s.sw_power_cap_active, label=s.label)

    ax_clk.set_ylabel("SM MHz")
    ax_pwr.set_ylabel("Power (W)")
    ax_tmp.set_ylabel("Temp (C)")
    ax_cap.set_ylabel("SW power cap")
    ax_cap.set_xlabel("Seconds")

    ax_clk.grid(True, alpha=0.3)
    ax_pwr.grid(True, alpha=0.3)
    ax_tmp.grid(True, alpha=0.3)
    ax_cap.grid(True, alpha=0.3)
    ax_cap.set_yticks([0, 1])
    ax_cap.set_yticklabels(["No", "Yes"])

    ax_clk.legend(loc="best", fontsize=9)
    ax_pwr.legend(loc="best", fontsize=9)
    ax_tmp.legend(loc="best", fontsize=9)
    ax_cap.legend(loc="best", fontsize=9)

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

