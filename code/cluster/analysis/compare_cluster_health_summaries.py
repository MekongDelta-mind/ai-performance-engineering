#!/usr/bin/env python3
"""
Compare two cluster health suite summary JSONs and flag regressions.

This intentionally uses only the Python stdlib so it can be run with system python.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON: {path}: {exc}")


def _fmt_num(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return f"{x:.{digits}f}"


def _pct(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b - 1.0


def _flatten_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    iperf = summary.get("iperf3") or {}
    for direction in ("fwd", "rev"):
        gbps = (iperf.get(direction) or {}).get("gbps")
        if isinstance(gbps, (int, float)):
            out[f"iperf3.{direction}.gbps"] = float(gbps)

    for ib_key in ("ib_write_bw", "ib_read_bw", "ib_send_bw"):
        ib = summary.get(ib_key) or {}
        if not isinstance(ib, dict):
            continue
        for hca, payload in ib.items():
            if not isinstance(payload, dict):
                continue
            gbps = payload.get("avg_gbps")
            if isinstance(gbps, (int, float)):
                out[f"{ib_key}.{hca}.avg_gbps"] = float(gbps)

    nccl = summary.get("nccl") or {}
    if isinstance(nccl, dict):
        for name, payload in nccl.items():
            if not isinstance(payload, dict):
                continue
            max_busbw = payload.get("max_busbw") or {}
            if isinstance(max_busbw, dict):
                busbw = max_busbw.get("busbw_gbps")
                if isinstance(busbw, (int, float)):
                    out[f"nccl.{name}.max_busbw_gbps"] = float(busbw)

    torchdist = summary.get("torchdist") or {}
    if isinstance(torchdist, dict):
        max_busbw = torchdist.get("max_busbw") or {}
        if isinstance(max_busbw, dict):
            busbw = max_busbw.get("busbw_gbps")
            if isinstance(busbw, (int, float)):
                out["torchdist.max_busbw_gbps"] = float(busbw)

    return out


def _meta_warnings(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    warns: List[str] = []

    # Note: older summaries used "sudo_clock_lock" while newer ones use
    # "require_clock_lock". Treat them as the same semantic for meta checks.
    def require_clock_lock(s: Dict[str, Any]) -> Any:
        if "require_clock_lock" in s:
            return s.get("require_clock_lock")
        return s.get("sudo_clock_lock")

    for key in ("hosts", "gpus_per_node", "oob_if", "nccl_ib_hca", "cuda_visible_devices"):
        av = a.get(key)
        bv = b.get(key)
        if av != bv:
            warns.append(f"meta differs: {key}: baseline={av!r} candidate={bv!r}")

    av = require_clock_lock(a)
    bv = require_clock_lock(b)
    if av != bv:
        warns.append(f"meta differs: require_clock_lock: baseline={av!r} candidate={bv!r}")
    return warns


@dataclass(frozen=True)
class Row:
    metric: str
    baseline: Optional[float]
    candidate: Optional[float]
    abs_diff: Optional[float]
    pct_diff: Optional[float]
    status: str


def _compare(
    base: Dict[str, float], cand: Dict[str, float], threshold: float
) -> List[Row]:
    rows: List[Row] = []
    keys = sorted(set(base) | set(cand))
    for k in keys:
        b = base.get(k)
        c = cand.get(k)
        if b is None:
            rows.append(Row(k, None, c, None, None, "NEW"))
            continue
        if c is None:
            rows.append(Row(k, b, None, None, None, "MISSING"))
            continue
        abs_diff = c - b
        pct_diff = _pct(c, b)
        status = "OK"
        if pct_diff is not None:
            if pct_diff <= -threshold:
                status = "REGRESSION"
            elif pct_diff >= threshold:
                status = "IMPROVEMENT"
        rows.append(Row(k, b, c, abs_diff, pct_diff, status))
    return rows


def _to_markdown(rows: Iterable[Row]) -> str:
    lines = []
    lines.append("| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for r in rows:
        abs_s = _fmt_num(r.abs_diff, digits=3) if r.abs_diff is not None else "n/a"
        pct_s = f"{_fmt_num(100.0 * r.pct_diff, digits=2)}%" if r.pct_diff is not None else "n/a"
        lines.append(
            f"| `{r.metric}` | {_fmt_num(r.baseline)} | {_fmt_num(r.candidate)} | {abs_s} | {pct_s} | {r.status} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline summary JSON path")
    ap.add_argument("--candidate", required=True, help="Candidate summary JSON path")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Regression/improvement threshold as a fraction (default: 0.05 = 5%%)",
    )
    ap.add_argument("--output-md", help="Write markdown report to this path")
    ap.add_argument("--output-json", help="Write machine-readable JSON report to this path")
    ap.add_argument("--only", choices=["all", "regressions"], default="all")
    args = ap.parse_args()

    base_path = Path(args.baseline).expanduser().resolve()
    cand_path = Path(args.candidate).expanduser().resolve()

    base_sum = _read_json(base_path)
    cand_sum = _read_json(cand_path)

    warnings = _meta_warnings(base_sum, cand_sum)

    base_metrics = _flatten_metrics(base_sum)
    cand_metrics = _flatten_metrics(cand_sum)
    rows = _compare(base_metrics, cand_metrics, threshold=args.threshold)

    if args.only == "regressions":
        rows = [r for r in rows if r.status in ("REGRESSION", "MISSING")]

    md = []
    md.append("# Cluster Health Suite Summary Comparison\n")
    md.append(f"- Baseline: `{base_path}`")
    md.append(f"- Candidate: `{cand_path}`")
    md.append(f"- Threshold: {args.threshold:.2%}\n")
    if warnings:
        md.append("## Meta Warnings")
        for w in warnings:
            md.append(f"- {w}")
        md.append("")
    md.append("## Metrics\n")
    md.append(_to_markdown(rows))
    md_text = "\n".join(md)

    print(md_text)

    if args.output_md:
        out_md = Path(args.output_md).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_text, encoding="utf-8")

    if args.output_json:
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline": str(base_path),
            "candidate": str(cand_path),
            "threshold": args.threshold,
            "meta_warnings": warnings,
            "rows": [
                {
                    "metric": r.metric,
                    "baseline": r.baseline,
                    "candidate": r.candidate,
                    "abs_diff": r.abs_diff,
                    "pct_diff": r.pct_diff,
                    "status": r.status,
                }
                for r in rows
            ],
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
