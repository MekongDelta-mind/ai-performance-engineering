#!/usr/bin/env python3
"""
Summarize variance across multiple cluster health suite summary JSONs.

Stdlib-only so it can be run with system python.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: failed to read {path}: {exc}")


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
        vals = []
        for hca, payload in ib.items():
            if not isinstance(payload, dict):
                continue
            gbps = payload.get("avg_gbps")
            if isinstance(gbps, (int, float)):
                out[f"{ib_key}.{hca}.avg_gbps"] = float(gbps)
                vals.append(float(gbps))
        if vals:
            out[f"{ib_key}.mean_avg_gbps"] = mean(vals)

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


def _fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "n/a"
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return f"{x:.{digits}f}"


@dataclass(frozen=True)
class Run:
    path: Path
    run_id: str
    variant: str  # base|extended|unknown
    metrics: Dict[str, float]


def _variant(summary: Dict[str, Any]) -> str:
    if isinstance(summary.get("ib_read_bw"), dict) or isinstance(summary.get("ib_send_bw"), dict):
        return "extended"
    if isinstance(summary.get("ib_write_bw"), dict):
        return "base"
    return "unknown"


def _load_runs(paths: Iterable[Path]) -> List[Run]:
    runs: List[Run] = []
    for p in paths:
        s = _read_json(p)
        run_id = str(s.get("run_id") or p.stem)
        runs.append(Run(path=p, run_id=run_id, variant=_variant(s), metrics=_flatten_metrics(s)))
    runs.sort(key=lambda r: (r.variant, r.run_id))
    return runs


def _stats(vals: List[float]) -> Tuple[float, float, float, float, float]:
    # mean, stdev(population), min, max, cv%
    m = mean(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    mn = min(vals)
    mx = max(vals)
    cv = (sd / m * 100.0) if m != 0 else float("nan")
    return m, sd, mn, mx, cv


def _render_md(runs: List[Run]) -> str:
    md: List[str] = []
    md.append("# Cluster Health Suite Variance Summary\n")
    md.append("## Runs\n")
    for r in runs:
        md.append(f"- `{r.run_id}` ({r.variant}): `{r.path}`")
    md.append("")

    by_variant: Dict[str, List[Run]] = {}
    for r in runs:
        by_variant.setdefault(r.variant, []).append(r)

    for variant, vruns in by_variant.items():
        md.append(f"## Stats ({variant})\n")
        # Gather per-metric lists.
        per_metric: Dict[str, List[float]] = {}
        for r in vruns:
            for k, v in r.metrics.items():
                per_metric.setdefault(k, []).append(v)

        md.append("| Metric | N | Mean | Stddev | Min | Max | CV% |")
        md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for metric in sorted(per_metric):
            vals = per_metric[metric]
            m, sd, mn, mx, cv = _stats(vals)
            md.append(
                f"| `{metric}` | {len(vals)} | {_fmt(m)} | {_fmt(sd)} | {_fmt(mn)} | {_fmt(mx)} | {_fmt(cv, digits=2)} |"
            )
        md.append("")

    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=[], help="Summary JSON paths")
    ap.add_argument("--glob", help="Glob pattern for summary JSONs")
    ap.add_argument("--output-md", help="Write markdown report to this path")
    ap.add_argument("--output-json", help="Write JSON report to this path")
    args = ap.parse_args()

    paths: List[Path] = []
    for raw in args.inputs:
        paths.append(Path(raw).expanduser().resolve())
    if args.glob:
        paths.extend(sorted(Path().glob(args.glob)))
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("ERROR: no input files found")

    runs = _load_runs(paths)
    md = _render_md(runs)
    print(md)

    if args.output_md:
        out = Path(args.output_md).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runs": [
                {"run_id": r.run_id, "variant": r.variant, "path": str(r.path), "metrics": r.metrics}
                for r in runs
            ]
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

