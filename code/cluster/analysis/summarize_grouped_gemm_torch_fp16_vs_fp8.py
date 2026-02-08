#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_RE_SHAPE = re.compile(r"^Benchmarking G=(\d+), M=(\d+), N=(\d+), K=(\d+)\.\.\.$")
_RE_FP16 = re.compile(r"^\s*torch_fp16:\s*([0-9.]+)\s*TFLOPS,\s*([0-9.]+)\s*ms\b")
_RE_FP8 = re.compile(r"^\s*torch_fp8:\s*([0-9.]+)\s*TFLOPS,\s*([0-9.]+)\s*ms\b")
_RE_DEEPGEMM = re.compile(r"^\s*deepgemm_fp8:\s*(.+)$")


def _p50(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(statistics.median(sorted(vals)))


def _minmax(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    return float(min(vals)), float(max(vals))


def _parse_log(path: Path) -> Dict[str, Any]:
    shapes: List[Dict[str, Any]] = []
    deepgemm_unsupported_reason: Optional[str] = None
    deepgemm_seen_any: bool = False

    cur_shape: Optional[Tuple[int, int, int, int]] = None
    cur: Dict[str, Any] = {}

    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip("\n")

        # Capture a single "DeepGEMM unsupported: ..." reason (if present).
        if line.startswith("DeepGEMM unsupported:"):
            # Keep the whole reason text after the prefix for fidelity.
            deepgemm_unsupported_reason = line.split("DeepGEMM unsupported:", 1)[1].strip()

        m = _RE_SHAPE.match(line)
        if m:
            if cur_shape is not None:
                shapes.append(cur)
            g, m_, n, k = map(int, m.groups())
            cur_shape = (g, m_, n, k)
            cur = {
                "shape": {"groups": g, "m": m_, "n": n, "k": k},
                "torch_fp16": None,
                "torch_fp8": None,
                "deepgemm_fp8": None,
            }
            continue

        m = _RE_FP16.match(line)
        if m and cur_shape is not None:
            tflops, ms = float(m.group(1)), float(m.group(2))
            cur["torch_fp16"] = {"tflops": tflops, "elapsed_ms": ms}
            continue

        m = _RE_FP8.match(line)
        if m and cur_shape is not None:
            tflops, ms = float(m.group(1)), float(m.group(2))
            cur["torch_fp8"] = {"tflops": tflops, "elapsed_ms": ms}
            continue

        m = _RE_DEEPGEMM.match(line)
        if m and cur_shape is not None:
            deepgemm_seen_any = True
            rest = m.group(1).strip()
            if rest.startswith("N/A"):
                cur["deepgemm_fp8"] = {"status": "na", "detail": rest}
            else:
                # If DeepGEMM ever reports TFLOPS, capture it here.
                # Keep parsing permissive to avoid breaking on format drift.
                t = re.search(r"([0-9.]+)\s*TFLOPS", rest)
                ms = re.search(r"([0-9.]+)\s*ms", rest)
                cur["deepgemm_fp8"] = {
                    "status": "ok",
                    "detail": rest,
                    "tflops": float(t.group(1)) if t else None,
                    "elapsed_ms": float(ms.group(1)) if ms else None,
                }
            continue

    if cur_shape is not None:
        shapes.append(cur)

    # Derive per-shape ratios.
    for row in shapes:
        fp16 = (row.get("torch_fp16") or {}).get("tflops")
        fp8 = (row.get("torch_fp8") or {}).get("tflops")
        d = row.get("deepgemm_fp8") or {}
        deep = d.get("tflops") if d.get("status") == "ok" else None

        fp16_over_fp8 = None
        if fp16 is not None and fp8 not in (None, 0.0):
            fp16_over_fp8 = float(fp16) / float(fp8)

        deep_over_fp8 = None
        if deep is not None and fp8 not in (None, 0.0):
            deep_over_fp8 = float(deep) / float(fp8)

        deep_over_fp16 = None
        if deep is not None and fp16 not in (None, 0.0):
            deep_over_fp16 = float(deep) / float(fp16)

        row["ratios"] = {
            "fp16_over_fp8": fp16_over_fp8,
            "deepgemm_over_torch_fp8": deep_over_fp8,
            "deepgemm_over_torch_fp16": deep_over_fp16,
        }

    # Global stats.
    fp16_vals = [float(r["torch_fp16"]["tflops"]) for r in shapes if r.get("torch_fp16")]
    fp8_vals = [float(r["torch_fp8"]["tflops"]) for r in shapes if r.get("torch_fp8")]
    ratio_vals = [float(r["ratios"]["fp16_over_fp8"]) for r in shapes if r.get("ratios", {}).get("fp16_over_fp8")]
    deep_vals = [
        float(r["deepgemm_fp8"]["tflops"])
        for r in shapes
        if (r.get("deepgemm_fp8") or {}).get("status") == "ok" and (r.get("deepgemm_fp8") or {}).get("tflops") is not None
    ]
    deep_over_fp8_vals = [
        float(r["ratios"]["deepgemm_over_torch_fp8"])
        for r in shapes
        if (r.get("ratios") or {}).get("deepgemm_over_torch_fp8") is not None
    ]
    deep_over_fp16_vals = [
        float(r["ratios"]["deepgemm_over_torch_fp16"])
        for r in shapes
        if (r.get("ratios") or {}).get("deepgemm_over_torch_fp16") is not None
    ]

    fp16_min, fp16_max = _minmax(fp16_vals)
    fp8_min, fp8_max = _minmax(fp8_vals)
    ratio_min, ratio_max = _minmax(ratio_vals)
    deep_min, deep_max = _minmax(deep_vals)
    deep_over_fp8_min, deep_over_fp8_max = _minmax(deep_over_fp8_vals)
    deep_over_fp16_min, deep_over_fp16_max = _minmax(deep_over_fp16_vals)

    # DeepGEMM datapoints count (only count status=ok with a parsed TFLOPS).
    deepgemm_ok = 0
    for r in shapes:
        d = r.get("deepgemm_fp8") or {}
        if d.get("status") == "ok" and d.get("tflops") is not None:
            deepgemm_ok += 1

    # Worst/best shapes by ratio for stakeholder-friendly callouts.
    ratios_with_shape = []
    for r in shapes:
        ratio = (r.get("ratios") or {}).get("fp16_over_fp8")
        if ratio is None:
            continue
        s = r["shape"]
        ratios_with_shape.append((float(ratio), s))
    ratios_with_shape.sort(key=lambda x: x[0], reverse=True)
    worst5 = [{"ratio": r, **s} for r, s in ratios_with_shape[:5]]
    best5 = [{"ratio": r, **s} for r, s in sorted(ratios_with_shape, key=lambda x: x[0])[:5]]

    # Worst/best shapes by DeepGEMM speedup over torch_fp8 baseline.
    deep_speedups_with_shape = []
    for r in shapes:
        speedup = (r.get("ratios") or {}).get("deepgemm_over_torch_fp8")
        if speedup is None:
            continue
        s = r["shape"]
        deep_speedups_with_shape.append((float(speedup), s))
    deep_speedups_with_shape.sort(key=lambda x: x[0], reverse=True)
    deep_best5 = [{"speedup": r, **s} for r, s in deep_speedups_with_shape[:5]]
    deep_worst5 = [{"speedup": r, **s} for r, s in sorted(deep_speedups_with_shape, key=lambda x: x[0])[:5]]

    return {
        "input_log": str(path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "counts": {
            "shapes": len(shapes),
            "deepgemm_datapoints_ok": deepgemm_ok,
            "deepgemm_lines_seen": deepgemm_seen_any,
        },
        "deepgemm": {
            "unsupported_reason": deepgemm_unsupported_reason,
        },
        "stats": {
            "torch_fp16_tflops": {
                "min": fp16_min,
                "p50": _p50(fp16_vals),
                "max": fp16_max,
            },
            "torch_fp8_tflops": {
                "min": fp8_min,
                "p50": _p50(fp8_vals),
                "max": fp8_max,
            },
            "deepgemm_fp8_tflops": {
                "min": deep_min,
                "p50": _p50(deep_vals),
                "max": deep_max,
            },
            "fp16_over_fp8_ratio": {
                "min": ratio_min,
                "p50": _p50(ratio_vals),
                "max": ratio_max,
                "count_fp8_faster_than_fp16": sum(1 for v in ratio_vals if v < 1.0),
                "count_fp8_slower_than_fp16": sum(1 for v in ratio_vals if v > 1.0),
                "count_ratio_ge_2": sum(1 for v in ratio_vals if v >= 2.0),
                "count_ratio_ge_5": sum(1 for v in ratio_vals if v >= 5.0),
                "count_ratio_ge_10": sum(1 for v in ratio_vals if v >= 10.0),
            },
            "deepgemm_over_torch_fp8_ratio": {
                "min": deep_over_fp8_min,
                "p50": _p50(deep_over_fp8_vals),
                "max": deep_over_fp8_max,
                "count_speedup_ge_1_2": sum(1 for v in deep_over_fp8_vals if v >= 1.2),
                "count_speedup_ge_1_5": sum(1 for v in deep_over_fp8_vals if v >= 1.5),
                "count_speedup_ge_2": sum(1 for v in deep_over_fp8_vals if v >= 2.0),
            },
            "deepgemm_over_torch_fp16_ratio": {
                "min": deep_over_fp16_min,
                "p50": _p50(deep_over_fp16_vals),
                "max": deep_over_fp16_max,
            },
            "worst5_by_ratio": worst5,
            "best5_by_ratio": best5,
            "deepgemm_best5_by_speedup_over_torch_fp8": deep_best5,
            "deepgemm_worst5_by_speedup_over_torch_fp8": deep_worst5,
        },
        "shapes": shapes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize grouped GEMM benchmark output (torch fp16/fp8 + DeepGEMM FP8xFP4 status)."
    )
    ap.add_argument("--in-log", required=True, help="Input grouped-gemm log (txt)")
    ap.add_argument("--out-json", required=True, help="Output summary JSON path")
    args = ap.parse_args()

    in_path = Path(args.in_log)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _parse_log(in_path)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
