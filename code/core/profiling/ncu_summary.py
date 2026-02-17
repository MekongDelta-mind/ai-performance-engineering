from __future__ import annotations

import csv
import io
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_float(text: str) -> Optional[float]:
    raw = (text or "").strip().replace(",", "")
    if not raw:
        return None
    if raw.lower() in {"nan", "inf", "+inf", "-inf"}:
        return None
    if raw.endswith("%"):
        raw = raw[:-1].strip()
    try:
        return float(raw)
    except Exception:
        return None


def _time_to_ms(value: float, unit: str) -> float:
    u = (unit or "").strip().lower()
    if u.endswith("ns"):
        return value / 1e6
    if u.endswith("us"):
        return value / 1e3
    if u.endswith("ms"):
        return value
    if u.endswith("s"):
        return value * 1e3
    # Nsight Compute CSV usually uses base units; treat unknown as-is.
    return value


def _parse_raw_csv(csv_text: str) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]]]:
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return [], {}, []

    header = rows[0]
    units_row: List[str] = rows[1] if len(rows) > 1 else []
    units: Dict[str, str] = {}
    for idx, key in enumerate(header):
        units[key] = units_row[idx] if idx < len(units_row) else ""

    records: List[Dict[str, str]] = []
    for row in rows[2:]:
        if not row:
            continue
        record: Dict[str, str] = {}
        for idx, key in enumerate(header):
            record[key] = row[idx] if idx < len(row) else ""
        records.append(record)
    return header, units, records


def _default_metrics() -> List[str]:
    # Chosen for "what kernel should I tune next?" triage: time + utilization + occupancy + resource limits.
    return [
        "gpu__time_duration.avg",
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block",
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    ]


def _ncu_import_raw_csv(report_path: Path, metrics: Iterable[str], timeout_seconds: int) -> Tuple[int, str, str, List[str]]:
    cmd = [
        "ncu",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metrics),
        "--import",
        str(report_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    return proc.returncode, proc.stdout, proc.stderr, cmd


def summarize_ncu_report(
    report_path: Path,
    *,
    top_k: int = 10,
    metrics: Optional[List[str]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Summarize an NCU report or exported CSV into a top-k kernel table.

    Supports:
    - `.ncu-rep` (imports via `ncu --import ... --page raw`)
    - `.csv` exported from `ncu --csv --page raw ...` (parsed directly)
    - Companion `<report>.csv` next to `.ncu-rep` (parsed directly when present)
    """
    path = Path(report_path)
    if not path.exists():
        return {"success": False, "error": f"NCU report not found: {path}", "report_path": str(path)}

    if top_k is None:
        top_k = 10
    try:
        top_k_int = int(top_k)
    except Exception:
        top_k_int = 10
    top_k_int = max(1, min(200, top_k_int))

    metrics_list = metrics or _default_metrics()
    metrics_list = [m.strip() for m in metrics_list if isinstance(m, str) and m.strip()]
    if not metrics_list:
        metrics_list = _default_metrics()

    csv_text = ""
    cmd: Optional[List[str]] = None
    stderr = ""
    returncode = 0

    # Prefer parsing exported CSV if user provides one (or if a companion exists).
    if path.suffix.lower() == ".csv":
        csv_text = path.read_text(encoding="utf-8", errors="replace")
    else:
        companion_csv = path.with_suffix(".csv")
        if companion_csv.exists():
            csv_text = companion_csv.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                returncode, out, stderr, cmd = _ncu_import_raw_csv(path, metrics_list, int(timeout_seconds))
                csv_text = out
            except FileNotFoundError:
                return {"success": False, "error": "ncu not found on PATH", "report_path": str(path)}
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"ncu import timed out after {timeout_seconds}s",
                    "report_path": str(path),
                    "command": cmd,
                }
            except Exception as exc:
                return {"success": False, "error": str(exc), "report_path": str(path), "command": cmd}

    header, units, records = _parse_raw_csv(csv_text)
    if not header:
        return {
            "success": False,
            "error": "Empty or unparseable NCU CSV",
            "report_path": str(path),
            "command": cmd,
            "stderr": stderr,
            "returncode": returncode,
        }

    kernels: List[Dict[str, Any]] = []
    total_time_sum_ms = 0.0

    for record in records:
        raw_id = (record.get("ID") or "").strip()
        if not raw_id.isdigit():
            continue
        kernel_id = int(raw_id)
        kernel_name = (record.get("Kernel Name") or record.get("launch__kernel_name") or "").strip()
        if not kernel_name:
            kernel_name = "<unknown>"

        block_size = (record.get("Block Size") or "").strip()
        grid_size = (record.get("Grid Size") or "").strip()
        stream = (record.get("Stream") or "").strip()
        device = (record.get("Device") or "").strip()
        cc = (record.get("CC") or "").strip()

        metrics_out: Dict[str, Any] = {}
        time_avg_ms: Optional[float] = None
        time_sum_ms: Optional[float] = None

        for key in header:
            if key in {
                "ID",
                "Kernel Name",
                "Block Size",
                "Grid Size",
                "Stream",
                "Device",
                "CC",
                "Context",
                "Process ID",
                "Process Name",
                "Host Name",
                "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
                "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
            }:
                continue
            raw_val = (record.get(key) or "").strip()
            val = _parse_float(raw_val)
            if val is None:
                continue
            unit = units.get(key, "")
            if key.startswith("gpu__time_duration"):
                val = _time_to_ms(val, unit)
            metrics_out[key] = val
            if key == "gpu__time_duration.avg":
                time_avg_ms = val
            if key == "gpu__time_duration.sum":
                time_sum_ms = val

        # Derive an occupancy limiting factor hint when the raw launch limits are present.
        limit_fields = {
            "blocks": metrics_out.get("launch__occupancy_limit_blocks"),
            "registers": metrics_out.get("launch__occupancy_limit_registers"),
            "shared_mem": metrics_out.get("launch__occupancy_limit_shared_mem"),
            "warps": metrics_out.get("launch__occupancy_limit_warps"),
        }
        numeric_limits = {k: v for k, v in limit_fields.items() if isinstance(v, (int, float))}
        occupancy_limit_reason: Optional[str] = None
        if numeric_limits:
            min_val = min(numeric_limits.values())
            reasons = [k for k, v in numeric_limits.items() if v == min_val]
            if reasons:
                occupancy_limit_reason = ",".join(sorted(reasons))

        # Accumulate total time for percent attribution.
        if time_sum_ms is not None:
            total_time_sum_ms += time_sum_ms

        kernels.append(
            {
                "id": kernel_id,
                "kernel_name": kernel_name,
                "block_size": block_size,
                "grid_size": grid_size,
                "stream": stream,
                "device": device,
                "cc": cc,
                "time_avg_ms": time_avg_ms,
                "time_sum_ms": time_sum_ms,
                "occupancy_limit_reason": occupancy_limit_reason,
                "metrics": metrics_out,
            }
        )

    if not kernels:
        return {
            "success": False,
            "error": "No kernel rows found in NCU CSV (expected numeric ID rows)",
            "report_path": str(path),
            "command": cmd,
            "stderr": stderr,
            "returncode": returncode,
        }

    # Prefer sum time when present (closest to "top kernels by total time").
    has_sum = any(k.get("time_sum_ms") is not None for k in kernels)
    sort_key = "time_sum_ms" if has_sum else "time_avg_ms"

    def _safe_key(k: Dict[str, Any]) -> float:
        v = k.get(sort_key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    kernels_sorted = sorted(kernels, key=_safe_key, reverse=True)
    kernels_top = kernels_sorted[:top_k_int]

    if total_time_sum_ms > 0:
        for k in kernels_top:
            t = k.get("time_sum_ms")
            if isinstance(t, (int, float)):
                k["time_pct"] = 100.0 * float(t) / total_time_sum_ms

    return {
        "success": True,
        "report_path": str(path),
        "top_k": top_k_int,
        "sort_by": sort_key,
        "kernel_count": len(kernels_sorted),
        "total_time_sum_ms": total_time_sum_ms if total_time_sum_ms > 0 else None,
        "kernels": kernels_top,
        "metrics_requested": metrics_list,
        "command": cmd,
        "stderr": stderr if stderr else None,
        "returncode": returncode,
    }
