from __future__ import annotations

from typing import Any, Dict, List, Optional


def _query_nvml_clocks(physical_index: int) -> Dict[str, Optional[int]]:
    try:
        import pynvml  # type: ignore
    except Exception as exc:
        return {"error": f"pynvml import failed: {exc}", "app_sm_mhz": None, "app_mem_mhz": None, "cur_sm_mhz": None, "cur_mem_mhz": None}

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(physical_index))
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
        cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        return {"app_sm_mhz": app_sm, "app_mem_mhz": app_mem, "cur_sm_mhz": cur_sm, "cur_mem_mhz": cur_mem}
    except Exception as exc:
        return {"error": f"NVML clock query failed: {exc}", "app_sm_mhz": None, "app_mem_mhz": None, "cur_sm_mhz": None, "cur_mem_mhz": None}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def clock_lock_check(
    *,
    sm_clock_mhz: Optional[int] = None,
    mem_clock_mhz: Optional[int] = None,
    devices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Attempt to lock GPU clocks via the harness and report before/after clocks.

    This is a diagnostic tool to validate the environment supports clock locking.
    It uses `core.harness.benchmark_harness.lock_gpu_clocks` (nvidia-smi + sudo -n fallback).
    """
    try:
        import torch
    except Exception as exc:
        return {"success": False, "error": f"torch import failed: {exc}"}

    from core.harness.benchmark_harness import lock_gpu_clocks, _resolve_physical_device_index

    if not torch.cuda.is_available():
        return {"success": False, "error": "CUDA not available (torch.cuda.is_available() is False)"}

    gpu_count = int(torch.cuda.device_count())
    if devices is None:
        devices = list(range(gpu_count))

    results: List[Dict[str, Any]] = []
    for device in devices:
        record: Dict[str, Any] = {"device": int(device)}
        try:
            physical_index = int(_resolve_physical_device_index(int(device)))
        except Exception as exc:
            physical_index = int(device)
            record["physical_index_error"] = str(exc)

        record["physical_index"] = physical_index
        record["before"] = _query_nvml_clocks(physical_index)

        try:
            with lock_gpu_clocks(device=int(device), sm_clock_mhz=sm_clock_mhz, mem_clock_mhz=mem_clock_mhz) as (
                theoretical_tflops,
                theoretical_gbps,
            ):
                record["locked"] = True
                record["theoretical_tflops_fp16"] = theoretical_tflops
                record["theoretical_gbps"] = theoretical_gbps
                record["during"] = _query_nvml_clocks(physical_index)
        except Exception as exc:
            record["locked"] = False
            record["error"] = str(exc)
            results.append(record)
            continue

        record["after"] = _query_nvml_clocks(physical_index)
        results.append(record)

    overall_success = all(bool(r.get("locked")) for r in results) if results else False
    return {"success": overall_success, "gpu_count": gpu_count, "results": results}
