#!/usr/bin/env python3
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from core.harness.benchmark_harness import (  # type: ignore
    lock_gpu_clocks,
    ramp_gpu_clocks,
    _resolve_physical_device_index,
)
from core.utils.logger import setup_logging, get_logger


def _get_env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _app_clock_snapshot(device_index: int) -> Dict[str, Any]:
    try:
        import pynvml
    except ImportError as exc:
        return {"error": f"pynvml import failed: {exc}"}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
        cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        return {
            "applications_sm_mhz": app_sm,
            "applications_mem_mhz": app_mem,
            "current_sm_mhz": cur_sm,
            "current_mem_mhz": cur_mem,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def main() -> int:
    if "--" not in sys.argv:
        print("Usage: nccl_lock_wrapper.py -- <command...>", file=sys.stderr)
        return 2
    idx = sys.argv.index("--")
    cmd = sys.argv[idx + 1 :]
    if not cmd:
        print("Missing command after --", file=sys.stderr)
        return 2

    setup_logging(level="INFO")
    logger = get_logger("nccl_lock_wrapper")

    local_rank = _get_env_int("OMPI_COMM_WORLD_LOCAL_RANK", 0)
    world_rank = _get_env_int("OMPI_COMM_WORLD_RANK", 0)
    world_size = _get_env_int("OMPI_COMM_WORLD_SIZE", 1)
    node = socket.gethostname()
    require_lock = True
    ramp_requested = _env_bool("RAMP_GPU_CLOCKS", default=True)

    # Leave CUDA_VISIBLE_DEVICES unchanged so nccl-tests can map local_rank -> GPU index.
    physical_index = _resolve_physical_device_index(local_rank)

    logger.info(
        "Rank %d/%d on %s using local_rank=%d -> physical GPU %d",
        world_rank,
        world_size,
        node,
        local_rank,
        physical_index,
    )

    with lock_gpu_clocks(device=local_rank) as locked:
        theoretical_tflops, theoretical_mem_gbps = locked
        lock_meta = {
            "locked": bool(theoretical_tflops) or bool(theoretical_mem_gbps),
            "theoretical_tflops_fp16": theoretical_tflops,
            "theoretical_mem_gbps": theoretical_mem_gbps,
        }
        clocks = _app_clock_snapshot(physical_index)
        payload = {
            "global_rank": world_rank,
            "local_rank": local_rank,
            "node": node,
            "physical_gpu": physical_index,
            "lock": lock_meta,
            "clocks": clocks,
        }
        print(f"APP_CLOCKS {json.dumps(payload, sort_keys=True)}", flush=True)

        if require_lock and not lock_meta["locked"]:
            logger.error("Clock lock required but lock_gpu_clocks did not lock clocks; aborting.")
            return 3

        # Ensure GPUs are in a steady-state P-state before launching nccl-tests.
        # This is outside nccl-tests timing, but helps avoid "cold" low-clock regimes.
        if ramp_requested:
            try:
                ramp_gpu_clocks(device=local_rank)
            except Exception as exc:
                logger.warning("ramp_gpu_clocks failed (continuing): %s", exc)

        proc = subprocess.Popen(cmd)
        return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
