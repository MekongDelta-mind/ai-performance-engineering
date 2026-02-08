#!/usr/bin/env python3
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.distributed as dist

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks, _resolve_physical_device_index  # type: ignore
from core.utils.logger import setup_logging, get_logger


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sizes_from_env() -> List[int]:
    raw = os.environ.get("SIZES_BYTES", "")
    if not raw:
        return [1 << 20, 8 << 20, 64 << 20]
    sizes = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        sizes.append(int(token))
    return sizes


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
    setup_logging(level="INFO")
    logger = get_logger("torchrun_allreduce")

    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    require_lock = True
    ramp_requested = _env_bool("RAMP_GPU_CLOCKS", default=True)

    torch.cuda.set_device(local_rank)

    warmup = _env_int("WARMUP_ITERS", 5)
    iters = _env_int("MEASURE_ITERS", 20)
    sizes = _sizes_from_env()

    dist.init_process_group(backend="nccl")
    try:
        node = socket.gethostname()

        with lock_gpu_clocks(device=local_rank) as locked:
            theoretical_tflops, theoretical_mem_gbps = locked
            lock_meta = {
                "locked": bool(theoretical_tflops) or bool(theoretical_mem_gbps),
                "theoretical_tflops_fp16": theoretical_tflops,
                "theoretical_mem_gbps": theoretical_mem_gbps,
            }
            physical_index = int(_resolve_physical_device_index(local_rank))
            clocks = _app_clock_snapshot(physical_index)
            clock_payload = {
                "global_rank": rank,
                "local_rank": local_rank,
                "node": node,
                "physical_gpu": physical_index,
                "lock": lock_meta,
                "clocks": clocks,
            }
            print(f"APP_CLOCKS {json.dumps(clock_payload, sort_keys=True)}", flush=True)

            # Avoid hangs: decide lock validity collectively.
            ok = 1 if lock_meta["locked"] else 0
            ok_t = torch.tensor([ok], device="cuda", dtype=torch.int32)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            lock_ok_all = int(ok_t.item())
            if require_lock and lock_ok_all != 1:
                if rank == 0:
                    logger.error("Clock lock required but at least one rank failed to lock.")
                return 3

            gathered_clocks: List[Dict[str, Any]] = [None] * world_size
            dist.all_gather_object(gathered_clocks, clock_payload)

            if ramp_requested:
                # Outside the timed section; avoids "cold GPU" clock regimes.
                ramp_gpu_clocks(device=local_rank)

            results: List[Dict[str, Any]] = []
            for size_bytes in sizes:
                elem_count = size_bytes // 4  # float32
                tensor = torch.ones(elem_count, device="cuda", dtype=torch.float32)
                for _ in range(warmup):
                    dist.all_reduce(tensor)
                torch.cuda.synchronize()
                dist.barrier()
                start = time.perf_counter()
                for _ in range(iters):
                    dist.all_reduce(tensor)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                avg = elapsed / iters
                algbw = (size_bytes / avg) / 1e9
                busbw = algbw * (2 * (world_size - 1) / world_size)
                if rank == 0:
                    results.append(
                        {
                            "size_bytes": size_bytes,
                            "dtype": "float32",
                            "time_s": avg,
                            "algbw_gbps": algbw,
                            "busbw_gbps": busbw,
                        }
                    )

            if rank == 0:
                out_path = os.environ.get("OUTPUT_JSON", "")
                payload = {
                    "run_id": os.environ.get("RUN_ID", ""),
                    "world_size": world_size,
                    "warmup_iters": warmup,
                    "measure_iters": iters,
                    "sizes_bytes": sizes,
                    "results": results,
                    "app_clocks": gathered_clocks,
                }
                if out_path:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, sort_keys=True)
                    logger.info("Wrote %s", out_path)
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
