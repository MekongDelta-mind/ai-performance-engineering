#!/usr/bin/env python3
"""
All-reduce stability profiler.

Profiles a SINGLE payload size over many iterations to detect network
jitter, transient congestion, and routing instability.

Unlike the standard NCCL sweep (which averages across message sizes),
this test reveals PER-ITERATION bandwidth variance at a single large
payload. A healthy network should show tight clustering (CV < 2%);
high variance or bimodal distributions indicate congestion, bad routing,
or hardware issues.

Inspired by the --profile_stability mode in:
  https://github.com/stas00/ml-engineering (all_reduce_bench.py)

Adapted for our harness with:
- Structured JSON output (per-iteration + summary statistics)
- CV (coefficient of variation) and p99/p50 ratio as jitter metrics
- CUDA event timing
- Integration with our clock-locking wrapper

Usage (2 nodes, 8 GPUs total):
  torchrun --nproc_per_node=4 --nnodes=2 \\
    --rdzv_endpoint=<master>:29500 --rdzv_backend=c10d \\
    scripts/allreduce_stability_bench.py \\
    --payload-gib 2.0 --iters 200 --warmup 20 \\
    --output results/structured/allreduce_stability.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
from pathlib import Path
from statistics import mean, median, stdev

import torch
import torch.distributed as dist

if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[2]
    _env = os.environ.copy()
    _pythonpath = _env.get("PYTHONPATH")
    _env["PYTHONPATH"] = str(_repo_root) if not _pythonpath else os.pathsep.join([str(_repo_root), _pythonpath])
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "cluster.scripts.allreduce_stability_bench", *sys.argv[1:]],
        _env,
    )

try:
    from core.harness.benchmark_harness import _resolve_physical_device_index  # type: ignore
except Exception:
    _resolve_physical_device_index = None  # type: ignore[assignment]


def _busbw_factor(world_size: int) -> float:
    """Bus bandwidth correction factor for all-reduce."""
    return 2.0 * (world_size - 1) / world_size


def _conv_to_gbps(bps: float) -> float:
    """Convert bytes/sec to GBps (base-10 convention)."""
    return bps / 1e9


def _clock_snapshot(logical_device: int, hostname: str, rank: int) -> dict:
    physical_device = logical_device
    if _resolve_physical_device_index is not None:
        try:
            physical_device = int(_resolve_physical_device_index(logical_device))
        except Exception:
            physical_device = logical_device

    try:
        import pynvml
    except Exception as exc:
        return {
            "rank": rank,
            "hostname": hostname,
            "logical_device": logical_device,
            "physical_device": physical_device,
            "error": f"pynvml unavailable: {exc}",
        }

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        return {
            "rank": rank,
            "hostname": hostname,
            "logical_device": logical_device,
            "physical_device": physical_device,
            "gpu_name": gpu_name,
            "app_sm_mhz": int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)),
            "app_mem_mhz": int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)),
            "cur_sm_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            "cur_mem_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="All-reduce stability profiler")
    parser.add_argument("--payload-gib", type=float, default=2.0, help="Payload size in GiB (default: 2.0)")
    parser.add_argument("--iters", type=int, default=200, help="Measurement iterations (default: 200)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations (default: 20)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    if os.environ.get("AISP_CLOCK_LOCKED") != "1":
        raise SystemExit(
            "ERROR: GPU clock lock is required for all-reduce stability profiling.\n"
            "Run via scripts/run_with_gpu_clocks.sh (or wrapper scripts that call it)."
        )

    # Init distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0 = rank == 0
    hostname = socket.gethostname()
    local_clock = _clock_snapshot(local_rank, hostname, rank)
    all_clocks = [None for _ in range(world_size)]
    dist.all_gather_object(all_clocks, local_clock)

    payload_bytes = int(args.payload_gib * (2**30))
    num_elements = payload_bytes // 4  # FP32 = 4 bytes
    busbw_coeff = _busbw_factor(world_size)

    if is_rank0:
        print(f"All-reduce stability profiler")
        print(f"  Ranks: {world_size}")
        print(f"  Payload: {args.payload_gib} GiB ({payload_bytes:,} bytes)")
        print(f"  Warmup: {args.warmup}, Iterations: {args.iters}")
        print(f"  BusBW correction: {busbw_coeff:.4f}")
        print()

    tensor = torch.rand(num_elements, dtype=torch.float32, device=f"cuda:{local_rank}")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for i in range(args.warmup):
        dist.barrier()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    if is_rank0:
        print("Warmup complete. Starting measurement...")

    # Measurement
    algbw_list = []
    busbw_list = []
    duration_ms_list = []

    for i in range(args.iters):
        dist.barrier()
        start_event.record()
        dist.all_reduce(tensor)
        end_event.record()
        torch.cuda.synchronize()

        duration_ms = start_event.elapsed_time(end_event)
        duration_s = duration_ms / 1000.0

        algbw = payload_bytes / duration_s
        busbw = algbw * busbw_coeff

        # Reduce algbw across all ranks (mean)
        algbw_tensor = torch.tensor([algbw], device=f"cuda:{local_rank}")
        dist.reduce(algbw_tensor, dst=0, op=dist.ReduceOp.SUM)

        if is_rank0:
            algbw_mean = algbw_tensor.item() / world_size
            busbw_mean = algbw_mean * busbw_coeff
            algbw_list.append(algbw_mean)
            busbw_list.append(busbw_mean)
            duration_ms_list.append(duration_ms)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{args.iters}] busbw={_conv_to_gbps(busbw_mean):.2f} GBps")

    del tensor
    gc.collect()

    if is_rank0:
        # Compute statistics
        busbw_gbps = [_conv_to_gbps(x) for x in busbw_list]
        algbw_gbps = [_conv_to_gbps(x) for x in algbw_list]

        busbw_mean_gbps = mean(busbw_gbps)
        busbw_median_gbps = median(busbw_gbps)
        busbw_std_gbps = stdev(busbw_gbps) if len(busbw_gbps) > 1 else 0.0
        busbw_cv = (busbw_std_gbps / busbw_mean_gbps * 100) if busbw_mean_gbps > 0 else 0.0
        busbw_min_gbps = min(busbw_gbps)
        busbw_max_gbps = max(busbw_gbps)

        sorted_busbw = sorted(busbw_gbps)
        p50 = sorted_busbw[len(sorted_busbw) // 2]
        p99_idx = max(0, int(0.99 * len(sorted_busbw)) - 1)
        p99 = sorted_busbw[p99_idx]
        p01_idx = min(len(sorted_busbw) - 1, int(0.01 * len(sorted_busbw)))
        p01 = sorted_busbw[p01_idx]

        # Jitter assessment
        p99_p50_ratio = p99 / p50 if p50 > 0 else 0
        max_min_ratio = busbw_max_gbps / busbw_min_gbps if busbw_min_gbps > 0 else 0

        if busbw_cv < 1.0:
            jitter_assessment = "excellent"
        elif busbw_cv < 2.0:
            jitter_assessment = "good"
        elif busbw_cv < 5.0:
            jitter_assessment = "moderate_jitter"
        else:
            jitter_assessment = "high_jitter"

        result = {
            "test": "allreduce_stability",
            "payload_gib": args.payload_gib,
            "payload_bytes": payload_bytes,
            "world_size": world_size,
            "warmup_iters": args.warmup,
            "measurement_iters": args.iters,
            "hostname": hostname,
            "app_clocks": all_clocks,
            "summary": {
                "busbw_mean_gbps": round(busbw_mean_gbps, 3),
                "busbw_median_gbps": round(busbw_median_gbps, 3),
                "busbw_std_gbps": round(busbw_std_gbps, 3),
                "busbw_cv_pct": round(busbw_cv, 3),
                "busbw_min_gbps": round(busbw_min_gbps, 3),
                "busbw_max_gbps": round(busbw_max_gbps, 3),
                "busbw_p01_gbps": round(p01, 3),
                "busbw_p50_gbps": round(p50, 3),
                "busbw_p99_gbps": round(p99, 3),
                "p99_p50_ratio": round(p99_p50_ratio, 4),
                "max_min_ratio": round(max_min_ratio, 4),
                "jitter_assessment": jitter_assessment,
                "algbw_mean_gbps": round(mean(algbw_gbps), 3),
            },
            "per_iteration": {
                "busbw_gbps": [round(x, 3) for x in busbw_gbps],
                "algbw_gbps": [round(x, 3) for x in algbw_gbps],
                "duration_ms": [round(x, 4) for x in duration_ms_list],
            },
            "software": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": ".".join(str(x) for x in torch.cuda.nccl.version()),
                "clock_lock_required": True,
            },
        }

        # Write output
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print()
        print("=" * 60)
        print(f"ALL-REDUCE STABILITY RESULTS ({args.payload_gib} GiB, {world_size} ranks)")
        print(f"  Bus BW: mean={busbw_mean_gbps:.2f} GBps, median={busbw_median_gbps:.2f} GBps")
        print(f"  Std:    {busbw_std_gbps:.3f} GBps (CV={busbw_cv:.2f}%)")
        print(f"  Range:  [{busbw_min_gbps:.2f}, {busbw_max_gbps:.2f}] GBps")
        print(f"  P01/P50/P99: {p01:.2f} / {p50:.2f} / {p99:.2f} GBps")
        print(f"  Jitter: {jitter_assessment} (p99/p50={p99_p50_ratio:.4f}, max/min={max_min_ratio:.4f})")
        print(f"  Output: {out_path}")
        print("=" * 60)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
