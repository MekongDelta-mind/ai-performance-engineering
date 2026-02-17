#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from statistics import mean

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from core.harness.benchmark_harness import _resolve_physical_device_index  # type: ignore
except Exception:  # pragma: no cover - best-effort for standalone use
    _resolve_physical_device_index = None  # type: ignore[assignment]


def _clock_snapshot(device_index: int) -> dict:
    """Best-effort NVML snapshot for the current app clocks."""
    try:
        import pynvml
    except ImportError:
        return {"error": "pynvml not installed"}

    physical_index = device_index
    if _resolve_physical_device_index is not None:
        try:
            physical_index = int(_resolve_physical_device_index(device_index))
        except Exception:
            physical_index = device_index

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
        cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        return {
            "physical_gpu": physical_index,
            "app_sm_mhz": app_sm,
            "app_mem_mhz": app_mem,
            "cur_sm_mhz": cur_sm,
            "cur_mem_mhz": cur_mem,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def benchmark_once(m, n, k, dtype, device, iters):
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    # Warmup
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return times


def tflops(m, n, k, secs):
    # 2 * M * N * K FLOPs
    return (2.0 * m * n * k) / secs / 1e12


def main():
    parser = argparse.ArgumentParser(description="Torch GEMM microbench")
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--label", default="node")
    args = parser.parse_args()

    if os.environ.get("AISP_CLOCK_LOCKED") != "1":
        raise SystemExit(
            "ERROR: GPU clock lock is required for this benchmark.\n"
            "\n"
            "Run via:\n"
            "  scripts/run_with_gpu_clocks.sh -- env/venv/bin/python scripts/torch_gemm_bench.py ...\n"
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    clocks = _clock_snapshot(torch.cuda.current_device())
    times = benchmark_once(args.m, args.n, args.k, dtype, device, args.iters)
    avg = mean(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[max(0, int(0.99 * len(times)) - 1)]

    avg_tflops = tflops(args.m, args.n, args.k, avg)
    p50_tflops = tflops(args.m, args.n, args.k, p50)
    p99_tflops = tflops(args.m, args.n, args.k, p99)

    header = [
        "label",
        "m",
        "n",
        "k",
        "dtype",
        "iters",
        "avg_ms",
        "p50_ms",
        "p99_ms",
        "avg_tflops",
        "p50_tflops",
        "p99_tflops",
        "physical_gpu",
        "app_sm_mhz",
        "app_mem_mhz",
        "cur_sm_mhz",
        "cur_mem_mhz",
    ]
    row = [
        args.label,
        args.m,
        args.n,
        args.k,
        args.dtype,
        args.iters,
        avg * 1000.0,
        p50 * 1000.0,
        p99 * 1000.0,
        avg_tflops,
        p50_tflops,
        p99_tflops,
        clocks.get("physical_gpu", ""),
        clocks.get("app_sm_mhz", ""),
        clocks.get("app_mem_mhz", ""),
        clocks.get("cur_sm_mhz", ""),
        clocks.get("cur_mem_mhz", ""),
    ]

    write_header = False
    try:
        expected = ",".join(header)
        with open(args.output_csv, "r", encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
        if not first:
            write_header = True
        elif first != expected:
            raise SystemExit(
                f"ERROR: Output CSV header mismatch for {args.output_csv}.\n"
                f"  Expected: {expected}\n"
                f"  Found:    {first}\n"
                "\n"
                "Fix: choose a new --output-csv path (recommended) or a new --run-id.\n"
                "Note: mixing old/new schema in one CSV produces invalid plots."
            )
    except FileNotFoundError:
        write_header = True

    with open(args.output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(
        f"{args.label} GEMM {args.m}x{args.k}x{args.n} {args.dtype}: "
        f"avg {avg_tflops:.2f} TFLOPS, p50 {p50_tflops:.2f}, p99 {p99_tflops:.2f}"
    )


if __name__ == "__main__":
    main()
