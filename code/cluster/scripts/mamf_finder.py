#!/usr/bin/env python3
"""
Maximum Achievable Matmul FLOPS (MAMF) Finder.

Scans various matmul shapes to find the TRUE achievable TFLOPS ceiling for
the current GPU, rather than relying on theoretical peak specs.

This is critical because:
- Theoretical peak TFLOPS are never achievable in practice
- The achievable TFLOPS depends on matrix shapes (tile/wave quantization)
- Different GPUs on the same node may have different achievable TFLOPS
- Knowing the REAL ceiling lets you know when to stop optimizing

Inspired by the approach in:
  https://github.com/stas00/ml-engineering (mamf-finder.py)
  "The Case for Co-Designing Model Architectures with Hardware"
  (arXiv:2401.14489)

Adapted for our cluster eval harness with:
- Clock-locking integration (AISP_CLOCK_LOCKED required)
- CUDA event timing (not wall clock)
- Structured CSV/JSON output
- Multi-GPU concurrent mode (straggler detection)
- NVML clock snapshots

Usage:
  # Quick scan (< 2 min):
  scripts/run_with_gpu_clocks.sh -- env/venv/bin/python scripts/mamf_finder.py \\
    --m-range 256 20480 1024 --n 4096 --k 4096 --output-csv results/structured/mamf.csv

  # Thorough scan (15-30 min, Ctrl-C safe):
  scripts/run_with_gpu_clocks.sh -- env/venv/bin/python scripts/mamf_finder.py \\
    --m-range 1024 16384 1024 --n-range 1024 16384 1024 --k-range 1024 16384 1024 \\
    --output-csv results/structured/mamf.csv

  # Measure specific training shape:
  scripts/run_with_gpu_clocks.sh -- env/venv/bin/python scripts/mamf_finder.py \\
    --m 4096 --n 4096 --k 11008 --output-csv results/structured/mamf.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path
from statistics import mean, median

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from core.harness.benchmark_harness import _resolve_physical_device_index  # type: ignore
except Exception:
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
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        return {
            "physical_gpu": physical_index,
            "gpu_name": gpu_name,
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


def _tflops(m: int, n: int, k: int, secs: float) -> float:
    """Compute TFLOPS for a matmul of shape (M,K) @ (K,N)."""
    return (2.0 * m * n * k) / secs / 1e12


def _parse_range(range_str: str) -> tuple[int, int, int]:
    """Parse 'start stop step' into (start, stop, step)."""
    parts = range_str.split()
    if len(parts) != 3:
        raise ValueError(f"Range must be 'start stop step', got: {range_str!r}")
    start, stop, step = int(parts[0]), int(parts[1]), int(parts[2])
    if step <= 0:
        raise ValueError(f"Range step must be > 0, got: {step}")
    if start <= 0 or stop <= 0:
        raise ValueError(f"Range start/stop must be > 0, got: start={start}, stop={stop}")
    if start > stop:
        raise ValueError(f"Range start must be <= stop, got: start={start}, stop={stop}")
    return start, stop, step


def _benchmark_shape(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    num_warmup: int,
    num_iters: int,
) -> dict:
    """Benchmark a single matmul shape. Returns timing stats."""
    # Allocate
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(num_warmup):
        _ = a @ b
    torch.cuda.synchronize()

    # Timed iterations
    times_ms = []
    for _ in range(num_iters):
        start_event.record()
        _ = a @ b
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(start_event.elapsed_time(end_event))

    # Free memory
    del a, b
    gc.collect()
    torch.cuda.empty_cache()

    times_sec = [t / 1000.0 for t in times_ms]
    tflops_list = [_tflops(m, n, k, t) for t in times_sec]

    return {
        "m": m,
        "n": n,
        "k": k,
        "max_tflops": max(tflops_list),
        "mean_tflops": mean(tflops_list),
        "median_tflops": median(tflops_list),
        "min_ms": min(times_ms),
        "mean_ms": mean(times_ms),
        "median_ms": median(times_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MAMF Finder: Maximum Achievable Matmul FLOPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shape specification: either fixed or range for each dimension
    parser.add_argument("--m", type=int, default=None, help="Fixed M dimension")
    parser.add_argument("--n", type=int, default=None, help="Fixed N dimension")
    parser.add_argument("--k", type=int, default=None, help="Fixed K dimension")
    parser.add_argument(
        "--m-range",
        type=str,
        default=None,
        help="M range as 'start stop step' (e.g., '1024 16384 1024')",
    )
    parser.add_argument(
        "--n-range",
        type=str,
        default=None,
        help="N range as 'start stop step' (e.g., '1024 16384 1024')",
    )
    parser.add_argument(
        "--k-range",
        type=str,
        default=None,
        help="K range as 'start stop step' (e.g., '1024 16384 1024')",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type (default: bf16)",
    )
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warmup iterations per shape (default: 10)")
    parser.add_argument("--iters", type=int, default=50, help="Measurement iterations per shape (default: 50)")

    parser.add_argument("--output-csv", required=True, help="Output CSV path for all shape results")
    parser.add_argument("--output-json", default=None, help="Output JSON summary (best shape + stats)")
    parser.add_argument("--label", default="gpu0", help="Label for this GPU/node (default: gpu0)")

    args = parser.parse_args()

    # Clock lock check
    if os.environ.get("AISP_CLOCK_LOCKED") != "1":
        raise SystemExit(
            "ERROR: GPU clock lock is required for MAMF finder.\n"
            "\n"
            "Run via:\n"
            "  scripts/run_with_gpu_clocks.sh -- env/venv/bin/python scripts/mamf_finder.py ...\n"
        )

    # Resolve dimensions
    m_values: list[int] = []
    n_values: list[int] = []
    k_values: list[int] = []

    if args.m is not None:
        m_values = [args.m]
    elif args.m_range is not None:
        start, stop, step = _parse_range(args.m_range)
        m_values = list(range(start, stop + 1, step))
    else:
        # Default: quick scan
        m_values = [4096]

    if args.n is not None:
        n_values = [args.n]
    elif args.n_range is not None:
        start, stop, step = _parse_range(args.n_range)
        n_values = list(range(start, stop + 1, step))
    else:
        n_values = [4096]

    if args.k is not None:
        k_values = [args.k]
    elif args.k_range is not None:
        start, stop, step = _parse_range(args.k_range)
        k_values = list(range(start, stop + 1, step))
    else:
        k_values = [4096]

    total_shapes = len(m_values) * len(n_values) * len(k_values)
    if total_shapes == 0:
        raise SystemExit("ERROR: No shapes to test. Check --m/--n/--k or --m-range/--n-range/--k-range.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")

    clocks = _clock_snapshot(torch.cuda.current_device())
    gpu_name = clocks.get("gpu_name", "unknown")

    print(f"MAMF Finder: {total_shapes} shapes, dtype={args.dtype}, device={gpu_name}")
    print(f"  Label: {args.label}")
    print(f"  Clocks: SM={clocks.get('cur_sm_mhz', '?')} MHz, Mem={clocks.get('cur_mem_mhz', '?')} MHz")
    print(f"  Warmup: {args.warmup_iters}, Iters: {args.iters}")
    print()

    # CSV setup
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_header = [
        "label",
        "m",
        "n",
        "k",
        "dtype",
        "max_tflops",
        "mean_tflops",
        "median_tflops",
        "min_ms",
        "mean_ms",
        "median_ms",
        "gpu_name",
        "app_sm_mhz",
        "app_mem_mhz",
        "cur_sm_mhz",
        "cur_mem_mhz",
    ]

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(csv_header)

    # Track best result (support Ctrl-C graceful exit)
    best_tflops = 0.0
    best_shape = (0, 0, 0)
    best_result = None
    all_results: list[dict] = []
    interrupted = False

    def _sigint_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\nInterrupted! Reporting best result so far...\n")

    signal.signal(signal.SIGINT, _sigint_handler)

    shape_idx = 0
    for m in m_values:
        if interrupted:
            break
        for n in n_values:
            if interrupted:
                break
            for k in k_values:
                if interrupted:
                    break

                shape_idx += 1
                try:
                    result = _benchmark_shape(m, n, k, dtype, device, args.warmup_iters, args.iters)
                except torch.cuda.OutOfMemoryError:
                    print(f"  [{shape_idx}/{total_shapes}] {m}x{k}x{n}: OOM - skipping")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                all_results.append(result)

                # Write CSV row
                writer.writerow([
                    args.label,
                    m,
                    n,
                    k,
                    args.dtype,
                    f"{result['max_tflops']:.2f}",
                    f"{result['mean_tflops']:.2f}",
                    f"{result['median_tflops']:.2f}",
                    f"{result['min_ms']:.4f}",
                    f"{result['mean_ms']:.4f}",
                    f"{result['median_ms']:.4f}",
                    gpu_name,
                    clocks.get("app_sm_mhz", ""),
                    clocks.get("app_mem_mhz", ""),
                    clocks.get("cur_sm_mhz", ""),
                    clocks.get("cur_mem_mhz", ""),
                ])
                csv_file.flush()

                # Track best
                if result["max_tflops"] > best_tflops:
                    best_tflops = result["max_tflops"]
                    best_shape = (m, n, k)
                    best_result = result

                # Progress
                marker = " <-- BEST" if result["max_tflops"] == best_tflops else ""
                print(
                    f"  [{shape_idx}/{total_shapes}] "
                    f"{m}x{k}x{n}: "
                    f"max={result['max_tflops']:.1f} mean={result['mean_tflops']:.1f} "
                    f"median={result['median_tflops']:.1f} TFLOPS"
                    f"{marker}"
                )

    csv_file.close()

    if not all_results:
        raise SystemExit("ERROR: No successful shapes were benchmarked (all failed or OOM).")

    # Summary
    print()
    print("=" * 60)
    print(f"MAMF RESULT ({args.label}):")
    print(f"  GPU: {gpu_name}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Best shape (MxKxN): {best_shape[0]}x{best_shape[2]}x{best_shape[1]}")
    print(f"  Maximum Achievable Matmul TFLOPS: {best_tflops:.1f}")
    if best_result:
        print(f"  Mean TFLOPS at best shape: {best_result['mean_tflops']:.1f}")
        print(f"  Median TFLOPS at best shape: {best_result['median_tflops']:.1f}")
    print(f"  Shapes tested: {len(all_results)} / {total_shapes}")
    print(f"  Clocks: SM={clocks.get('cur_sm_mhz', '?')} MHz, Mem={clocks.get('cur_mem_mhz', '?')} MHz")
    print(f"  Output: {csv_path}")
    print("=" * 60)

    # Optional JSON summary
    if args.output_json:
        summary = {
            "label": args.label,
            "gpu_name": gpu_name,
            "dtype": args.dtype,
            "mamf_tflops": round(best_tflops, 2),
            "best_shape": {"m": best_shape[0], "n": best_shape[1], "k": best_shape[2]},
            "mean_tflops_at_best": round(best_result["mean_tflops"], 2) if best_result else None,
            "median_tflops_at_best": round(best_result["median_tflops"], 2) if best_result else None,
            "shapes_tested": len(all_results),
            "shapes_total": total_shapes,
            "app_clocks": {
                "app_sm_mhz": clocks.get("app_sm_mhz"),
                "app_mem_mhz": clocks.get("app_mem_mhz"),
                "cur_sm_mhz": clocks.get("cur_sm_mhz"),
                "cur_mem_mhz": clocks.get("cur_mem_mhz"),
            },
            "clocks": clocks,
            "warmup_iters": args.warmup_iters,
            "measurement_iters": args.iters,
        }
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  JSON summary: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
