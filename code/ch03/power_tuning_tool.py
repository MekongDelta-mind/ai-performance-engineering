#!/usr/bin/env python3
"""Sweep power caps and report perf-per-watt for a GEMM workload."""

from __future__ import annotations

import argparse
import time
from typing import Iterable, Optional

import torch

from core.harness.benchmark_harness import lock_gpu_clocks, ramp_gpu_clocks
from core.utils.power_sampling import PowerSampler, ensure_nvml_initialized

try:
    import pynvml  # type: ignore
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError("power_tuning_tool requires pynvml (nvidia-ml-py)") from exc


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}.")
    return mapping[name]


def _parse_power_limits(csv: str) -> Iterable[float]:
    limits = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        limits.append(float(part))
    if not limits:
        raise ValueError("--power-limits must include at least one value")
    return limits


def _set_power_limit(handle, limit_watts: float) -> None:
    min_limit, max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    limit_mw = int(limit_watts * 1000)
    if limit_mw < min_limit or limit_mw > max_limit:
        raise ValueError(
            f"Requested power limit {limit_watts:.1f} W outside supported range "
            f"{min_limit / 1000:.1f}-{max_limit / 1000:.1f} W."
        )
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, limit_mw)


def _run_gemm(
    *,
    device: torch.device,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    iterations: int,
    warmup: int,
    sample_interval: float,
) -> tuple[float, dict]:
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)

    for _ in range(warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    sampler = PowerSampler([device.index], interval=sample_interval)
    sampler.start()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start
    power = sampler.stop()
    sampler.close()
    return elapsed_s, power


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep power caps and report perf-per-watt.")
    parser.add_argument("--m", type=int, default=4096, help="GEMM M dimension.")
    parser.add_argument("--n", type=int, default=4096, help="GEMM N dimension.")
    parser.add_argument("--k", type=int, default=4096, help="GEMM K dimension.")
    parser.add_argument("--dtype", type=str, default="fp16", help="Data type: fp16, bf16, fp32.")
    parser.add_argument("--iterations", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Power sampling interval in seconds.",
    )
    parser.add_argument(
        "--power-limits",
        type=str,
        required=True,
        help="Comma-separated power limits in watts (e.g., 300,350,400).",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM clock lock.")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory clock lock.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for power tuning sweeps.")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    dtype = _parse_dtype(args.dtype)
    limits = list(_parse_power_limits(args.power_limits))

    ensure_nvml_initialized()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    original_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)

    print("Power tuning sweep")
    print(f"  Device: cuda:{args.device}")
    print(f"  GEMM: {args.m}x{args.k} @ {args.k}x{args.n} ({args.dtype})")
    print(f"  Limits: {', '.join(f'{l:.1f}W' for l in limits)}")

    try:
        with lock_gpu_clocks(device=args.device, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz):
            ramp_gpu_clocks(device=args.device)
            for limit in limits:
                _set_power_limit(handle, limit)
                elapsed_s, power = _run_gemm(
                    device=device,
                    m=args.m,
                    n=args.n,
                    k=args.k,
                    dtype=dtype,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    sample_interval=args.sample_interval,
                )
                flops_per_iter = 2.0 * args.m * args.n * args.k
                total_flops = flops_per_iter * args.iterations
                tflops = (total_flops / elapsed_s) / 1e12
                avg_power = float(power["avg_watts"])
                perf_per_watt = (total_flops / elapsed_s) / avg_power

                print("-")
                print(f"  Power limit: {limit:.1f} W")
                print(f"    Time: {elapsed_s:.4f} s")
                print(f"    Throughput: {tflops:.2f} TFLOP/s")
                print(f"    Power: avg {avg_power:.1f} W, max {float(power['max_watts']):.1f} W")
                print(f"    Perf/Watt: {perf_per_watt / 1e12:.3f} TFLOP/J")
    finally:
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(original_limit))


if __name__ == "__main__":
    main()
