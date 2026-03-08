"""Baseline: CUDA decode kernel (global-load) for the TMA double-buffered comparison.

This benchmark reuses the `labs/moe_cuda` baseline kernel wrapper so that
`optimized_decode_double_buffer_tma.py` has a valid, equivalent baseline.
"""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.baseline_decode_kernel import BaselineDecodeKernelBenchmark


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
