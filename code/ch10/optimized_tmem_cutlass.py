"""Optimized CUTLASS TMEM benchmark (tcgen05) for harness discovery.

This re-exports the tcgen05 optimized matmul kernel (CUTLASS/CuTe + TMEM
epilogue) under the `tmem_cutlass` example name within Chapter 10.
"""

from __future__ import annotations

from pathlib import Path

from ch10.optimized_matmul_tcgen05 import OptimizedMatmulTCGen05Benchmark


class OptimizedTmemCutlassBenchmark(OptimizedMatmulTCGen05Benchmark):
    """CUTLASS/CuTe TMEM benchmark wrapper."""


def get_benchmark() -> OptimizedTmemCutlassBenchmark:
    return OptimizedTmemCutlassBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
