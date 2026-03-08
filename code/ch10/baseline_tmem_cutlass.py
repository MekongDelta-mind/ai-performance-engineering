"""Baseline CUTLASS TMEM benchmark (tcgen05) exposed for harness discovery.

This simply reuses the tcgen05 baseline matmul benchmark and surfaces it under
the `tmem_cutlass` example name within Chapter 10 (no cross-chapter aliasing).
"""

from __future__ import annotations

from pathlib import Path

from ch10.baseline_matmul_tcgen05 import BaselineMatmulTCGen05Benchmark


class BaselineTmemCutlassBenchmark(BaselineMatmulTCGen05Benchmark):
    """Uses the torch matmul baseline as the CUTLASS/TMEM reference."""


def get_benchmark() -> BaselineTmemCutlassBenchmark:
    return BaselineTmemCutlassBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
