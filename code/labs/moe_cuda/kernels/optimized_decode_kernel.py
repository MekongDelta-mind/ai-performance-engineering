"""Kernel-level wrapper reusing the lab optimized decode kernel benchmark."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.optimized_decode_kernel import OptimizedDecodeKernelBenchmark as _OptimizedDecodeKernelBenchmark


class OptimizedDecodeKernelBenchmark(_OptimizedDecodeKernelBenchmark):
    """Expose the optimized decode kernel benchmark from the lab package."""

    pass


def get_benchmark() -> BaseBenchmark:
    return OptimizedDecodeKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
