"""Kernel-level wrapper reusing the lab decode kernel benchmark."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.moe_cuda.baseline_decode_kernel import BaselineDecodeKernelBenchmark as _BaselineDecodeKernelBenchmark


class BaselineDecodeKernelBenchmark(_BaselineDecodeKernelBenchmark):
    """Expose the baseline decode kernel benchmark from the lab package."""

    pass


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
