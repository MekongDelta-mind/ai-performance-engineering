"""Software-dequant baseline for the block scaling lab."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark
from labs.block_scaling.block_scaling_benchmarks import (
    BaselineBlockScalingBenchmarkBase,
)


class BaselineBlockScalingBenchmark(BaselineBlockScalingBenchmarkBase):
    """Materialize scale factors in BF16 and then call matmul."""


def get_benchmark() -> BaseBenchmark:
    return BaselineBlockScalingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
