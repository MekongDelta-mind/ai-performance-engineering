"""Inline tcgen05 baseline benchmark (SM100 inline path)."""

from __future__ import annotations

from labs.fullstack_cluster import optimized_matmul_tcgen05
from labs.fullstack_cluster.capstone_benchmarks import CapstoneMatmulBenchmark
from labs.fullstack_cluster.gpu_requirements import ensure_tcgen05_supported


class BaselineCapstoneGemmTCGen05Benchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        super().__init__(
            runner=optimized_matmul_tcgen05,
            label="capstone_baseline_tcgen05_inline",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            validate_against_baseline=False,
        )

    def setup(self) -> None:
        ensure_tcgen05_supported()
        super().setup()



def get_benchmark() -> BaselineCapstoneGemmTCGen05Benchmark:
    return BaselineCapstoneGemmTCGen05Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
