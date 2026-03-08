"""Benchmark wrapper for the capstone baseline GEMM kernel."""

from __future__ import annotations

from labs.fullstack_cluster import baseline_matmul
from labs.fullstack_cluster.capstone_benchmarks import CapstoneMatmulBenchmark


class BaselineCapstoneGemmBenchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        super().__init__(
            runner=baseline_matmul,
            label="capstone_baseline",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            validate_against_baseline=False,
        )

def get_benchmark() -> BaselineCapstoneGemmBenchmark:
    return BaselineCapstoneGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
