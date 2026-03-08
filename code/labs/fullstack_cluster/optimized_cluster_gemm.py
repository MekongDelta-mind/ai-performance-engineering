"""Benchmark wrapper for the capstone optimized GEMM kernel."""

from __future__ import annotations

from labs.fullstack_cluster import optimized_matmul
from labs.fullstack_cluster.capstone_benchmarks import CapstoneMatmulBenchmark


class OptimizedCapstoneGemmBenchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        super().__init__(
            runner=optimized_matmul,
            label="capstone_optimized",
            iterations=3,
            warmup=5,
            timeout_seconds=300,
            validate_against_baseline=True,
        )



def get_benchmark() -> OptimizedCapstoneGemmBenchmark:
    return OptimizedCapstoneGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
