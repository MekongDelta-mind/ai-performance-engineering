"""Python harness wrapper for optimized cuBLASLt GEMM binary."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCublasltGemmBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized cuBLASLt GEMM driver."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m = n = k = 1024
        micro_batches = 32
        iterations=1
        bytes_a = m * k * 4
        bytes_b = k * n * 4
        bytes_c = m * n * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cublaslt_gemm",
            friendly_name="Optimized Cublaslt Gemm",
            iterations=1,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "M": m,
                "N": n,
                "K": k,
                "micro_batches": micro_batches,
                "iterations": iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(bytes_a + bytes_b + bytes_c),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for GEMM."""
        return None  # Metrics computed by CUDA binary


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedCublasltGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
