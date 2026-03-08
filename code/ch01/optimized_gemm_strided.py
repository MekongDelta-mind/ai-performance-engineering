"""Python harness wrapper for optimized_gemm_strided.cu."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedGemmStridedBenchmark(CudaBinaryBenchmark):
    """Wraps the strided batched GEMM CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m, n, k = 32, 256, 256
        batch_count = 40
        inner_iterations = 100
        bytes_a = m * k * 4
        bytes_b = k * n * 4
        bytes_c = m * n * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_gemm_strided",
            friendly_name="Optimized GEMM (Strided Batched)",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            workload_params={
                "M": m,
                "N": n,
                "K": k,
                "batch_count": batch_count,
                "inner_iterations": inner_iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float((bytes_a + bytes_b + bytes_c) * batch_count),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGemmStridedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
