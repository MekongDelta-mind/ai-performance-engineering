"""Python harness wrapper for optimized_cublaslt_gemm_fp4.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCublasltGemmFp4Benchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cublaslt_gemm_fp4",
            friendly_name="Optimized Cublaslt Gemm Fp4",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "kIterations": 10,
                "kBatchCount": 8,
                "dtype": 'fp4',
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedCublasltGemmFp4Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
