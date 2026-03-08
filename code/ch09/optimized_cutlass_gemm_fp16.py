"""Python harness wrapper for optimized_cutlass_gemm_fp16.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCutlassGemmFp16Benchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cutlass_gemm_fp16",
            friendly_name="Optimized Cutlass Gemm Fp16",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "M": 2048,
                "N": 2048,
                "K": 2048,
                "kIterations": 10,
                "kRepeats": 64,
                "dtype": "float16",
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedCutlassGemmFp16Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
