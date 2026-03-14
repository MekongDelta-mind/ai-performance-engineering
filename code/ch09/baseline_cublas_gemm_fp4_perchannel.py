"""Python harness wrapper for baseline_cublas_gemm_fp4_perchannel.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCublasGemmFp4PerchannelBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cublas_gemm_fp4_perchannel",
            friendly_name="Baseline Cublas Gemm Fp4 Perchannel",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "FP4_BLOCK_SIZE": 16,
                "kM": 4096,
                "kN": 4096,
                "kK": 4096,
                "kIterations": 10,
                "kBatchCount": 1,
                "dtype": 'fp4',
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def get_config(self) -> BenchmarkConfig:
        config = super().get_config()
        # Deep-dive Nsight Systems captures for this binary exceed the default budget.
        config.nsys_timeout_seconds = 1200
        return config


def get_benchmark() -> BaseBenchmark:
    return BaselineCublasGemmFp4PerchannelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
