"""Python harness wrapper for optimized_cutlass_gemm_fp4_perchannel.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCutlassGemmFp4PerchannelBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cutlass_gemm_fp4_perchannel",
            friendly_name="Optimized Cutlass Gemm Fp4 Perchannel",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "kM": 4096,
                "kN": 4096,
                "kK": 4096,
                "kIterations": 10,
                "AlignmentA": 32,
                "AlignmentB": 32,
                "AlignmentC": 128,
                "AlignmentD": 128,
                "dtype": 'fp4',
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedCutlassGemmFp4PerchannelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
