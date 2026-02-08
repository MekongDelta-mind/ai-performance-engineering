"""Python harness wrapper for baseline_cutlass_gemm_fp8.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCutlassGemmFp8Benchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm_fp8",
            friendly_name="Baseline Cutlass Gemm Fp8",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "M": 4096,
                "N": 4096,
                "K": 4096,
                "kIterations": 10,
                "kRepeats": 16,
                "dtype": "fp8_e4m3",
            },
        )

    def setup(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for CUTLASS FP8")
        major, _minor = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError("SKIPPED: CUTLASS FP8 requires SM90+")
        super().setup()

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineCutlassGemmFp8Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
