"""Python harness wrapper for baseline_warp_specialized_pipeline.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineWarpSpecializedPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_warp_specialized_pipeline",
            friendly_name="Baseline Warp Specialized Pipeline",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "TILE_SIZE": 64,
                "tiles": 512,
                "dtype": "float32",
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineWarpSpecializedPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
