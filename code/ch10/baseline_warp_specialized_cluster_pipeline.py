"""Python harness wrapper for baseline_warp_specialized_cluster_pipeline.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineWarpSpecializedClusterPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_warp_specialized_cluster_pipeline",
            friendly_name="Baseline Warp Specialized Cluster Pipeline",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "N": 1024,
                "dtype": "float32",
                "batch_size": 1,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineWarpSpecializedClusterPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
