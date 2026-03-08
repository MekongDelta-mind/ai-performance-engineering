"""Python harness wrapper for baseline_hbm_peak.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineHbmPeakBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        size_bytes = 512 * 1024 * 1024
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_hbm_peak",
            friendly_name="Baseline Hbm Peak",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "bytes": size_bytes,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(size_bytes * 2))

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineHbmPeakBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
