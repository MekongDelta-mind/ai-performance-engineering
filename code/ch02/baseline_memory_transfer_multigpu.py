"""Python harness wrapper for baseline_memory_transfer_multigpu.cu."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineMemoryTransferMultigpuBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    multi_gpu_required = True

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_memory_transfer_multigpu",
            friendly_name="Baseline Memory Transfer Multigpu",
            iterations=5,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "dtype": 'float32',
                "batch_size": 1,
            },
        )

    def get_config(self) -> BenchmarkConfig:
        cfg = super().get_config()
        cfg.multi_gpu_required = True
        return cfg

    def get_custom_metrics(self) -> Optional[dict]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMemoryTransferMultigpuBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
