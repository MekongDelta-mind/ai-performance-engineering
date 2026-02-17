"""Python harness wrapper for optimized_memory_transfer_multigpu.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
import torch

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedMemoryTransferMultigpuBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    multi_gpu_required = True

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_memory_transfer_multigpu",
            friendly_name="Optimized Memory Transfer Multigpu",
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
    return OptimizedMemoryTransferMultigpuBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
