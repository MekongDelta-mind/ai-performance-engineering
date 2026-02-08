"""Python harness wrapper for baseline_cutlass_gemm_fp4.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, ExecutionMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCutlassGemmFp4Benchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm_fp4",
            friendly_name="Baseline Cutlass Gemm Fp4",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            workload_params={
                "M": 128,
                "N": 7168,
                "K": 16384,
                "kIterations": 50,
                "dtype": "nvfp4",
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def get_config(self):
        config = super().get_config()
        # Avoid subprocess isolation here; the cutlass fp4 binary is stable
        # but subprocess mode intermittently hits BrokenPipe in the harness.
        config.use_subprocess = False
        config.execution_mode = ExecutionMode.THREAD
        # Match explicit CLI intent: keep NCU minimal and stable for this binary.
        config.ncu_metric_set = "minimal"
        config.ncu_replay_mode = "kernel"
        config.ncu_replay_mode_override = True
        config._sync_execution_mode()
        return config


def get_benchmark() -> BaseBenchmark:
    return BaselineCutlassGemmFp4Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
