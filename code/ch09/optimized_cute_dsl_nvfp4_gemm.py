"""Python harness wrapper for optimized_cute_dsl_nvfp4_gemm.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, ExecutionMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedCuteDslNvfp4GemmBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_cute_dsl_nvfp4_gemm",
            friendly_name="Optimized Cute DSL Nvfp4 Gemm",
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
        # Match baseline config and avoid subprocess isolation here; subprocess
        # mode intermittently hits BrokenPipe under profiling.
        config.use_subprocess = False
        config.execution_mode = ExecutionMode.THREAD
        # Keep NCU capture overhead low and stable for this binary benchmark.
        config.ncu_metric_set = "minimal"
        config.ncu_replay_mode = "kernel"
        config.ncu_replay_mode_override = True
        config._sync_execution_mode()
        return config


def get_benchmark() -> BaseBenchmark:
    return OptimizedCuteDslNvfp4GemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
