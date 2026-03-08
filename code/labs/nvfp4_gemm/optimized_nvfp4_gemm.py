"""Python harness wrapper for optimized_nvfp4_gemm.cu."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import BaseBenchmark, ExecutionMode


class OptimizedNvfp4GemmBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized NVFP4 GEMM CUDA binary."""

    def __init__(self) -> None:
        lab_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=lab_dir,
            binary_name="optimized_nvfp4_gemm",
            friendly_name="Optimized NVFP4 GEMM",
            iterations=1,
            warmup=5,
            timeout_seconds=240,
            workload_params={
                "dtype": "nvfp4",
                "shapes": "(128,7168,16384),(128,4096,7168),(128,7168,2048)",
                "kIterations": 50,
            },
        )

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def get_input_signature(self) -> dict:
        # Explicitly encode leaderboard shapes as numeric fields so signature
        # construction remains deterministic and int-only.
        return simple_signature(
            batch_size=3,
            dtype="nvfp4",
            m0=128, n0=7168, k0=16384,
            m1=128, n1=4096, k1=7168,
            m2=128, n2=7168, k2=2048,
            iterations=50,
        )

    def get_config(self):
        config = super().get_config()
        config.use_subprocess = False
        config.execution_mode = ExecutionMode.THREAD
        config.ncu_metric_set = "minimal"
        config.ncu_replay_mode = "kernel"
        config.ncu_replay_mode_override = True
        config._sync_execution_mode()
        return config


def get_benchmark() -> BaseBenchmark:
    return OptimizedNvfp4GemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
