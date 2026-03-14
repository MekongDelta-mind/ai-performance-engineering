"""Python harness wrapper for optimized_dsmem_reduction_v3.cu - Working DSMEM for B200."""

from __future__ import annotations
from typing import Optional
from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class OptimizedDSMEMReductionV3Benchmark(CudaBinaryBenchmark):
    """Wraps the working DSMEM cluster reduction kernel for B200.
    
    KEY FIXES for B200/CUDA 13.0:
    1. NO __cluster_dims__ attribute (conflicts with runtime cluster dims)
    2. STATIC shared memory (dynamic extern fails on B200)
    3. cudaLaunchKernelExC with void* args[] (not typed parameters)
    4. Final cluster.sync() before exit
    """

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_dsmem_reduction_v3",
            friendly_name="Optimized Dsmem Reduction V3",
            iterations=3,
            warmup=5,
            # Full deep-dive instrumentation can slow the standalone CUDA binary
            # enough that the default subprocess budget becomes a false failure.
            timeout_seconds=600,
            workload_params={
                "batch_size": 2048,
                "dtype": "float32",
                "N": 64 * 1024 * 1024,
                "cluster_size": 2,
                "block_elems": 4096,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(64 * 1024 * 1024 * 4))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from core.benchmark.metrics import compute_reduction_metrics
        return compute_reduction_metrics(
            num_elements=getattr(self, 'num_elements', 64 * 1024 * 1024),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 0.03),
        )

    def get_input_signature(self) -> dict:
        """Signature for DSMEM reduction v3."""
        return simple_signature(
            batch_size=1,
            dtype="float32",
            N=64 * 1024 * 1024,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def get_config(self) -> BenchmarkConfig:
        config = super().get_config()
        # This standalone CUDA binary does not benefit from full library tracing,
        # and deep-dive NSYS captures are more reliable with the light preset.
        config.nsys_timeout_seconds = 300
        config.nsys_preset_override = "light"
        config.profiling_warmup = 0
        config.profiling_iterations = 1
        config.profile_env_overrides = {
            "AISP_CUDA_BINARY_PROFILE_WARMUP": "0",
            "AISP_CUDA_BINARY_PROFILE_ITERATIONS": "1",
        }
        return config


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDSMEMReductionV3Benchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
