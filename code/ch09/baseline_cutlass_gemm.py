"""Python harness wrapper for baseline CUTLASS GEMM binary."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCutlassGemmBenchmark(CudaBinaryBenchmark):
    """Wraps the baseline CUTLASS GEMM kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        m = n = k = 1024
        iterations=1
        repeats = 32
        bytes_a = m * k * 4
        bytes_b = k * n * 4
        bytes_c = m * n * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm",
            friendly_name="Baseline Cutlass Gemm",
            iterations=1,
            warmup=5,
            timeout_seconds=120,
            workload_params={
                "M": m,
                "N": n,
                "K": k,
                "kIterations": iterations,
                "kRepeats": repeats,
                "dtype": "tf32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(bytes_a + bytes_b + bytes_c),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for cutlass_gemm."""
        return None  # Metrics computed by CUDA binary



def get_benchmark() -> BaselineCutlassGemmBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCutlassGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
