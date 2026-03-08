"""Python harness wrapper for optimized_fused_l2norm.cu."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedFusedL2NormBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized fused L2 norm kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        n_elems = 1 << 20
        iterations=1
        bytes_per_elem = 12  # two reads + one write (float32)
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_fused_l2norm",
            friendly_name="Optimized Fused L2Norm",
            iterations=1,
            warmup=5,
            timeout_seconds=90,
            workload_params={
                "N": n_elems,
                "iterations": iterations,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(
            bytes_per_iteration=float(n_elems * bytes_per_elem),
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics."""
        return None



def get_benchmark() -> OptimizedFusedL2NormBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedFusedL2NormBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
