"""Optimized tiling benchmark that reuses shared-memory tiles."""

from __future__ import annotations

from ch08.tiling_benchmark_base import TilingBenchmarkBase


class OptimizedTilingBenchmark(TilingBenchmarkBase):
    """Optimized implementation that loads tiles into shared memory."""

    nvtx_label = "optimized_tiling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        self.extension.matmul_tiled_fast(self.matrix_a, self.matrix_b, self.output)



def get_benchmark() -> TilingBenchmarkBase:
    """Factory function for harness discovery."""
    return OptimizedTilingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
