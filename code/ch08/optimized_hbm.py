"""HBM optimized benchmark with vectorized, contiguous access."""

from __future__ import annotations

from ch08.hbm_benchmark_base import HBMBenchmarkBase


class OptimizedHBMBenchmark(HBMBenchmarkBase):
    nvtx_label = "optimized_hbm"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_row is not None
        assert self.output is not None
        self.extension.hbm_optimized(self.matrix_row, self.output)



def get_benchmark() -> HBMBenchmarkBase:
    return OptimizedHBMBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
