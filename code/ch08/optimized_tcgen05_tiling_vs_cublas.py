"""Optimized cuBLAS matmul for tcgen05 tiling comparison."""

from __future__ import annotations

import torch

from ch08.tcgen05_tiling_vs_cublas_benchmark_base import Tcgen05TilingVsCublasBase
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTcgen05TilingVsCublasBenchmark(Tcgen05TilingVsCublasBase):
    """Optimized: cuBLAS-backed torch.matmul."""

    nvtx_label = "optimized_tcgen05_tiling_vs_cublas"

    def benchmark_fn(self) -> None:
        if self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.no_grad():
                self.output = torch.matmul(self.matrix_a, self.matrix_b)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05TilingVsCublasBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
