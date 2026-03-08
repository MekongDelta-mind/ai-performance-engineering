"""Predicated threshold benchmark."""

from __future__ import annotations

from ch08.threshold_benchmark_base import ThresholdBenchmarkBase


class OptimizedThresholdBenchmark(ThresholdBenchmarkBase):
    nvtx_label = "optimized_threshold"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        self.extension.threshold_optimized(self.inputs, self.outputs, self.threshold)



def get_benchmark() -> ThresholdBenchmarkBase:
    return OptimizedThresholdBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
