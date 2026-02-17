"""Python harness wrapper for baseline_micro_tiling_matmul.cu."""

from __future__ import annotations

from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineMicroTilingMatmulBenchmark(CudaBinaryBenchmark):
    """Wraps the naive (no tiling) matmul kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_micro_tiling_matmul",
            friendly_name="Baseline Micro Tiling Matmul",
            iterations=1,
            warmup=5,
            timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for micro_tiling_matmul."""
        from core.benchmark.metrics import compute_roofline_metrics

        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=getattr(self, "_last_elapsed_ms", 1.0),
            precision="fp32",
        )

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return simple_signature(batch_size=1, dtype="float32", workload=1).to_dict()


def get_benchmark() -> BaselineMicroTilingMatmulBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineMicroTilingMatmulBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
