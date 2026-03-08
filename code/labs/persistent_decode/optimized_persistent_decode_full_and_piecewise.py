"""Graph-mode sweep for persistent decode: full vs piecewise vs fallback."""

from __future__ import annotations

from labs.persistent_decode.optimized_persistent_decode_graphs import (
    GraphMode,
    OptimizedPersistentDecodeGraphsBenchmark,
)
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedPersistentDecodeFullAndPiecewiseBenchmark(OptimizedPersistentDecodeGraphsBenchmark):
    """Default to FULL_AND_PIECEWISE to mirror vLLM graph heuristics."""

    def __init__(self) -> None:
        super().__init__(graph_mode=GraphMode.FULL_AND_PIECEWISE)
        # Inherit real output/shape behavior from parent


def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeFullAndPiecewiseBenchmark()

if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
