from __future__ import annotations

from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiler_wrapper import _resolve_wrapper_loop_budget


def test_wrapper_loop_budget_defaults_to_existing_benchmark_counts() -> None:
    config = BenchmarkConfig(iterations=20, warmup=5)
    assert _resolve_wrapper_loop_budget(config) == (5, 10)


def test_wrapper_loop_budget_honors_profiling_specific_overrides() -> None:
    config = BenchmarkConfig(
        iterations=20,
        warmup=5,
        profiling_warmup=0,
        profiling_iterations=1,
    )
    assert _resolve_wrapper_loop_budget(config) == (0, 1)
