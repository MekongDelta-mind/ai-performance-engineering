from __future__ import annotations

import pytest
import torch

import ch06.baseline_bank_conflicts as baseline_bank_conflicts
import ch06.optimized_bank_conflicts as optimized_bank_conflicts
from ch06.baseline_adaptive import BaselineAdaptiveBenchmark
from ch06.baseline_autotuning import BaselineAutotuningBenchmark
from ch06.baseline_bank_conflicts import BaselineBankConflictsBenchmark
from ch06.optimized_adaptive import OptimizedAdaptiveBenchmark
from ch06.optimized_autotuning import OptimizedAutotuningBenchmark
from ch06.optimized_bank_conflicts import OptimizedBankConflictsBenchmark


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "benchmark_cls",
    [
        BaselineAdaptiveBenchmark,
        OptimizedAdaptiveBenchmark,
        BaselineAutotuningBenchmark,
        OptimizedAutotuningBenchmark,
    ],
)
def test_chunked_processing_setup_keeps_public_output_empty(benchmark_cls: type) -> None:
    bench = benchmark_cls()
    bench.N = 4096
    if hasattr(bench, "static_chunk"):
        bench.static_chunk = 256
    if hasattr(bench, "candidates"):
        bench.candidates = [128, 256]
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()


class _FakeBankConflictsExtension:
    def bank_conflicts(self, output: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output.copy_(input_tensor)

    def bank_conflicts_padded(self, output: torch.Tensor, input_tensor: torch.Tensor) -> None:
        output.copy_(input_tensor)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("benchmark_cls", "module"),
    [
        (BaselineBankConflictsBenchmark, baseline_bank_conflicts),
        (OptimizedBankConflictsBenchmark, optimized_bank_conflicts),
    ],
)
def test_bank_conflicts_setup_keeps_public_output_empty(
    benchmark_cls: type,
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "load_bank_conflicts_extension", lambda: _FakeBankConflictsExtension())

    bench = benchmark_cls()
    bench.N = 4096
    bench.repeats = 2
    try:
        bench.setup()
        assert bench.output is None
        assert bench._output_buffer is not None
        bench.benchmark_fn()
        assert isinstance(bench.output, torch.Tensor)
    finally:
        bench.teardown()
