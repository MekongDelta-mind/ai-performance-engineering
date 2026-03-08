"""Optimized: Dynamic quantized KV cache with adaptive bit-widths.

Chapter 19: Blackwell-Native Precision Operations

The optimized version uses quantization to reduce memory traffic:
- INT8 for early tokens (highest precision)
- INT6 for middle tokens
- INT4 for late tokens (minimal memory)

This provides 4-8x memory reduction vs full FP32 baseline.
"""

from __future__ import annotations

from pathlib import Path

from ch19.baseline_dynamic_quantized_cache import (  # noqa: E402
    _DynamicQuantizedCacheBenchmark,
)


class OptimizedDynamicQuantizedCacheBenchmark(_DynamicQuantizedCacheBenchmark):
    """Optimized: Quantized KV cache (less memory traffic).
    
    Uses adaptive bit-widths:
    - INT8 for early tokens (highest precision needed)
    - INT6 for middle tokens 
    - INT4 for late tokens (memory savings)
    """

    def __init__(self) -> None:
        schedule = [8] * 12 + [6] * 8 + [4] * 12
        # use_fp32_baseline=False means we use quantization (default)
        super().__init__(schedule_bits=schedule, use_fp32_baseline=False)


def get_benchmark():
    return OptimizedDynamicQuantizedCacheBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
