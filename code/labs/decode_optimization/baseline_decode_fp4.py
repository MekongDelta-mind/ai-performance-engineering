"""Baseline for FP4 decode optimization: BF16 prefill-only workload.

This baseline matches `optimized_decode_fp4.py` exactly, but runs in BF16.
Keeping a dedicated baseline ensures the FP4 comparison is workload-equivalent.
"""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    # Prefill-dominant regime where FP4 benefits are visible.
    cfg = DecodeConfig(
        batch_size=64,
        prompt_tokens=2048,
        decode_tokens=0,
        hidden_size=8192,
        use_te_mlp=True,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_decode_fp4",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
