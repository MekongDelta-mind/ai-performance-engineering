"""Baseline for FP8 decode optimization: BF16 prefill-only workload.

This baseline matches `optimized_decode_fp8.py` exactly, but runs in BF16.
Keeping a dedicated baseline ensures the FP8 comparison is workload-equivalent.
"""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    # Prefill-dominant regime where FP8 benefits are visible.
    cfg = DecodeConfig(
        batch_size=128,
        prompt_tokens=1024,
        decode_tokens=0,
        hidden_size=8192,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_decode_fp8",
        iterations=12,
        warmup=15,
    )
    bench = attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
    bench.signature_equivalence_group = "labs_decode_fp8_precision"
    bench.signature_equivalence_ignore_fields = ("precision_flags",)
    return bench


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
