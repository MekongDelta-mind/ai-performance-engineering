"""Baseline decode loop - eager mode, no optimizations."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=256,
        decode_tokens=64,
        hidden_size=1024,
        use_pinned_host=False,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_decode",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
