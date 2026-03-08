"""Optimized: Pinned host memory + copy stream for async H2D transfers."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Optimized decode with pinned memory and torch.compile.
    
    Key optimizations:
    1. Pinned host memory for async non-blocking transfers
    2. torch.compile for fused operations
    3. Copy stream for overlapping H2D with compute
    """
    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=256,
        decode_tokens=64,
        hidden_size=1024,
        use_pinned_host=True,
        use_copy_stream=True,
        use_torch_compile=True,  # Enable compilation for speedup
        label="optimized_decode_pinned",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
