"""Optimized: Dual streams for overlapped copy and compute."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Dual-stream decode with overlapped copy and compute.
    
    Optimization: Uses separate CUDA streams for memory operations vs compute,
    allowing overlap and hiding transfer latency with computation.
    """
    cfg = DecodeConfig(
        batch_size=64,
        prompt_tokens=2048,
        decode_tokens=64,
        prefetch_batches=2,
        host_payload_mb=512,
        hidden_size=256,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="optimized_decode_streams",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
