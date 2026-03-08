"""Baseline KV offload benchmark using pageable host memory."""

from __future__ import annotations

from pathlib import Path

from labs.persistent_decode.nvlink_offload_common import NvlinkOffloadBenchmark, OffloadConfig


def get_benchmark() -> NvlinkOffloadBenchmark:
    cfg = OffloadConfig(
        use_pinned=False,
        non_blocking=False,
        use_copy_stream=False,
        batch_size=4,
        num_layers=4,
        num_heads=16,
        head_dim=64,
        max_seq_len=4096,
        chunk_tokens=4096,
    )
    return NvlinkOffloadBenchmark(cfg, label="nvlink_offload_baseline")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
