"""Optimized paged KV-cache prefetch benchmark (pinned async + host prefetch + prefetch).

- Uses a host staging thread plus pinned buffers with an async copy stream.
- Prefetches the next page to overlap H2D copies with compute.
- Uses a pinned host cache (memmap disabled) to isolate overlap effects.
"""

from __future__ import annotations

import torch

from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark


def get_benchmark() -> PagedKVOffloadBenchmark:
    cfg = PagedKVConfig(
        batch_size=4,
        num_heads=16,
        head_dim=128,
        max_seq_len=65536,
        page_tokens=8192,
        decode_tokens=1,
        repeat_pages=128,
        use_pinned_stage=True,
        use_async_stream=True,
        use_memmap=False,
        prefer_fp8=False,
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=True,
        use_direct_h2d=False,
        use_host_prefetch_thread=True,
    )
    return PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_prefetch_optimized")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
