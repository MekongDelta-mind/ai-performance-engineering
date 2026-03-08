"""Optimized variant for cuDNN/Flash SDPA lab (shares implementation with baseline)."""

from __future__ import annotations

from labs.cudnn_sdpa_bench.baseline_flash_sdp import (
    FlashSDPLabBenchmark,
    _parse_cli_backend,
    _select_backend,
)

_DEFAULT_BACKEND = "flash"


def get_benchmark() -> FlashSDPLabBenchmark:
    # Reuse the same benchmark but bias toward the Flash backend for peak throughput.
    bench = FlashSDPLabBenchmark()
    bench.backend = _select_backend(_DEFAULT_BACKEND)
    return bench


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
