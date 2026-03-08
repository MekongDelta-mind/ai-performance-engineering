"""Optimized NVSHMEM IBGDA microbenchmark (single-GPU)."""

from __future__ import annotations

from ch04.nvshmem_ibgda_microbench_multigpu import NvshmemIbgdaMicrobench
from core.harness.benchmark_harness import BenchmarkConfig

_DEFAULT_KWARGS = dict(
    mode="put",
    bytes_per_message=1048576,
    ctas=16,
    threads=256,
    iters=200,
)


class OptimizedNvshmemIbgdaMicrobenchSingle(NvshmemIbgdaMicrobench):
    # NVSHMEM IBGDA behavior is defined for PE-to-PE communication; require >=2 GPUs.
    multi_gpu_required = True

    def __init__(self) -> None:
        super().__init__(enable_ibgda=True, world_size=2, ibgda_batch=16, **_DEFAULT_KWARGS)

    def get_config(self) -> BenchmarkConfig:
        config = super().get_config()
        config.multi_gpu_required = True
        config.single_gpu = False
        return config


def get_benchmark() -> NvshmemIbgdaMicrobench:
    return OptimizedNvshmemIbgdaMicrobenchSingle()
