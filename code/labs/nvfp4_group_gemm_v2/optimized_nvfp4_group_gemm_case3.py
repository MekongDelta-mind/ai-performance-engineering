"""Optimized NVFP4 grouped GEMM (competition case 3, v2 custom CUDA path).

This is the v2 integration point for the from-scratch Blackwell kernel work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tuning knobs for the v2 CUDA extension (kept explicit so harness defaults stay unchanged).
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_BLOCK_M", "8")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_BLOCK_N", "32")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_KPACK_TILE", "64")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N", "2")
# Optional experiments (disabled by default):
# - Cluster launch + TMA multicast (share A/SFA across the N tiles of each M tile):
#     AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X=8
#     AISP_NVFP4_GROUP_GEMM_V2_ENABLE_TMA_MULTICAST=1
# NOTE: The cta_group::2 / cluster path is still under active bring-up. Enable it explicitly via
# environment variables when testing correctness/perf, e.g.:
#   AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X=2
#   AISP_NVFP4_GROUP_GEMM_V2_ENABLE_EXPERIMENTAL_CTA2=1

from core.harness.benchmark_harness import BaseBenchmark
from labs.nvfp4_group_gemm_v2.custom_cuda_submission import (
    custom_kernel_v2_custom_cuda_tcgen05,
    prepare_v2_custom_cuda_tcgen05,
)
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_common import (
    COMPETITION_CASES,
    NVFP4GroupGemmBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    case = COMPETITION_CASES[3]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_v2_custom_cuda_tcgen05,
        prepare=prepare_v2_custom_cuda_tcgen05,
        inputs_per_iteration=15,
        capture_iter_graph=True,
        name=f"nvfp4_group_gemm_{case.name}_optimized_v2_custom_cuda_tcgen05",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
