"""Baseline NVFP4 grouped GEMM (competition case 0, v2 custom CUDA tcgen05 kernel).

This baseline uses the from-scratch tcgen05/UMMA path in its most conservative configuration
(no cluster/cta_group::2, UnrollN=1, no CUDA-graph replay) so profiling stays tractable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep baseline behavior stable by default, but allow explicit overrides for experiments.
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_UNROLL_N", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_CLUSTER_DIM_X", "1")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_ENABLE_EXPERIMENTAL_CTA2", "0")
os.environ.setdefault("AISP_NVFP4_GROUP_GEMM_V2_ENABLE_TMA_MULTICAST", "0")

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
    case = COMPETITION_CASES[0]
    bench = NVFP4GroupGemmBenchmark(
        case=case,
        custom_kernel=custom_kernel_v2_custom_cuda_tcgen05,
        prepare=prepare_v2_custom_cuda_tcgen05,
        inputs_per_iteration=15,
        capture_iter_graph=False,
        name=f"nvfp4_group_gemm_{case.name}_baseline_v2_custom_cuda_tcgen05",
    )
    return attach_benchmark_metadata(bench, __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
