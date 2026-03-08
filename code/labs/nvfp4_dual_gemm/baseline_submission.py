"""Baseline submission wrapper for NVFP4 dual GEMM.

This intentionally uses the official reference kernel directly.
"""

from __future__ import annotations

try:
    from reference_submission import check_implementation, generate_input, ref_kernel
    from task import input_t, output_t
except ModuleNotFoundError:
    from labs.nvfp4_dual_gemm.reference_submission import check_implementation, generate_input, ref_kernel
    from labs.nvfp4_dual_gemm.task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return ref_kernel(data)


__all__ = ["custom_kernel", "generate_input", "check_implementation"]
