"""Type stubs for NVFP4 GEMV challenge inputs/outputs."""

from __future__ import annotations

from typing import TypeVar

import torch

input_t = TypeVar(
    "input_t",
    bound=tuple[
        torch.Tensor,  # a: [m, k//2, l] as float4_e2m1fn_x2 view
        torch.Tensor,  # b: [128, k//2, l] as float4_e2m1fn_x2 view
        torch.Tensor,  # sfa: [m, k//16, l] float8
        torch.Tensor,  # sfb: [128, k//16, l] float8
        torch.Tensor,  # sfa_permuted
        torch.Tensor,  # sfb_permuted
        torch.Tensor,  # c: [m, 1, l] float16
    ],
)

output_t = TypeVar("output_t", bound=torch.Tensor)

