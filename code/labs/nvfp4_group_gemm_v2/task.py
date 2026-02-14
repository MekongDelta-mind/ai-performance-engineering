"""Type stubs for NVFP4 group GEMM benchmark inputs/outputs.

This mirrors the reference-kernels task.py so we can reuse the CuTe DSL submission
implementation inside the AISP harness without depending on Popcorn's eval harness.
"""

from __future__ import annotations

from typing import List, Tuple, TypeVar

import torch

input_t = TypeVar(
    "input_t",
    bound=tuple[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[torch.Tensor, torch.Tensor]],
        list[tuple[int, int, int, int]],
    ],
)
output_t = TypeVar("output_t", bound=list[torch.Tensor])

