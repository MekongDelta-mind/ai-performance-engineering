"""Shared helpers for single-GPU symmetric-memory perf benchmarks."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def build_square_verification_probe(
    tensor: torch.Tensor,
    *,
    max_elements: int = 256 * 256,
) -> Tuple[torch.Tensor, int]:
    """Return the largest square probe view that fits within the tensor."""
    available = int(tensor.numel())
    if available <= 0:
        raise ValueError("Verification probe requires a non-empty tensor")

    probe_numel = min(available, max_elements)
    side = math.isqrt(probe_numel)
    if side <= 0:
        raise ValueError("Verification probe side length must be positive")
    probe_numel = side * side
    return tensor[:probe_numel].view(side, side).detach(), probe_numel
