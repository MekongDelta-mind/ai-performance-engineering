"""Helpers for strict GPU integration tests."""

from __future__ import annotations

import pytest
import torch

from core.harness.validity_checks import validate_environment


def skip_if_strict_benchmark_env_invalid() -> None:
    """Skip when the harness would correctly reject the live benchmark host."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required - NVIDIA GPU and tools must be available")
    result = validate_environment(device=torch.device("cuda"))
    if result.errors:
        reason = " | ".join(result.errors[:2])
        pytest.skip(f"Strict GPU benchmark environment unavailable: {reason}")
