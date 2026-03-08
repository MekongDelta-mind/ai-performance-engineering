"""Fixture helpers for cross-module hot-path analysis tests."""

from __future__ import annotations

import torch


def imported_random_input(device: torch.device) -> torch.Tensor:
    return torch.randn(8, 8, device=device)


class ImportedHostTransferHelper:
    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.cpu()
