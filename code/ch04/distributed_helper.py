from __future__ import annotations

"""Helper module for distributed training with auto-fallback to single-GPU mode."""

import os


def setup_single_gpu_env() -> None:
    """Setup environment variables for single-GPU mode if not already set."""
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("LOCAL_RANK", "0")
