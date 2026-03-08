"""Dynamic allocator retry helper for Chapter 19."""

from __future__ import annotations
import argparse
import contextlib
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

try:
    from torch.cuda import memory as _cuda_memory
except Exception:  # pragma: no cover - PyTorch without CUDA memory APIs
    _CUDA_MEMORY_API_AVAILABLE = False
    _cuda_memory = None  # type: ignore[assignment]
else:
    _CUDA_MEMORY_API_AVAILABLE = True


def configure_allocator_in_process() -> None:
    """Attempt to switch to cudaMallocAsync without respawning the process."""
    if (not torch.cuda.is_available()) or (not _CUDA_MEMORY_API_AVAILABLE) or (_cuda_memory is None):
        return
    try:
        backend = _cuda_memory.get_allocator_backend()
        if backend == "cudaMallocAsync":
            return
        _cuda_memory._set_allocator_settings("backend:cudaMallocAsync")
        _cuda_memory.change_current_allocator(_cuda_memory._get_current_allocator())
        print("[allocator] Switched allocator backend to cudaMallocAsync in-process.")
    except RuntimeError as exc:
        print(f"[allocator] In-process allocator swap unavailable: {exc}")
    except AttributeError:
        # Older PyTorch builds may lack _set_allocator_settings
        pass


def comm_buffer_pool() -> contextlib.AbstractContextManager[None]:
    """
    Create a MemPool context for NCCL/NVLS communication buffers.

    Returns a no-op context when MemPool APIs are unavailable.
    """
    if (not torch.cuda.is_available()) or (not _CUDA_MEMORY_API_AVAILABLE) or (_cuda_memory is None):
        return contextlib.nullcontext()
    pool = _cuda_memory.MemPool(use_on_oom=True)
    return _cuda_memory.use_mem_pool(pool)


def allocate_on_gpu(size_mb: int) -> None:
    tensors = []
    try:
        for _ in range(8):
            tensors.append(torch.randn(size_mb * 256, 1024, device="cuda"))
            time.sleep(0.05)
    except RuntimeError as exc:
        print(f"Allocation failed: {exc}")
        raise
    finally:
        del tensors
        torch.cuda.empty_cache()
        gc.collect()


def run_child(size_mb: int) -> None:
    torch.cuda.set_device(0)
    configure_allocator_in_process()
    allocate_on_gpu(size_mb)


def run_parent(size_mb: int) -> None:
    script = Path(__file__).resolve()
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        subprocess.run([sys.executable, str(script), "--child", f"--size-mb={size_mb}"], check=True, env=env)
    except subprocess.CalledProcessError:
        print("Retrying after cleanup with half the allocation size...")
        torch.cuda.empty_cache()
        gc.collect()
        new_size = max(32, size_mb // 2)
        subprocess.run([sys.executable, str(script), "--child", f"--size-mb={new_size}"], check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Allocator retry helper")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--size-mb", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.child:
        run_child(args.size_mb)
    else:
        run_parent(args.size_mb)


if __name__ == "__main__":
    main()
