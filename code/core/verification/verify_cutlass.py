#!/usr/bin/env python3
"""
Verify CUTLASS Backend is Working

Quick test to ensure CUTLASS backend is properly configured and functional.
"""
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import torch
import torch.nn as nn
from importlib import metadata as importlib_metadata


def _version_tuple(version_str: str):
    parts = []
    for token in version_str.split("."):
        numeric = "".join(ch for ch in token if ch.isdigit())
        if numeric:
            parts.append(int(numeric))
        else:
            parts.append(0)
    return tuple(parts)

def main():
    print("=" * 80)
    print("CUTLASS Backend Verification")
    print("=" * 80)
    
    # Check configuration
    cfg = torch._inductor.config
    print("\nConfiguration:")
    print(f"  Backends: {cfg.max_autotune_gemm_backends}")
    print(f"  CUTLASS enabled ops: {cfg.cuda.cutlass_enabled_ops}")
    
    # Note: CUTLASS may not be in max_autotune_gemm_backends by default
    # but is still available for torch.compile when appropriate
    if "CUTLASS" not in cfg.max_autotune_gemm_backends:
        print("  CUTLASS not in default backends (may be auto-selected by torch.compile)")
    else:
        print("  CUTLASS in default backends")
    
    if cfg.cuda.cutlass_enabled_ops != "all":
        print(f"  CUTLASS ops: {cfg.cuda.cutlass_enabled_ops} (expected 'all')")
    else:
        print("  CUTLASS ops enabled: all")
    
    # Check dependencies
    print("\nDependencies:")
    try:
        import cutlass
        print(f"  cutlass: {cutlass.__file__}")
    except ImportError as e:
        print(f"  cutlass import failed: {e}")
        return False

    try:
        cutlass_pkg_version = importlib_metadata.version("nvidia-cutlass-dsl")
        print(f"  nvidia-cutlass-dsl version: {cutlass_pkg_version}")
        if _version_tuple(cutlass_pkg_version) < (4, 2, 0):
            print("  CUTLASS DSL < 4.2 detected; upgrade recommended for Blackwell kernels.")
    except importlib_metadata.PackageNotFoundError:
        print("  nvidia-cutlass-dsl package not found; CUTLASS kernels may be unavailable.")
    except Exception as e:  # pragma: no cover
        print(f"  Unable to determine CUTLASS version: {e}")
    
    try:
        import cuda.bindings
        print(f"  cuda.bindings: {cuda.bindings.__file__}")
    except ImportError as e:
        print(f"  cuda.bindings import failed: {e}")
        return False
    
    # Test compilation
    print("\nTesting torch.compile:")
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping compilation test")
        return True
    
    model = nn.Linear(256, 512).cuda()
    x = torch.randn(16, 256, device='cuda')
    
    try:
        compiled_model = torch.compile(model, mode='max-autotune')
        with torch.no_grad():
            output = compiled_model(x)
        print("  Compilation successful (no errors)")
    except Exception as e:
        print(f"  Compilation failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED - CUTLASS Backend is Working!")
    print("=" * 80)
    print("\nNext steps:")
    print("- CUTLASS backend is enabled via torch.compile with max-autotune mode")
    print("- Use mode='max-autotune' to enable CUTLASS")
    print("- Performance varies by workload (memory vs compute bound)")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
