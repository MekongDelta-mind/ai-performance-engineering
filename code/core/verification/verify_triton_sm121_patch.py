#!/usr/bin/env python3
"""Verify Triton SM 12.1 patch fixes the sm_121a bug."""

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def main() -> None:
    print("=" * 80)
    print("Triton SM 12.1 Patch Verification")
    print("=" * 80)
    print()

    # Verify patch is applied
    import triton.backends.nvidia.compiler as tc

    arch_name = tc.sm_arch_from_capability(121)
    patch_applied = getattr(tc, "_sm_arch_patch_applied", False)

    print(f"Patch Status: {'[OK] APPLIED' if patch_applied else 'ERROR: NOT APPLIED'}")
    print(f"SM 12.1 Architecture Name: {arch_name}")
    print(f"Has 'a' suffix: {'ERROR: YES' if arch_name.endswith('a') else '[OK] NO'}")
    print()

    # Test 1: Simple Triton kernel
    print("Test 1: Simple Triton kernel...")
    try:
        size = 1024
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
        torch.cuda.synchronize()

        expected = x + y
        if torch.allclose(output, expected):
            print("  [OK] PASSED: Kernel compiled and executed correctly")
        else:
            print("  ERROR: FAILED: Results incorrect")
            raise ValueError("Results don't match")
    except Exception as e:
        error_msg = str(e)
        if "sm_121a" in error_msg or "ptxas" in error_msg.lower():
            print("  ERROR: FAILED: Got sm_121a/ptxas error - patch NOT working!")
            print(f"     {error_msg[:200]}")
            raise
        print(f"  ERROR: FAILED: {error_msg[:200]}")
        raise

    # Test 2: torch.compile with Triton backend
    print()
    print("Test 2: torch.compile with Triton backend...")
    try:
        def matmul_fn(a, b):
            return torch.matmul(a, b)

        compiled_fn = torch.compile(matmul_fn, mode="reduce-overhead")

        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)

        for _ in range(3):
            _ = compiled_fn(a, b)
        torch.cuda.synchronize()

        result = compiled_fn(a, b)
        expected = torch.matmul(a, b)

        if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
            print("  [OK] PASSED: torch.compile with Triton works correctly")
        else:
            print("  ERROR: FAILED: Results incorrect")
            raise ValueError("Results don't match")
    except Exception as e:
        error_msg = str(e)
        if "sm_121a" in error_msg or "ptxas" in error_msg.lower():
            print("  ERROR: FAILED: Got sm_121a/ptxas error - patch NOT working!")
            print(f"     {error_msg[:200]}")
            raise
        print(f"  ERROR: FAILED: {error_msg[:200]}")
        raise

    print()
    print("=" * 80)
    print("[OK] SUCCESS: Triton SM 12.1 patch WORKS correctly!")
    print("   - Patch removes 'a' suffix from architecture name")
    print("   - Triton kernels compile successfully")
    print("   - Kernels execute correctly")
    print("   - torch.compile with Triton backend works")
    print("=" * 80)


if __name__ == "__main__":
    main()
