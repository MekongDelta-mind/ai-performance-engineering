"""Shared Triton batched persistent GEMM helpers (Chapter 14).

Used by multiple benchmark variants to keep workloads equivalent while exploring
different launch strategies.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_persistent_batched_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    num_tiles_m,
    num_tiles_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent batched GEMM with swizzled tile ordering.

    - Launch grid is 1D: (NUM_SMS,)
    - Each program processes many (batch, m_tile, n_tile) tiles via a strided loop.
    """
    pid = tl.program_id(0)

    tiles_per_batch = num_tiles_m * num_tiles_n
    total_tiles = B * tiles_per_batch

    for tile_id in range(pid, total_tiles, NUM_SMS):
        batch_id = tile_id // tiles_per_batch
        tile_in_batch = tile_id - batch_id * tiles_per_batch

        group_id = tile_in_batch // (GROUP_M * num_tiles_n)
        first_tile_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_tiles_m - first_tile_m, GROUP_M)

        tile_in_group = tile_in_batch - group_id * (GROUP_M * num_tiles_n)
        pid_m = first_tile_m + (tile_in_group % group_size_m)
        pid_n = tile_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + batch_id * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + batch_id * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            mask_a = (batch_id < B) & (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
            mask_b = (batch_id < B) & (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            acc += tl.dot(a, b)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + batch_id * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (batch_id < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask_c)


def matmul_persistent_batched(
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
    num_sms: int,
    out: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    group_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Launch a batched persistent GEMM kernel.

    Args:
        a_batch: (B, M, K) tensor
        b_batch: (B, K, N) tensor
        num_sms: number of SMs to launch (grid = (num_sms,))
    """
    if a_batch.ndim != 3 or b_batch.ndim != 3:
        raise ValueError("matmul_persistent_batched expects 3D batched inputs (B, M, K) and (B, K, N)")
    if a_batch.shape[0] != b_batch.shape[0]:
        raise ValueError("Batch dimension mismatch between a_batch and b_batch")
    if a_batch.shape[2] != b_batch.shape[1]:
        raise ValueError("Inner K dimension mismatch between a_batch and b_batch")

    B, M, K = a_batch.shape
    _, _, N = b_batch.shape

    if out.shape != (B, M, N):
        raise ValueError("matmul_persistent_batched() requires a preallocated output buffer with matching shape")
    c_batch = out

    num_tiles_m = triton.cdiv(M, block_m)
    num_tiles_n = triton.cdiv(N, block_n)

    grid = (num_sms,)
    gemm_persistent_batched_kernel[grid](
        a_batch,
        b_batch,
        c_batch,
        B,
        M,
        N,
        K,
        a_batch.stride(0),
        a_batch.stride(1),
        a_batch.stride(2),
        b_batch.stride(0),
        b_batch.stride(1),
        b_batch.stride(2),
        c_batch.stride(0),
        c_batch.stride(1),
        c_batch.stride(2),
        num_tiles_m,
        num_tiles_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        NUM_SMS=num_sms,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return c_batch
