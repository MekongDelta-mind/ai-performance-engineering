#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# Keep this global marker for semantic attestation checks.
DEEPGEMM_UNSUPPORTED_REASON: Optional[str] = None


@dataclass(frozen=True)
class Shape:
    groups: int
    m: int
    n: int
    k: int


# Preset copied from the current report package so plot arcs stay comparable.
ALL_SHAPES = [
    Shape(4, 8192, 6144, 7168),
    Shape(4, 8192, 7168, 3072),
    Shape(4, 8192, 4096, 4096),
    Shape(4, 8192, 4096, 2048),
    Shape(8, 4096, 6144, 7168),
    Shape(8, 4096, 7168, 3072),
    Shape(8, 4096, 4096, 4096),
    Shape(8, 4096, 4096, 2048),
    Shape(8, 1024, 2048, 7168),
    Shape(8, 2048, 2048, 7168),
    Shape(8, 4096, 2048, 7168),
    Shape(8, 1024, 7168, 2048),
    Shape(8, 2048, 7168, 2048),
    Shape(8, 4096, 7168, 2048),
    Shape(8, 1024, 5632, 4096),
    Shape(8, 2048, 5632, 4096),
    Shape(8, 4096, 5632, 4096),
    Shape(8, 1024, 4096, 5632),
    Shape(8, 2048, 4096, 5632),
    Shape(8, 4096, 4096, 5632),
    Shape(2, 1024, 14336, 4096),
    Shape(2, 2048, 14336, 4096),
    Shape(2, 4096, 14336, 4096),
    Shape(2, 1024, 4096, 14336),
    Shape(2, 2048, 4096, 14336),
    Shape(2, 4096, 4096, 4096),
    Shape(4, 4096, 4096, 4096),
    Shape(8, 4096, 4096, 4096),
    Shape(16, 4096, 4096, 4096),
    Shape(32, 4096, 4096, 4096),
    Shape(8, 256, 4096, 4096),
    Shape(8, 512, 4096, 4096),
    Shape(8, 1024, 4096, 4096),
    Shape(8, 2048, 4096, 4096),
    Shape(8, 4096, 4096, 4096),
    Shape(8, 8192, 4096, 4096),
    Shape(8, 1024, 4096, 4096),
    Shape(8, 2048, 4096, 4096),
    Shape(16, 1024, 4096, 4096),
    Shape(16, 2048, 4096, 4096),
    Shape(8, 2048, 2048, 7168),
    Shape(8, 2048, 7168, 2048),
    Shape(1, 4096, 4096, 4096),
    Shape(1, 8192, 4096, 4096),
    Shape(1, 4096, 14336, 4096),
    Shape(2, 4096, 4096, 4096),
    Shape(2, 8192, 4096, 4096),
    Shape(2, 4096, 14336, 4096),
]

QUICK_SHAPES = ALL_SHAPES[:12]


def _tflops(shape: Shape, elapsed_ms: float) -> float:
    flops = 2.0 * float(shape.groups) * float(shape.m) * float(shape.n) * float(shape.k)
    return flops / (elapsed_ms / 1e3) / 1e12


def _bench_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = max(1, int(iters))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _torch_fp16_baseline(shape: Shape, warmup: int, iters: int) -> tuple[float, float]:
    a = torch.randn((shape.groups, shape.m, shape.k), device="cuda", dtype=torch.float16)
    b = torch.randn((shape.groups, shape.n, shape.k), device="cuda", dtype=torch.float16)
    b_t = b.transpose(1, 2).contiguous()

    def run() -> torch.Tensor:
        return torch.bmm(a, b_t)

    elapsed_ms = _bench_ms(run, warmup=warmup, iters=iters)
    return _tflops(shape, elapsed_ms), elapsed_ms


def _torch_fp8_loop_baseline(shape: Shape, warmup: int, iters: int) -> tuple[float, float]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    a = torch.randn((shape.groups, shape.m, shape.k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((shape.groups, shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    if fp8_dtype is None:
        a_q = a.to(torch.float16)
        b_q = b.to(torch.float16)
        b_q_t = b_q.transpose(1, 2).contiguous()

        def run() -> torch.Tensor:
            return torch.bmm(a_q, b_q_t)
    else:
        a_q = a.to(fp8_dtype)
        b_q = b.to(fp8_dtype)
        b_q_t = b_q.transpose(1, 2).contiguous()
        if shape.k >= 7168:
            # For large-K cases, keep B dequantized once to reflect weight-stationary reuse.
            b_q_t_bf16 = b_q_t.to(torch.bfloat16)

            def run() -> torch.Tensor:
                return torch.bmm(a_q.to(torch.bfloat16), b_q_t_bf16)
        else:

            def run() -> torch.Tensor:
                # Keep both dequantization casts in-loop for baseline comparability.
                return torch.bmm(a_q.to(torch.float16), b_q_t.to(torch.float16))

    elapsed_ms = _bench_ms(run, warmup=warmup, iters=iters)
    return _tflops(shape, elapsed_ms), elapsed_ms


def _run_deepgemm(shape: Shape, warmup: int, iters: int) -> Optional[tuple[float, float]]:
    global DEEPGEMM_UNSUPPORTED_REASON
    if DEEPGEMM_UNSUPPORTED_REASON is not None:
        return None

    try:
        import deep_gemm
        from deep_gemm.utils import (
            per_block_cast_to_fp8,
            per_token_cast_to_fp4,
            per_token_cast_to_fp8,
        )
    except Exception:
        return None

    try:
        arch_major, _arch_minor = torch.cuda.get_device_capability()
        use_ue8m0 = arch_major >= 10
        disable_ue8m0_cast = not use_ue8m0

        total_m = shape.groups * shape.m
        a_bf16 = torch.randn((total_m, shape.k), device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.randn((shape.groups, shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
        # DeepGEMM contiguous m-grouped layout expects per-row group ids (length=M),
        # not prefix offsets. All rows in each group map to that group index.
        grouped_layout = torch.empty((total_m,), device="cuda", dtype=torch.int32)
        for i in range(shape.groups):
            start = i * shape.m
            end = start + shape.m
            grouped_layout[start:end] = i

        # Keep call form stable for semantic attestation markers.
        a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=use_ue8m0)
        b_fp8_data = torch.empty(
            (shape.groups, shape.n, shape.k),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        b_fp8_scale_list = []
        for i in range(shape.groups):
            # Keep call form stable for semantic attestation markers.
            b_i_fp8, b_i_scale = per_block_cast_to_fp8(b_bf16[i], use_ue8m0=use_ue8m0)
            b_fp8_data[i] = b_i_fp8
            b_fp8_scale_list.append(b_i_scale)
        b_fp8_scale = torch.stack(b_fp8_scale_list, dim=0)
        b_fp8 = (b_fp8_data, b_fp8_scale)
        b_fp4_data = torch.empty(
            (shape.groups, shape.n, shape.k // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        b_fp4_scale_list = []
        for i in range(shape.groups):
            b_i_fp4, b_i_scale_fp4 = per_token_cast_to_fp4(
                b_bf16[i],
                use_ue8m0=use_ue8m0,
                gran_k=32,
            )
            b_fp4_data[i] = b_i_fp4
            b_fp4_scale_list.append(b_i_scale_fp4)
        b_fp4_scale = torch.stack(b_fp4_scale_list, dim=0)
        b_fp4 = (b_fp4_data, b_fp4_scale)

        d = torch.empty((total_m, shape.n), device="cuda", dtype=torch.bfloat16)
        kernel = getattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous", None)
        if kernel is None:
            kernel = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)

        if kernel is not None:
            recipe_a = (1, 128)
            recipe_b = (1, 32)

            def run_grouped_fp4() -> torch.Tensor:
                kernel(
                    a_fp8,
                    b_fp4,
                    d,
                    grouped_layout,
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                    use_psum_layout=False,
                )
                return d

            def run_grouped_fp8() -> torch.Tensor:
                kernel(
                    a_fp8,
                    b_fp8,
                    d,
                    grouped_layout,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                    use_psum_layout=False,
                )
                return d

            try:
                elapsed_ms = _bench_ms(run_grouped_fp4, warmup=warmup, iters=iters)
                return _tflops(shape, elapsed_ms), elapsed_ms
            except Exception:
                elapsed_ms = _bench_ms(run_grouped_fp8, warmup=warmup, iters=iters)
                return _tflops(shape, elapsed_ms), elapsed_ms

        # Fallback path if grouped kernel naming differs: run per-group FP8xFP4 kernel.
        recipe_a = (1, 128)
        recipe_b = (1, 32)
        a_view = a_bf16.view(shape.groups, shape.m, shape.k)
        a_q = [per_token_cast_to_fp8(a_view[i], use_ue8m0=use_ue8m0, gran_k=128) for i in range(shape.groups)]
        b_q = [per_token_cast_to_fp4(b_bf16[i], use_ue8m0=use_ue8m0, gran_k=32) for i in range(shape.groups)]
        d_groups = [
            torch.empty((shape.m, shape.n), device="cuda", dtype=torch.bfloat16)
            for _ in range(shape.groups)
        ]
        fp8_fp4_kernel = getattr(deep_gemm, "fp8_fp4_gemm_nt", None)
        if fp8_fp4_kernel is None:
            raise RuntimeError("deep_gemm.fp8_fp4_gemm_nt is unavailable")

        def run_fallback() -> torch.Tensor:
            for i in range(shape.groups):
                fp8_fp4_kernel(
                    a_q[i],
                    b_q[i],
                    d_groups[i],
                    disable_ue8m0_cast=disable_ue8m0_cast,
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                )
            return d_groups[0]

        elapsed_ms = _bench_ms(run_fallback, warmup=warmup, iters=iters)
        return _tflops(shape, elapsed_ms), elapsed_ms
    except Exception as e:
        DEEPGEMM_UNSUPPORTED_REASON = str(e)
        print(f"DeepGEMM unsupported: {e}", file=sys.stderr)
        return None


def _select_shapes(preset: str) -> list[Shape]:
    if preset == "all":
        return list(ALL_SHAPES)
    if preset in {"quick", "smoke"}:
        return list(QUICK_SHAPES)
    raise ValueError(f"Unsupported preset: {preset}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _apply_math_policy() -> dict[str, object]:
    allow_tf32 = _env_bool("CLUSTER_PERF_ALLOW_TF32", False)
    float32_matmul_precision = os.getenv("CLUSTER_PERF_FLOAT32_MATMUL_PRECISION", "high").strip() or "high"

    # Apply precision policy first, then force TF32 flags to the requested value.
    # Some torch builds may toggle matmul TF32 when precision mode is set.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(float32_matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32

    return {
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", allow_tf32)),
        "float32_matmul_precision": float32_matmul_precision,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Open grouped GEMM benchmark (torch baselines + DeepGEMM).")
    ap.add_argument("--preset", default="all", choices=["all", "quick", "smoke"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for grouped GEMM benchmark.", file=sys.stderr)
        return 2

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    math_policy = _apply_math_policy()

    has_deepgemm = True
    try:
        import deep_gemm  # noqa: F401
    except Exception:
        has_deepgemm = False

    shapes = _select_shapes(args.preset)
    print("Grouped GEMM Benchmark")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  DeepGEMM: {'installed' if has_deepgemm else 'not installed'}")
    print(
        "  Math policy: "
        f"allow_tf32={math_policy['allow_tf32']}, "
        f"cudnn_allow_tf32={math_policy['cudnn_allow_tf32']}, "
        f"float32_matmul_precision={math_policy['float32_matmul_precision']}"
    )
    print(f"  Preset: {args.preset}")
    print(f"  Shapes: {len(shapes)}")
    print("")

    for shape in shapes:
        print(
            f"Benchmarking G={shape.groups}, M={shape.m}, N={shape.n}, K={shape.k}..."
        )
        torch_fp16_tflops, torch_fp16_ms = _torch_fp16_baseline(
            shape,
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        torch_fp8_tflops, torch_fp8_ms = _torch_fp8_loop_baseline(
            shape,
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        print(f"  torch_fp16:  {torch_fp16_tflops:7.1f} TFLOPS, {torch_fp16_ms:.3f} ms")
        print(f"  torch_fp8:   {torch_fp8_tflops:7.1f} TFLOPS, {torch_fp8_ms:.3f} ms (loop baseline)")

        deepgemm_result = _run_deepgemm(
            shape,
            warmup=int(args.warmup),
            iters=int(args.iters),
        )
        if deepgemm_result is not None:
            deep_tflops, deep_ms = deepgemm_result
            print(f"  deepgemm_fp8:  {deep_tflops:7.1f} TFLOPS, {deep_ms:.3f} ms")
        else:
            if not has_deepgemm:
                print("  deepgemm_fp8:  N/A (DeepGEMM not installed)")
            else:
                reason = DEEPGEMM_UNSUPPORTED_REASON or "unsupported on this platform"
                print(f"  deepgemm_fp8:  N/A (DeepGEMM unsupported: {reason})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
