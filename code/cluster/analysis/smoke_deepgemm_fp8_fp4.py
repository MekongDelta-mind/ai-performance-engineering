#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _p50(values: List[float]) -> float:
    vals = sorted(values)
    return float(vals[len(vals) // 2])


def _p99(values: List[float]) -> float:
    vals = sorted(values)
    idx = max(0, int(0.99 * len(vals)) - 1)
    return float(vals[idx])


def _tflops(m: int, n: int, k: int, ms: float) -> float:
    # 2*M*N*K FLOPs, ms -> s.
    return (2.0 * float(m) * float(n) * float(k)) / (float(ms) / 1e3) / 1e12


def _bench_ms(fn, torch_mod, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch_mod.cuda.synchronize()

    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    out: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(float(start.elapsed_time(end)))
    return out


def _stats(values: List[float], m: int, n: int, k: int) -> Dict[str, float]:
    avg_ms = float(sum(values) / len(values))
    p50_ms = _p50(values)
    p99_ms = _p99(values)
    return {
        "avg_ms": avg_ms,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "avg_tflops": _tflops(m, n, k, avg_ms),
        "p50_tflops": _tflops(m, n, k, p50_ms),
        "p99_tflops": _tflops(m, n, k, p99_ms),
    }


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="DeepGEMM FP8xFP4 smoke/perf check for GB200-class systems."
    )
    ap.add_argument("--m", type=int, default=4096, help="M dimension (default: 4096)")
    ap.add_argument("--n", type=int, default=4096, help="N dimension (default: 4096)")
    ap.add_argument("--k", type=int, default=4096, help="K dimension (default: 4096)")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    ap.add_argument("--iters", type=int, default=30, help="Measured iterations (default: 30)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    ap.add_argument("--out-json", default="", help="Optional output JSON path.")
    args = ap.parse_args()

    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "fail",
        "shape": {"m": int(args.m), "n": int(args.n), "k": int(args.k)},
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "seed": int(args.seed),
    }

    try:
        import torch
        import deep_gemm
        from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4
    except Exception as e:
        payload["status"] = "import_fail"
        payload["error"] = f"{type(e).__name__}: {e}"
        _write_json(args.out_json, payload)
        print(f"IMPORT_FAIL: {type(e).__name__}: {e}")
        return 2

    if not torch.cuda.is_available():
        payload["status"] = "cuda_unavailable"
        _write_json(args.out_json, payload)
        print("CUDA_UNAVAILABLE")
        return 2

    major, minor = torch.cuda.get_device_capability()
    dev = torch.cuda.get_device_name(0)
    use_ue8m0 = bool(major >= 10)
    disable_ue8m0_cast = not use_ue8m0

    payload["device"] = {
        "name": dev,
        "capability": f"{major}.{minor}",
        "use_ue8m0": use_ue8m0,
    }

    # Upstream DeepGEMM FP8xFP4 path for SM100 uses recipe_a=(1,128), recipe_b=(1,32).
    gran_k_a = 128
    gran_k_b = 32
    recipe_a = (1, gran_k_a)
    recipe_b = (1, gran_k_b)
    payload["quantization"] = {
        "gran_k_a": gran_k_a,
        "gran_k_b": gran_k_b,
        "recipe_a": list(recipe_a),
        "recipe_b": list(recipe_b),
    }

    try:
        torch.manual_seed(int(args.seed))
        m, n, k = int(args.m), int(args.n), int(args.k)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        a_q = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0, gran_k=gran_k_a)
        b_q = per_token_cast_to_fp4(b, use_ue8m0=use_ue8m0, gran_k=gran_k_b)
        d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def run_deepgemm() -> None:
            deep_gemm.fp8_fp4_gemm_nt(
                a_q,
                b_q,
                d,
                disable_ue8m0_cast=disable_ue8m0_cast,
                recipe_a=recipe_a,
                recipe_b=recipe_b,
            )

        def run_torch_ref():
            return a @ b.t()

        # One correctness pass (reference computed in FP32 to avoid masking quantization error).
        run_deepgemm()
        torch.cuda.synchronize()
        ref = a.float() @ b.float().t()
        diff = (d.float() - ref).abs()
        max_abs = float(diff.max().item())
        ref_max = float(ref.abs().max().item())
        max_rel = float(max_abs / max(ref_max, 1e-6))

        deep_ms = _bench_ms(run_deepgemm, torch, warmup=int(args.warmup), iters=int(args.iters))
        ref_ms = _bench_ms(run_torch_ref, torch, warmup=int(args.warmup), iters=int(args.iters))
        deep_stats = _stats(deep_ms, m, n, k)
        ref_stats = _stats(ref_ms, m, n, k)
        speedup = float(ref_stats["avg_ms"] / deep_stats["avg_ms"]) if deep_stats["avg_ms"] > 0 else None

        payload["status"] = "ok"
        payload["correctness"] = {"max_abs_diff": max_abs, "max_rel_diff_vs_fp32_ref": max_rel}
        payload["results"] = {
            "deepgemm_fp8_fp4": deep_stats,
            "torch_bf16_baseline": ref_stats,
            "deepgemm_over_torch_bf16_speedup": speedup,
        }

        _write_json(args.out_json, payload)
        print(
            "FP8xFP4_OK:",
            f"m={m} n={n} k={k}",
            f"deepgemm_avg={deep_stats['avg_tflops']:.1f} TFLOPS",
            f"torch_bf16_avg={ref_stats['avg_tflops']:.1f} TFLOPS",
            f"speedup={speedup:.2f}x" if speedup is not None else "speedup=n/a",
            f"max_abs_diff={max_abs:.4f}",
            f"max_rel_diff={max_rel:.6f}",
        )
        return 0
    except Exception as e:
        payload["status"] = "kernel_fail"
        payload["error"] = f"{type(e).__name__}: {e}"
        _write_json(args.out_json, payload)
        print(f"FP8xFP4_FAIL: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
