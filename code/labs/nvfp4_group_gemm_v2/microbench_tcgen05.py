"""Microbenchmark for the v2 tcgen05 NVFP4 grouped GEMM kernel.

This is a development tool (not a harness-comparable baseline/optimized pair).
It lets us iterate on kernel performance without paying the harness overhead
of running both baseline+optimized variants and full verification on every edit.

The timing model matches GPU MODE's evaluation: run a loop of `inputs` calls and
report the per-call latency (total loop time / inputs).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.l2_cache_utils import create_l2_flush_buffer, flush_l2_cache
from labs.nvfp4_group_gemm_v2.custom_cuda_submission import (
    custom_kernel_v2_custom_cuda_tcgen05,
    prepare_v2_custom_cuda_tcgen05,
)
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_common import COMPETITION_CASES
from labs.nvfp4_group_gemm_v2.nvfp4_group_gemm_inputs import generate_input


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _to_blocked_scale(input_matrix: torch.Tensor) -> torch.Tensor:
    """Match GPU MODE's reference `to_blocked()` scale layout (flattened)."""
    if input_matrix.ndim != 2:
        raise ValueError("Expected 2D scale matrix [rows, cols]")
    rows, cols = input_matrix.shape

    n_row_blocks = _ceil_div(int(rows), 128)
    n_col_blocks = _ceil_div(int(cols), 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
        padded = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _verify_one_input(data_raw, data_prepared, *, rtol: float = 1e-3, atol: float = 1e-3) -> None:
    """Compare tcgen05 output vs torch._scaled_mm reference for a single input."""
    abc_tensors, sfasfb_tensors, _, problem_sizes = data_raw

    # Run tcgen05 kernel once (writes into c_ref tensors and returns them).
    out = custom_kernel_v2_custom_cuda_tcgen05(data_prepared)
    if out is None:
        raise RuntimeError("Kernel did not produce output")

    max_abs = 0.0
    max_rel = 0.0
    worst = None

    for group_idx, ((a_ref, b_ref, _c_ref), (sfa_ref, sfb_ref), (m, n, k, l), c_out) in enumerate(
        zip(abc_tensors, sfasfb_tensors, problem_sizes, out)
    ):
        if int(l) != 1:
            raise ValueError(f"microbench verify expects l=1, got l={l}")
        # Blocked scale factors (flattened) as expected by torch._scaled_mm.
        scale_a = _to_blocked_scale(sfa_ref[:, :, 0]).cuda()
        scale_b = _to_blocked_scale(sfb_ref[:, :, 0]).cuda()

        ref = torch._scaled_mm(
            a_ref[:, :, 0].view(torch.float4_e2m1fn_x2),
            b_ref[:, :, 0].transpose(0, 1).view(torch.float4_e2m1fn_x2),
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
        )

        got = c_out[:, :, 0]
        if got.shape != ref.shape:
            raise RuntimeError(f"Group {group_idx}: shape mismatch got={tuple(got.shape)} ref={tuple(ref.shape)}")

        diff = (got - ref).abs()
        abs_err = float(diff.max().item())
        denom = ref.abs().clamp_min(1e-6)
        rel_err = float((diff / denom).max().item())

        if abs_err > max_abs or rel_err > max_rel:
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            worst = (group_idx, int(m), int(n), int(k), abs_err, rel_err)

        ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
        if not ok:
            flat_idx = int(diff.reshape(-1).argmax().item())
            row = flat_idx // int(diff.size(1))
            col = flat_idx - row * int(diff.size(1))
            got_val = float(got[row, col].item())
            ref_val = float(ref[row, col].item())
            raise RuntimeError(
                f"Verification failed for group {group_idx} (m={m} n={n} k={k}): "
                f"max_abs={abs_err:.6e} max_rel={rel_err:.6e} (rtol={rtol} atol={atol}) "
                f"argmax=(row={row}, col={col}) got={got_val:.6e} ref={ref_val:.6e}"
            )

    if worst is not None:
        g, m, n, k, abs_err, rel_err = worst
        print(
            f"verify_ok: groups={len(out)} worst_group={g} (m={m} n={n} k={k}) "
            f"max_abs={abs_err:.6e} max_rel={rel_err:.6e} (rtol={rtol} atol={atol})"
        )


def _run_loop(data_list) -> None:
    out = None
    for data in data_list:
        out = custom_kernel_v2_custom_cuda_tcgen05(data)
    # Keep one reference so Python can't trivially DCE the call chain.
    if out is None:
        raise RuntimeError("Kernel did not produce output")


def _capture_graph(data_list) -> torch.cuda.CUDAGraph:
    # Warmup once to ensure kernels and allocations are initialized before capture.
    _run_loop(data_list)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_loop(data_list)
    torch.cuda.synchronize()
    return graph


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=1, choices=range(4), help="Competition case index (0..3).")
    parser.add_argument("--inputs", type=int, default=15, help="Number of inputs (calls) per timing loop.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup loops before timing.")
    parser.add_argument("--repeats", type=int, default=50, help="Timing repeats (report stats across repeats).")
    parser.add_argument("--flush-l2", action="store_true", help="Flush L2 between repeats (recommended).")
    parser.add_argument("--use-graph", action="store_true", help="Capture a CUDA graph for the loop and replay.")
    parser.add_argument("--verify", action="store_true", help="Run correctness check vs torch._scaled_mm on the first input.")
    parser.add_argument(
        "--lock-gpu-clocks",
        action="store_true",
        help="Lock GPU clocks via core.harness.benchmark_harness.lock_gpu_clocks (recommended for stable numbers).",
    )
    args = parser.parse_args(argv)

    if args.inputs <= 0:
        raise ValueError("--inputs must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    case = COMPETITION_CASES[int(args.case)]
    base_seed = int(case.seed)

    data_list_raw = []
    for i in range(int(args.inputs)):
        seed = base_seed + 42 * i
        data_list_raw.append(
            generate_input(
                m=case.m,
                n=case.n,
                k=case.k,
                g=case.g,
                seed=seed,
            )
        )

    data_list = data_list_raw
    prepared = prepare_v2_custom_cuda_tcgen05(data_list)
    if prepared is not None:
        data_list = list(prepared)

    if args.verify:
        # Verify only the first input to keep turnaround fast.
        _verify_one_input(data_list_raw[0], data_list[0])
        torch.cuda.synchronize()

    clock_ctx = nullcontext()
    if args.lock_gpu_clocks:
        # Prefer the harness clock-locking mechanism over manual nvidia-smi calls.
        from core.harness.benchmark_harness import lock_gpu_clocks

        clock_ctx = lock_gpu_clocks(device=0)

    with clock_ctx:
        flush_buf = create_l2_flush_buffer() if args.flush_l2 else None

        graph: Optional[torch.cuda.CUDAGraph] = None
        if args.use_graph:
            graph = _capture_graph(data_list)

        for _ in range(int(args.warmup)):
            if graph is not None:
                graph.replay()
            else:
                _run_loop(data_list)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        per_call_us: list[float] = []

        for _ in range(int(args.repeats)):
            if flush_buf is not None:
                flush_l2_cache(buffer=flush_buf)

            start.record()
            if graph is not None:
                graph.replay()
            else:
                _run_loop(data_list)
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)
            per_call_us.append((elapsed_ms * 1000.0) / float(args.inputs))

    per_call_us_sorted = sorted(per_call_us)
    p50 = per_call_us_sorted[len(per_call_us_sorted) // 2]
    p99 = per_call_us_sorted[int(len(per_call_us_sorted) * 0.99) - 1] if len(per_call_us_sorted) >= 2 else p50

    mean = statistics.mean(per_call_us)
    stdev = statistics.pstdev(per_call_us) if len(per_call_us) > 1 else 0.0

    print(
        f"case={case.name} g={case.g} inputs={args.inputs} "
        f"use_graph={graph is not None} flush_l2={flush_buf is not None}"
    )
    print(f"per_call_us: mean={mean:.3f} p50={p50:.3f} p99={p99:.3f} min={min(per_call_us):.3f} max={max(per_call_us):.3f} stdev={stdev:.3f}")
    print(f"per_group_us: mean={(mean / case.g):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
