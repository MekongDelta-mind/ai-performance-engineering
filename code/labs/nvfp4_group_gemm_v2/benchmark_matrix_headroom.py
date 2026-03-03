#!/usr/bin/env python3
"""Run an apples-to-apples graph/fuse matrix for v2 baseline vs optimized wrappers.

This script forces the same runtime settings for both baseline and optimized variants:
  - AISP_NVFP4_GROUP_GEMM_V2_CAPTURE_ITER_GRAPH
  - AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS

It executes one bench run per matrix cell and summarizes baseline/optimized speedups so
kernel headroom can be quantified without graph/fuse confounders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ("case0", "case1", "case2", "case3")


@dataclass(frozen=True)
class MatrixCell:
    graph: int
    fuse: int


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_case_token(token: str) -> str:
    raw = token.strip().lower()
    if raw == "":
        raise ValueError("Empty case token is not allowed")
    if raw.isdigit():
        raw = f"case{raw}"
    if not raw.startswith("case"):
        raise ValueError(f"Unsupported case token {token!r}; use case0..case3 or 0..3")
    if raw not in DEFAULT_CASES:
        raise ValueError(f"Unsupported case token {token!r}; use case0..case3")
    return raw


def _parse_cases(tokens: list[str]) -> list[str]:
    if not tokens:
        return list(DEFAULT_CASES)
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        case_name = _normalize_case_token(tok)
        if case_name in seen:
            continue
        seen.add(case_name)
        out.append(case_name)
    return out


def _parse_bool_int_list(raw: str, *, arg_name: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if item == "":
            continue
        if item not in {"0", "1"}:
            raise ValueError(f"{arg_name} must contain only 0/1 values; got {item!r}")
        val = int(item)
        if val in seen:
            continue
        seen.add(val)
        values.append(val)
    if not values:
        raise ValueError(f"{arg_name} must provide at least one value")
    return values


def _geomean(values: list[float]) -> float | None:
    positives = [v for v in values if v > 0.0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def _extract_case(example_name: str) -> str:
    match = re.search(r"case\d+", example_name)
    if match:
        return match.group(0)
    return example_name


def _target_for_case(case_name: str) -> str:
    return f"labs/nvfp4_group_gemm_v2:nvfp4_group_gemm_{case_name}"


def _parse_run_results(results_path: Path, *, cell: MatrixCell) -> list[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = payload.get("results", [])
    rows: list[dict[str, Any]] = []
    for entry in entries:
        for bench in entry.get("benchmarks", []):
            example = str(bench.get("example", ""))
            case_name = _extract_case(example)
            baseline_ms = _to_float(bench.get("baseline_time_ms"))

            best_opt: dict[str, Any] | None = None
            for opt in bench.get("optimizations", []):
                if str(opt.get("status", "")).lower() != "succeeded":
                    continue
                opt_ms = _to_float(opt.get("time_ms"))
                speedup = _to_float(opt.get("speedup"))
                if speedup is None and baseline_ms and opt_ms and opt_ms > 0.0:
                    speedup = baseline_ms / opt_ms
                if best_opt is None:
                    best_opt = {"opt": opt, "opt_ms": opt_ms, "speedup": speedup}
                    continue
                best_speedup = _to_float(best_opt.get("speedup")) or 0.0
                cur_speedup = speedup or 0.0
                if cur_speedup > best_speedup:
                    best_opt = {"opt": opt, "opt_ms": opt_ms, "speedup": speedup}

            opt_ms = None if best_opt is None else _to_float(best_opt.get("opt_ms"))
            speedup = None if best_opt is None else _to_float(best_opt.get("speedup"))

            rows.append(
                {
                    "graph": int(cell.graph),
                    "fuse": int(cell.fuse),
                    "chapter": entry.get("chapter"),
                    "example": example,
                    "case": case_name,
                    "baseline_ms": baseline_ms,
                    "optimized_ms": opt_ms,
                    "speedup": speedup,
                    "headroom_pct": None if speedup is None else (speedup - 1.0) * 100.0,
                    "status": bench.get("status"),
                    "results_path": str(results_path),
                }
            )
    return rows


def _build_run_command(
    *,
    targets: list[str],
    run_id: str,
    artifacts_dir: Path,
    profile: str,
    validity_profile: str,
    iterations: int | None,
    warmup: int | None,
    timeout_seconds: int | None,
    gpu_sm_clock_mhz: int | None,
    gpu_mem_clock_mhz: int | None,
    verify: bool,
    force_sync: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cli.aisp",
        "bench",
        "run",
        "--format",
        "json",
        "--profile",
        profile,
        "--validity-profile",
        validity_profile,
        "--artifacts-dir",
        str(artifacts_dir),
        "--run-id",
        run_id,
    ]
    for target in targets:
        cmd.extend(["--targets", target])
    if iterations is not None:
        cmd.extend(["--iterations", str(iterations)])
    if warmup is not None:
        cmd.extend(["--warmup", str(warmup)])
    if timeout_seconds is not None:
        cmd.extend(["--timeout-seconds", str(timeout_seconds)])
    if gpu_sm_clock_mhz is not None:
        cmd.extend(["--gpu-sm-clock-mhz", str(gpu_sm_clock_mhz)])
    if gpu_mem_clock_mhz is not None:
        cmd.extend(["--gpu-mem-clock-mhz", str(gpu_mem_clock_mhz)])
    if not verify:
        cmd.append("--skip-verify")
    if force_sync:
        cmd.append("--force-sync")
    return cmd


def _print_matrix_summary(rows: list[dict[str, Any]], cells: list[MatrixCell], run_ids: dict[str, str]) -> None:
    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((int(row["graph"]), int(row["fuse"])), []).append(row)

    print()
    print("Matrix Summary (same graph/fuse in baseline and candidate)")
    print("graph fuse  geo_speedup  headroom_%  cases  run_id")
    for cell in cells:
        key = (cell.graph, cell.fuse)
        cell_rows = by_cell.get(key, [])
        speedups = [r["speedup"] for r in cell_rows if isinstance(r.get("speedup"), (int, float))]
        speedups = [float(v) for v in speedups]
        geo = _geomean(speedups)
        headroom = None if geo is None else (geo - 1.0) * 100.0
        geo_str = "n/a" if geo is None else f"{geo:>10.4f}"
        headroom_str = "n/a" if headroom is None else f"{headroom:>9.2f}"
        print(
            f"{cell.graph:>5} {cell.fuse:>4} {geo_str} {headroom_str} {len(cell_rows):>6}  {run_ids[(cell.graph, cell.fuse)]}"
        )

    print()
    print("Per-Case Rows")
    print("graph fuse  case   baseline_ms  optimized_ms   speedup")
    for row in sorted(rows, key=lambda r: (int(r["graph"]), int(r["fuse"]), str(r["case"]))):
        baseline = row.get("baseline_ms")
        optimized = row.get("optimized_ms")
        speedup = row.get("speedup")
        b = "n/a" if baseline is None else f"{float(baseline):>11.6f}"
        o = "n/a" if optimized is None else f"{float(optimized):>12.6f}"
        s = "n/a" if speedup is None else f"{float(speedup):>8.4f}"
        print(f"{int(row['graph']):>5} {int(row['fuse']):>4}  {str(row['case']):<5} {b} {o} {s}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apples-to-apples matrix runner for labs/nvfp4_group_gemm_v2 baseline vs optimized "
            "benchmarks with explicit graph/fuse parity."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Cases to run: case0 case1 case2 case3 (or 0 1 2 3). Default: all.",
    )
    parser.add_argument(
        "--graph-values",
        default="0,1",
        help="Comma-separated capture-graph values (0/1). Default: 0,1.",
    )
    parser.add_argument(
        "--fuse-values",
        default="0,1",
        help="Comma-separated fuse-input values (0/1). Default: 0,1.",
    )
    parser.add_argument("--profile", default="none", help="bench run --profile value. Default: none.")
    parser.add_argument(
        "--validity-profile",
        default="strict",
        choices=("strict", "portable"),
        help="bench run validity profile. Default: strict.",
    )
    parser.add_argument("--iterations", type=int, default=None, help="Optional benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=None, help="Optional warmup iterations.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Optional suite timeout override.")
    parser.add_argument("--gpu-sm-clock-mhz", type=int, default=None, help="Optional SM app clock.")
    parser.add_argument("--gpu-mem-clock-mhz", type=int, default=None, help="Optional memory app clock.")
    parser.add_argument(
        "--timing-method",
        choices=("cuda_event", "wall_clock"),
        default="wall_clock",
        help=(
            "Benchmark timing method override for v2 wrappers via "
            "AISP_NVFP4_GROUP_GEMM_V2_TIMING_METHOD. Default: wall_clock."
        ),
    )
    parser.add_argument(
        "--cross-validate-timing",
        choices=("0", "1"),
        default="0",
        help=(
            "Benchmark timing cross-validation override for v2 wrappers via "
            "AISP_NVFP4_GROUP_GEMM_V2_CROSS_VALIDATE_TIMING. Default: 0."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        default=f"{_now_stamp()}__nvfp4_v2_a2a_matrix",
        help="Run-id prefix; each matrix cell appends __g<g>_f<f>.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/runs",
        help="Artifacts root directory. Default: artifacts/runs.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for matrix summary JSON. Default: <artifacts-dir>/<run-prefix>__summary.json.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable input/output verification (not recommended for promotion decisions).",
    )
    parser.add_argument(
        "--no-force-sync",
        action="store_true",
        help="Disable --force-sync in bench runs (default is enabled for timing stability).",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining cells if one cell fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands/env overrides and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cases = _parse_cases(list(args.cases))
        graph_values = _parse_bool_int_list(args.graph_values, arg_name="--graph-values")
        fuse_values = _parse_bool_int_list(args.fuse_values, arg_name="--fuse-values")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    targets = [_target_for_case(case_name) for case_name in cases]
    cells = [MatrixCell(graph=g, fuse=f) for g in graph_values for f in fuse_values]
    artifacts_dir = (REPO_ROOT / args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    cell_errors: list[dict[str, Any]] = []
    run_ids: dict[tuple[int, int], str] = {}
    commands: list[dict[str, Any]] = []

    for cell in cells:
        run_id = f"{args.run_prefix}__g{cell.graph}_f{cell.fuse}"
        run_ids[(cell.graph, cell.fuse)] = run_id

        cmd = _build_run_command(
            targets=targets,
            run_id=run_id,
            artifacts_dir=artifacts_dir,
            profile=args.profile,
            validity_profile=args.validity_profile,
            iterations=args.iterations,
            warmup=args.warmup,
            timeout_seconds=args.timeout_seconds,
            gpu_sm_clock_mhz=args.gpu_sm_clock_mhz,
            gpu_mem_clock_mhz=args.gpu_mem_clock_mhz,
            verify=not args.no_verify,
            force_sync=not args.no_force_sync,
        )
        env = os.environ.copy()
        env["AISP_NVFP4_GROUP_GEMM_V2_CAPTURE_ITER_GRAPH"] = str(cell.graph)
        env["AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS"] = str(cell.fuse)
        env["AISP_NVFP4_GROUP_GEMM_V2_TIMING_METHOD"] = str(args.timing_method)
        env["AISP_NVFP4_GROUP_GEMM_V2_CROSS_VALIDATE_TIMING"] = str(args.cross_validate_timing)

        commands.append(
            {
                "graph": cell.graph,
                "fuse": cell.fuse,
                "run_id": run_id,
                "command": cmd,
                "overrides": {
                    "AISP_NVFP4_GROUP_GEMM_V2_CAPTURE_ITER_GRAPH": str(cell.graph),
                    "AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS": str(cell.fuse),
                    "AISP_NVFP4_GROUP_GEMM_V2_TIMING_METHOD": str(args.timing_method),
                    "AISP_NVFP4_GROUP_GEMM_V2_CROSS_VALIDATE_TIMING": str(args.cross_validate_timing),
                },
            }
        )

        if args.dry_run:
            print()
            print(f"[dry-run] cell graph={cell.graph} fuse={cell.fuse} run_id={run_id}")
            print(
                "  env overrides: "
                f"AISP_NVFP4_GROUP_GEMM_V2_CAPTURE_ITER_GRAPH={cell.graph} "
                f"AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS={cell.fuse} "
                f"AISP_NVFP4_GROUP_GEMM_V2_TIMING_METHOD={args.timing_method} "
                f"AISP_NVFP4_GROUP_GEMM_V2_CROSS_VALIDATE_TIMING={args.cross_validate_timing}"
            )
            print(f"  cmd: {' '.join(cmd)}")
            continue

        print()
        print(f"[run] graph={cell.graph} fuse={cell.fuse} run_id={run_id}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        if proc.returncode != 0:
            error = {
                "graph": cell.graph,
                "fuse": cell.fuse,
                "run_id": run_id,
                "returncode": int(proc.returncode),
                "error": "bench run failed",
            }
            cell_errors.append(error)
            print(
                f"[error] graph={cell.graph} fuse={cell.fuse} run_id={run_id} returncode={proc.returncode}",
                file=sys.stderr,
            )
            if not args.keep_going:
                break
            continue

        results_path = artifacts_dir / run_id / "results" / "benchmark_test_results.json"
        if not results_path.exists():
            error = {
                "graph": cell.graph,
                "fuse": cell.fuse,
                "run_id": run_id,
                "returncode": 0,
                "error": f"missing results file: {results_path}",
            }
            cell_errors.append(error)
            print(f"[error] {error['error']}", file=sys.stderr)
            if not args.keep_going:
                break
            continue

        try:
            rows = _parse_run_results(results_path, cell=cell)
        except Exception as exc:  # pragma: no cover - defensive parser guard
            error = {
                "graph": cell.graph,
                "fuse": cell.fuse,
                "run_id": run_id,
                "returncode": 0,
                "error": f"failed to parse results: {exc}",
            }
            cell_errors.append(error)
            print(f"[error] {error['error']}", file=sys.stderr)
            if not args.keep_going:
                break
            continue

        all_rows.extend(rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "cases": cases,
        "targets": targets,
        "cells": [asdict(c) for c in cells],
        "run_prefix": args.run_prefix,
        "run_ids": {f"g{g}_f{f}": run_id for (g, f), run_id in run_ids.items()},
        "commands": commands,
        "rows": all_rows,
        "errors": cell_errors,
    }

    if args.output_json is None:
        output_json_path = artifacts_dir / f"{args.run_prefix}__summary.json"
    else:
        output_json_path = Path(args.output_json)
        if not output_json_path.is_absolute():
            output_json_path = (REPO_ROOT / output_json_path).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print()
    print(f"Wrote summary JSON: {output_json_path}")

    if args.dry_run:
        return 0

    if all_rows:
        _print_matrix_summary(all_rows, cells, run_ids)
    else:
        print("No benchmark rows parsed.", file=sys.stderr)

    if cell_errors:
        print()
        print("Matrix completed with errors:")
        for err in cell_errors:
            print(
                f"  graph={err['graph']} fuse={err['fuse']} run_id={err['run_id']} "
                f"returncode={err['returncode']} error={err['error']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
