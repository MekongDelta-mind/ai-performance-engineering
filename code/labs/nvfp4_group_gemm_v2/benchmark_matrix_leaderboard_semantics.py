#!/usr/bin/env python3
"""Run baseline/candidate matrix with official leaderboard eval semantics.

This orchestrates the official reference-kernels evaluator in `leaderboard` mode:
- same warmup/repeat/stopping logic as official eval.py
- same per-repeat correctness recheck behavior
- same L2-flush + CUDA-event timing path

Per matrix cell (`graph`, `fuse`), it evaluates both baseline and candidate submissions
under identical env overrides and reports per-case + geomean speedups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_HARNESS_DIR = Path("/tmp/reference-kernels/problems/nvidia/nvfp4_group_gemm")
DEFAULT_CASES = ("case0", "case1", "case2", "case3")


@dataclass(frozen=True)
class MatrixCell:
    graph: int
    fuse: int


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def _normalize_case_token(token: str) -> str:
    raw = token.strip().lower()
    if raw == "":
        raise ValueError("Empty case token is not allowed")
    if raw.isdigit():
        raw = f"case{raw}"
    if raw not in DEFAULT_CASES:
        raise ValueError(f"Unsupported case token {token!r}; use case0..case3 or 0..3")
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


def _parse_env_assignments(items: list[str], *, arg_name: str) -> dict[str, str]:
    env_map: dict[str, str] = {}
    for item in items:
        token = str(item).strip()
        if token == "":
            continue
        if "=" not in token:
            raise ValueError(f"{arg_name} expects KEY=VALUE entries, got {item!r}")
        key, value = token.split("=", 1)
        key = key.strip()
        if key == "":
            raise ValueError(f"{arg_name} contains empty KEY in {item!r}")
        env_map[key] = value
    return env_map


def _safe_jit_namespace(run_prefix: str, variant: str) -> str:
    raw = f"{run_prefix}_{variant}"
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if safe == "":
        safe = "nvfp4_v2"
    if len(safe) <= 48:
        return safe
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:35]}_{digest}"


def _case_spec_line(case_name: str) -> str:
    if case_name == "case0":
        return "m:[80,176,128,72,64,248,96,160]; n:[4096,4096,4096,4096,4096,4096,4096,4096]; k:[7168,7168,7168,7168,7168,7168,7168,7168]; g:8; seed:1111"
    if case_name == "case1":
        return "m:[40,76,168,72,164,148,196,160]; n:[7168,7168,7168,7168,7168,7168,7168,7168]; k:[2048,2048,2048,2048,2048,2048,2048,2048]; g:8; seed:1111"
    if case_name == "case2":
        return "m:[192,320]; n:[3072,3072]; k:[4096,4096]; g:2; seed:1111"
    if case_name == "case3":
        return "m:[128,384]; n:[4096,4096]; k:[1536,1536]; g:2; seed:1111"
    raise ValueError(f"unsupported case: {case_name}")


def _write_tests_file(path: Path, cases: list[str]) -> None:
    lines = [_case_spec_line(case_name) for case_name in cases]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_eval_log(log_path: Path, expected_cases: list[str]) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    stats_by_idx: dict[int, dict[str, float]] = {}
    for line in lines:
        m = re.match(r"benchmark\.(\d+)\.(runs|mean|std|err|best|worst):\s+(.+)$", line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        key = m.group(2)
        try:
            value = float(m.group(3).strip())
        except ValueError:
            continue
        stats_by_idx.setdefault(idx, {})[key] = value

    rows: list[dict[str, Any]] = []
    for idx, case_name in enumerate(expected_cases):
        stats = stats_by_idx.get(idx, {})
        mean_ns = stats.get("mean")
        rows.append(
            {
                "case": case_name,
                "benchmark_index": idx,
                "runs": None if stats.get("runs") is None else int(stats["runs"]),
                "mean_ns": None if mean_ns is None else float(mean_ns),
                "mean_us": None if mean_ns is None else float(mean_ns) / 1_000.0,
                "std_ns": None if stats.get("std") is None else float(stats["std"]),
                "err_ns": None if stats.get("err") is None else float(stats["err"]),
                "best_ns": None if stats.get("best") is None else float(stats["best"]),
                "worst_ns": None if stats.get("worst") is None else float(stats["worst"]),
            }
        )

    check_pass = any(line.strip() == "check: pass" for line in lines)
    bad_fd_tail = any("Bad file descriptor" in line for line in lines)

    return {
        "rows": rows,
        "check_pass": check_pass,
        "bad_fd_tail": bad_fd_tail,
        "raw_tail": lines[-40:],
    }


def _geomean(values: list[float]) -> float | None:
    positives = [v for v in values if v > 0.0]
    if not positives:
        return None
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def _run_eval(
    *,
    harness_dir: Path,
    submission_file: Path,
    cases: list[str],
    graph: int,
    fuse: int,
    run_dir: Path,
    torch_extensions_dir: Path,
    jit_namespace: str,
    extra_env: dict[str, str],
    keep_workdir: bool,
) -> dict[str, Any]:
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    for name in ("eval.py", "reference.py", "task.py", "utils.py"):
        src = harness_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing official harness file: {src}")
        shutil.copy2(src, work_dir / name)

    shutil.copy2(submission_file, work_dir / "submission.py")
    tests_file = work_dir / "tests.txt"
    _write_tests_file(tests_file, cases)

    log_path = run_dir / "eval.log"
    runner_log_path = run_dir / "runner.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    # Isolate extension builds from unrelated local runs to avoid lock contention
    # and stale cross-job artifact reuse during matrix sweeps.
    torch_extensions_dir.mkdir(parents=True, exist_ok=True)
    env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir.resolve())

    # Apply both graph knobs so v2 submissions using either naming convention stay aligned.
    env["AISP_NVFP4_GROUP_GEMM_V2_CAPTURE_ITER_GRAPH"] = str(int(graph))
    env["AISP_NVFP4_GROUP_GEMM_V2_USE_CUDA_GRAPH"] = str(int(graph))
    env["AISP_NVFP4_GROUP_GEMM_V2_FUSE_INPUTS"] = str(int(fuse))
    # Keep JIT namespace stable across cells for each variant so we compile once per matrix run.
    env["AISP_NVFP4_GROUP_GEMM_V2_EXT_NAME"] = f"nvfp4_v2_{jit_namespace}"
    env["AISP_NVFP4_GROUP_GEMM_CUTLASS_EXT_SUFFIX"] = jit_namespace[-64:]
    for k, v in extra_env.items():
        env[str(k)] = str(v)

    cmd = [sys.executable, "-u", "eval.py", "leaderboard", "tests.txt"]
    with log_path.open("w", encoding="utf-8") as popcorn_log, runner_log_path.open("w", encoding="utf-8") as runner_log:
        # Official evaluator writes structured output to the inherited POPCORN_FD.
        popcorn_fd = int(popcorn_log.fileno())
        env["POPCORN_FD"] = str(popcorn_fd)
        proc = subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            pass_fds=(popcorn_fd,),
            check=False,
        )

    parsed = _parse_eval_log(log_path, expected_cases=cases)
    if not parsed["check_pass"]:
        raise RuntimeError(
            f"official eval check failed for {submission_file} graph={graph} fuse={fuse}. "
            f"See {log_path} (runner: {runner_log_path})"
        )

    if proc.returncode not in {0, 112}:
        raise RuntimeError(
            f"official eval returned {proc.returncode} for {submission_file} graph={graph} fuse={fuse}. "
            f"See {log_path} (runner: {runner_log_path})"
        )
    if proc.returncode == 112 and parsed["check_pass"]:
        raise RuntimeError(
            f"official eval returned 112 with check pass for {submission_file}. "
            f"See {log_path} (runner: {runner_log_path})"
        )

    if not keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "returncode": int(proc.returncode),
        "log_path": str(log_path),
        "runner_log_path": str(runner_log_path),
        "cases": parsed["rows"],
        "raw_tail": parsed["raw_tail"],
    }


def _build_speedup_rows(
    *,
    graph: int,
    fuse: int,
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    base_by_case = {str(r["case"]): r for r in baseline_result["cases"]}
    cand_by_case = {str(r["case"]): r for r in candidate_result["cases"]}
    case_names = sorted(set(base_by_case.keys()) | set(cand_by_case.keys()))

    for case_name in case_names:
        b = base_by_case.get(case_name)
        c = cand_by_case.get(case_name)
        b_mean_ns = None if b is None else b.get("mean_ns")
        c_mean_ns = None if c is None else c.get("mean_ns")

        speedup = None
        if isinstance(b_mean_ns, (int, float)) and isinstance(c_mean_ns, (int, float)) and float(c_mean_ns) > 0.0:
            speedup = float(b_mean_ns) / float(c_mean_ns)

        out.append(
            {
                "graph": int(graph),
                "fuse": int(fuse),
                "case": case_name,
                "baseline_mean_ns": None if b_mean_ns is None else float(b_mean_ns),
                "candidate_mean_ns": None if c_mean_ns is None else float(c_mean_ns),
                "baseline_mean_us": None if b_mean_ns is None else float(b_mean_ns) / 1_000.0,
                "candidate_mean_us": None if c_mean_ns is None else float(c_mean_ns) / 1_000.0,
                "speedup": speedup,
                "headroom_pct": None if speedup is None else (speedup - 1.0) * 100.0,
                "baseline_runs": None if b is None else b.get("runs"),
                "candidate_runs": None if c is None else c.get("runs"),
            }
        )

    return out


def _print_summary(rows: list[dict[str, Any]], cells: list[MatrixCell]) -> None:
    print()
    print("Matrix Summary (official leaderboard semantics)")
    print("graph fuse  geo_speedup  headroom_%  cases")
    for cell in cells:
        cell_rows = [r for r in rows if int(r["graph"]) == cell.graph and int(r["fuse"]) == cell.fuse]
        speedups = [float(r["speedup"]) for r in cell_rows if isinstance(r.get("speedup"), (int, float))]
        geo = _geomean(speedups)
        headroom = None if geo is None else (geo - 1.0) * 100.0
        geo_str = "n/a" if geo is None else f"{geo:>10.4f}"
        headroom_str = "n/a" if headroom is None else f"{headroom:>9.2f}"
        print(f"{cell.graph:>5} {cell.fuse:>4} {geo_str} {headroom_str} {len(cell_rows):>6}")

    print()
    print("Per-Case Headroom")
    print("graph fuse  case   baseline_us  candidate_us   speedup")
    for row in sorted(rows, key=lambda r: (int(r["graph"]), int(r["fuse"]), str(r["case"]))):
        b = row.get("baseline_mean_us")
        c = row.get("candidate_mean_us")
        s = row.get("speedup")
        b_str = "n/a" if b is None else f"{float(b):>11.3f}"
        c_str = "n/a" if c is None else f"{float(c):>12.3f}"
        s_str = "n/a" if s is None else f"{float(s):>8.4f}"
        print(f"{int(row['graph']):>5} {int(row['fuse']):>4}  {str(row['case']):<5} {b_str} {c_str} {s_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline/candidate v2 submission matrix using official leaderboard eval semantics."
        )
    )
    parser.add_argument(
        "--baseline-submission",
        default="labs/nvfp4_group_gemm_v2/popcorn_submission_best_8cb0.py",
        help="Baseline submission file path.",
    )
    parser.add_argument(
        "--candidate-submission",
        default="labs/nvfp4_group_gemm_v2/popcorn_submission_tuned_router.py",
        help="Candidate submission file path.",
    )
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        help="Baseline-only env override KEY=VALUE (repeatable).",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        help="Candidate-only env override KEY=VALUE (repeatable).",
    )
    parser.add_argument(
        "--official-harness-dir",
        default=str(DEFAULT_OFFICIAL_HARNESS_DIR),
        help="Path to official reference-kernels nvfp4_group_gemm harness dir containing eval.py/reference.py/task.py/utils.py.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(DEFAULT_CASES),
        help="Cases to include: case0 case1 case2 case3 (or 0 1 2 3). Default: all.",
    )
    parser.add_argument("--graph-values", default="0,1", help="Comma-separated graph values (0/1).")
    parser.add_argument("--fuse-values", default="0,1", help="Comma-separated fuse values (0/1).")
    parser.add_argument(
        "--run-prefix",
        default=f"{_now_stamp()}__nvfp4_v2_leaderboard_matrix",
        help="Run prefix for artifacts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/runs",
        help="Artifacts root directory. Default: artifacts/runs.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path. Default: <artifacts-dir>/<run-prefix>__summary.json",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep per-run copied harness/submission workdirs for debugging.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining cells if one cell fails.",
    )
    parser.add_argument(
        "--run-candidate-when-baseline-fails",
        action="store_true",
        help=(
            "Run candidate even when baseline fails correctness for a cell. "
            "Default: false (skip candidate because no apples-to-apples headroom is computable)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cases = _parse_cases(list(args.cases))
        graph_values = _parse_bool_int_list(args.graph_values, arg_name="--graph-values")
        fuse_values = _parse_bool_int_list(args.fuse_values, arg_name="--fuse-values")
        baseline_env = _parse_env_assignments(list(args.baseline_env), arg_name="--baseline-env")
        candidate_env = _parse_env_assignments(list(args.candidate_env), arg_name="--candidate-env")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    harness_dir = Path(args.official_harness_dir)
    if not harness_dir.exists():
        print(f"error: official harness dir not found: {harness_dir}", file=sys.stderr)
        return 2

    baseline_submission = Path(args.baseline_submission)
    candidate_submission = Path(args.candidate_submission)
    if not baseline_submission.is_absolute():
        baseline_submission = (REPO_ROOT / baseline_submission).resolve()
    if not candidate_submission.is_absolute():
        candidate_submission = (REPO_ROOT / candidate_submission).resolve()

    if not baseline_submission.exists():
        print(f"error: baseline submission not found: {baseline_submission}", file=sys.stderr)
        return 2
    if not candidate_submission.exists():
        print(f"error: candidate submission not found: {candidate_submission}", file=sys.stderr)
        return 2

    cells = [MatrixCell(graph=g, fuse=f) for g in graph_values for f in fuse_values]

    artifacts_dir = (REPO_ROOT / args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch_ext_root = artifacts_dir / f"{args.run_prefix}__torch_extensions"
    torch_ext_root.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    baseline_jit_namespace = _safe_jit_namespace(args.run_prefix, "baseline")
    candidate_jit_namespace = _safe_jit_namespace(args.run_prefix, "candidate")

    for cell in cells:
        baseline_failed = False
        baseline_record: dict[str, Any] | None = None
        candidate_record: dict[str, Any] | None = None

        # Baseline first.
        baseline_run_id = f"{args.run_prefix}__g{cell.graph}_f{cell.fuse}__baseline"
        baseline_run_dir = artifacts_dir / baseline_run_id
        baseline_run_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(
                f"[dry-run] run_id={baseline_run_id} graph={cell.graph} fuse={cell.fuse} "
                f"variant=baseline submission={baseline_submission}"
            )
        else:
            print(
                f"[run] graph={cell.graph} fuse={cell.fuse} variant=baseline "
                f"submission={baseline_submission.name}"
            )
            try:
                baseline_result = _run_eval(
                    harness_dir=harness_dir,
                    submission_file=baseline_submission,
                    cases=cases,
                    graph=cell.graph,
                    fuse=cell.fuse,
                    run_dir=baseline_run_dir,
                    torch_extensions_dir=torch_ext_root / "baseline",
                    jit_namespace=baseline_jit_namespace,
                    extra_env=baseline_env,
                    keep_workdir=bool(args.keep_workdir),
                )
                baseline_record = {
                    "graph": int(cell.graph),
                    "fuse": int(cell.fuse),
                    "variant": "baseline",
                    "submission_file": str(baseline_submission),
                    "run_id": baseline_run_id,
                    "run_dir": str(baseline_run_dir),
                    "returncode": int(baseline_result["returncode"]),
                    "log_path": str(baseline_result["log_path"]),
                    "runner_log_path": str(baseline_result["runner_log_path"]),
                    "cases": baseline_result["cases"],
                }
                run_records.append(baseline_record)
            except Exception as exc:
                baseline_failed = True
                err = {
                    "graph": int(cell.graph),
                    "fuse": int(cell.fuse),
                    "variant": "baseline",
                    "submission_file": str(baseline_submission),
                    "error": str(exc),
                }
                errors.append(err)
                print(f"[error] {err}", file=sys.stderr)
                if not args.keep_going:
                    break

        if errors and not args.keep_going:
            break

        # Candidate second; skip when baseline failed unless explicitly requested.
        if baseline_failed and not args.run_candidate_when_baseline_fails:
            skip_info = {
                "graph": int(cell.graph),
                "fuse": int(cell.fuse),
                "variant": "candidate",
                "submission_file": str(candidate_submission),
                "error": "skipped candidate because baseline failed correctness for this cell",
            }
            errors.append(skip_info)
            print(f"[skip] {skip_info}")
            continue

        candidate_run_id = f"{args.run_prefix}__g{cell.graph}_f{cell.fuse}__candidate"
        candidate_run_dir = artifacts_dir / candidate_run_id
        candidate_run_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            print(
                f"[dry-run] run_id={candidate_run_id} graph={cell.graph} fuse={cell.fuse} "
                f"variant=candidate submission={candidate_submission}"
            )
            continue

        print(
            f"[run] graph={cell.graph} fuse={cell.fuse} variant=candidate "
            f"submission={candidate_submission.name}"
        )
        try:
            candidate_result = _run_eval(
                harness_dir=harness_dir,
                submission_file=candidate_submission,
                cases=cases,
                graph=cell.graph,
                fuse=cell.fuse,
                run_dir=candidate_run_dir,
                torch_extensions_dir=torch_ext_root / "candidate",
                jit_namespace=candidate_jit_namespace,
                extra_env=candidate_env,
                keep_workdir=bool(args.keep_workdir),
            )
            candidate_record = {
                "graph": int(cell.graph),
                "fuse": int(cell.fuse),
                "variant": "candidate",
                "submission_file": str(candidate_submission),
                "run_id": candidate_run_id,
                "run_dir": str(candidate_run_dir),
                "returncode": int(candidate_result["returncode"]),
                "log_path": str(candidate_result["log_path"]),
                "runner_log_path": str(candidate_result["runner_log_path"]),
                "cases": candidate_result["cases"],
            }
            run_records.append(candidate_record)
        except Exception as exc:
            err = {
                "graph": int(cell.graph),
                "fuse": int(cell.fuse),
                "variant": "candidate",
                "submission_file": str(candidate_submission),
                "error": str(exc),
            }
            errors.append(err)
            print(f"[error] {err}", file=sys.stderr)
            if not args.keep_going:
                break

        if errors and not args.keep_going:
            break

        if baseline_record is None or candidate_record is None:
            if not args.keep_going:
                errors.append(
                    {
                        "graph": int(cell.graph),
                        "fuse": int(cell.fuse),
                        "error": "missing baseline or candidate record for cell",
                    }
                )
                break
            continue

        matrix_rows.extend(
            _build_speedup_rows(
                graph=cell.graph,
                fuse=cell.fuse,
                baseline_result=baseline_record,
                candidate_result=candidate_record,
            )
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "cases": cases,
        "cells": [asdict(c) for c in cells],
        "baseline_submission": str(baseline_submission),
        "candidate_submission": str(candidate_submission),
        "official_harness_dir": str(harness_dir),
        "run_prefix": args.run_prefix,
        "baseline_env": baseline_env,
        "candidate_env": candidate_env,
        "runs": run_records,
        "rows": matrix_rows,
        "errors": errors,
    }

    if args.output_json is None:
        output_json_path = artifacts_dir / f"{args.run_prefix}__summary.json"
    else:
        output_json_path = Path(args.output_json)
        if not output_json_path.is_absolute():
            output_json_path = (REPO_ROOT / output_json_path).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print(f"Wrote summary JSON: {output_json_path}")

    if args.dry_run:
        return 0

    if matrix_rows:
        _print_summary(matrix_rows, cells)

    if errors:
        print()
        print("Matrix completed with errors:")
        for err in errors:
            print(json.dumps(err, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
