#!/usr/bin/env python3
"""Direct official-semantics evaluator without multiprocessing pool IPC.

Implements the same benchmark loop semantics used by official eval.py leaderboard mode:
- warmup: run_single_benchmark(test0, recheck=False, max_repeats=100, max_time_ns=1e7)
- benchmark: per test run_single_benchmark(recheck=True, max_repeats=100, max_time_ns=30e9)
- stop criteria after at least 3 samples:
  - err/mean < 0.001 OR
  - mean * runs > max_time_ns OR
  - wallclock benchmark loop > 120s
- per-iteration L2 clear + CUDA-event timing
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch


@dataclasses.dataclass
class TestCase:
    args: dict[str, Any]
    spec: str


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def _combine(a: int, b: int) -> int:
    return int(a + (a + b) * (a + b + 1) // 2)


def get_test_cases(file_name: str, seed: Optional[int]) -> list[TestCase]:
    content = Path(file_name).read_text(encoding="utf-8")
    tests: list[TestCase] = []
    lines = content.splitlines()
    match = r"\s*([a-zA-Z_]+)\s*:\s*(\[[^\]]*\]|\([^)]*\)|[a-zA-Z_]+|[+-]?[0-9]+)\s*"
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(";")
        case: dict[str, Any] = {}
        for part in parts:
            if not part.strip():
                continue
            matched = re.fullmatch(match, part)
            if not matched:
                raise ValueError(f"invalid test case: '{line}': '{part}'")
            key = matched[1]
            val: Any = matched[2]
            try:
                val = int(val)
            except ValueError:
                if (val.startswith("(") and val.endswith(")")) or (val.startswith("[") and val.endswith("]")):
                    inner = val[1:-1].strip()
                    val = tuple(int(x.strip()) for x in inner.split(",")) if inner else tuple()
            case[key] = val
        tests.append(TestCase(spec=line, args=case))

    if seed is not None:
        for test in tests:
            if "seed" in test.args:
                test.args["seed"] = _combine(int(test.args["seed"]), seed)
    return tests


def _clone_data(data: Any) -> Any:
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    if isinstance(data, list):
        return [_clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return data


def calculate_stats(durations: list[float]) -> Stats:
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)
    avg = total / runs
    variance = sum((x - avg) ** 2 for x in durations)
    std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)
    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst))


def run_single_benchmark(
    *,
    test: TestCase,
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
    generate_input: Any,
    check_implementation: Any,
    custom_kernel: Any,
    clear_l2_cache: Any,
) -> Stats | str:
    durations: list[float] = []
    data = generate_input(**test.args)
    check_copy = _clone_data(data)

    output = custom_kernel(_clone_data(data))
    good, message = check_implementation(check_copy, output)
    if not good:
        return str(message)

    bm_start_time = time.perf_counter_ns()
    for i in range(max_repeats):
        if recheck:
            if "seed" in test.args:
                test.args["seed"] = int(test.args["seed"]) + 13
            data = generate_input(**test.args)
            check_copy = _clone_data(data)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        clear_l2_cache()

        start_event.record()
        output = custom_kernel(data)
        end_event.record()
        torch.cuda.synchronize()

        duration_ns = float(start_event.elapsed_time(end_event) * 1e6)
        if recheck:
            good, message = check_implementation(check_copy, output)
            if not good:
                return str(message)

        del output
        durations.append(duration_ns)

        if i > 1:
            total_bm_duration = time.perf_counter_ns() - bm_start_time
            stats = calculate_stats(durations)
            if (
                stats.err / stats.mean < 0.001
                or stats.mean * stats.runs > max_time_ns
                or total_bm_duration > 120e9
            ):
                break

    return calculate_stats(durations)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct official leaderboard-semantics evaluator")
    p.add_argument("--tests-file", default="tests.txt")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("main", flush=True)

    work_dir = Path.cwd().resolve()
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))

    tests = get_test_cases(args.tests_file, args.seed)
    if len(tests) == 0:
        print("benchmark-count: 0", flush=True)
        print("check: fail", flush=True)
        return 112

    # Local imports after cwd setup by caller; these files live beside this process cwd.
    from reference import check_implementation, generate_input  # type: ignore
    from submission import custom_kernel  # type: ignore
    from utils import clear_l2_cache  # type: ignore

    # Warmup (official leaderboard semantics).
    warmup = run_single_benchmark(
        test=tests[0],
        recheck=False,
        max_repeats=100,
        max_time_ns=1e7,
        generate_input=generate_input,
        check_implementation=check_implementation,
        custom_kernel=custom_kernel,
        clear_l2_cache=clear_l2_cache,
    )
    if isinstance(warmup, str):
        print(f"warmup.status: fail", flush=True)
        print(f"warmup.error: {warmup}", flush=True)
        print("check: fail", flush=True)
        return 112

    print(f"benchmark-count: {len(tests)}", flush=True)
    passed = True
    for i, test in enumerate(tests):
        result = run_single_benchmark(
            test=test,
            recheck=True,
            max_repeats=100,
            max_time_ns=30e9,
            generate_input=generate_input,
            check_implementation=check_implementation,
            custom_kernel=custom_kernel,
            clear_l2_cache=clear_l2_cache,
        )
        print(f"benchmark.{i}.spec: {test.spec}", flush=True)
        if isinstance(result, Stats):
            for field in dataclasses.fields(Stats):
                print(f"benchmark.{i}.{field.name}: {getattr(result, field.name)}", flush=True)
        else:
            passed = False
            print(f"benchmark.{i}.status: fail", flush=True)
            print(f"benchmark.{i}.error: {result}", flush=True)

    print(f"check: {'pass' if passed else 'fail'}", flush=True)
    return 0 if passed else 112


if __name__ == "__main__":
    raise SystemExit(main())
