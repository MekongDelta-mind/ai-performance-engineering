"""Enforce that harness-discoverable Python benchmarks expose get_benchmark().

The benchmark harness imports each baseline_/optimized_ Python module and calls
`get_benchmark()` to construct a BaseBenchmark instance. This test ensures the
symbol is present (defined or re-exported) for every discoverable benchmark
module, matching the harness' non-recursive discovery behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from core.discovery import discover_all_chapters  # noqa: E402


def _has_get_benchmark_symbol(path: Path) -> bool:
    """Return True if module defines or re-exports get_benchmark."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_benchmark":
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "get_benchmark" or alias.asname == "get_benchmark":
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "get_benchmark":
                    return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "get_benchmark":
                    return True
    return False


def test_all_discoverable_python_benchmarks_have_get_benchmark() -> None:
    missing: list[Path] = []
    for bench_dir in discover_all_chapters(REPO_ROOT):
        for path in list(bench_dir.glob("baseline_*.py")) + list(bench_dir.glob("optimized_*.py")):
            try:
                if not _has_get_benchmark_symbol(path):
                    missing.append(path)
            except SyntaxError:
                missing.append(path)

    if not missing:
        return

    lines = ["Missing get_benchmark() symbol in harness-discoverable benchmark modules:"]
    for path in sorted(missing):
        lines.append(f"  - {path.relative_to(REPO_ROOT)}")
    raise AssertionError("\n".join(lines))
