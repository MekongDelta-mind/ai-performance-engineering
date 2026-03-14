"""Ensure baseline_/optimized_ benchmark naming is paired and clean.

This repository reserves `baseline_*` and `optimized_*` filenames for
harness-comparable benchmark variants.

This test suite enforces two invariants:
1) No generated LLM patch directories are present in the working tree
2) In harness-discoverable benchmark directories, every baseline/optimized file
   has a matching counterpart by prefix/stem rules.

Why this is important: the harness discovery is conservative — it only runs
benchmarks it can pair. Orphaned files silently become "dead" benchmarks (never
run or compared).

Pairing rule (per directory, per extension):
  - `baseline_<name>.<ext>` matches `optimized_<name>.<ext>` and any
    `optimized_<name>_*.<ext>` variants.
  - `optimized_<name>.<ext>` is considered matched if there exists any
    `baseline_<prefix>.<ext>` where `<name> == <prefix>` or `<name>` starts with
    `<prefix>_`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

from core.discovery import discover_all_chapters, should_ignore_benchmark_candidate

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep this aligned with `core.discovery._iter_benchmark_dirs()` so the test is fast
# and does not traverse huge artifact/cache trees.
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".torch_inductor",
    ".torch_extensions",
    ".next",
    ".turbo",
    "build",
    "dist",
    "out",
    "artifacts",
    "profiling_results",
    "hta_output",
    "gpt-oss-20b",
    "mixtral-8x7b",
    "phi-3.5-moe",
    "kernels",
    "third_party",
}

BENCHMARK_SUFFIXES = {".py"}
GENERATED_DIR_NAMES = {"llm_patches", "llm_patches_test"}

def _iter_named_dirs(repo_root: Path, names: set[str]) -> Iterable[Path]:
    """Yield directories with a name in `names`, pruning heavy/ignored trees."""
    for current, dirnames, filenames in os.walk(repo_root):
        pruned = []
        for d in dirnames:
            if (
                d in IGNORE_DIRS
                or d.startswith(".")
                or d.startswith("artifacts")
            ):
                continue
            if d in names:
                yield Path(current) / d
                continue  # Don't recurse into generated dirs
            pruned.append(d)
        dirnames[:] = pruned


def _matches_prefix(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix + "_")


def test_no_generated_llm_patch_dirs_present() -> None:
    """Fail if generated LLM patch dirs exist (they are gitignored, not committed)."""
    found = sorted(_iter_named_dirs(REPO_ROOT, GENERATED_DIR_NAMES))
    if not found:
        return
    lines: List[str] = []
    lines.append("Generated LLM patch directories found in working tree.")
    lines.append("These are outputs of `aisp bench run --llm-analysis --apply-llm-patches` and must be deleted.")
    lines.append("")
    for path in found:
        lines.append(f"  - {path.relative_to(REPO_ROOT)}")
    raise AssertionError("\n".join(lines))


def test_no_orphan_baseline_or_optimized_files_in_harness_dirs() -> None:
    """Fail if any harness-discoverable baseline_/optimized_ file is missing its counterpart."""
    orphan_baselines: List[Path] = []
    orphan_optimized: List[Path] = []

    # Limit to the same top-level directories the harness targets by default.
    # Within each directory, match only the immediate baseline_/optimized_ files,
    # consistent with the harness' non-recursive globbing.
    for bench_dir in discover_all_chapters(REPO_ROOT):
        for ext in BENCHMARK_SUFFIXES:
            groups: Dict[str, DefaultDict[str, List[Path]]] = {
                "baseline": defaultdict(list),
                "optimized": defaultdict(list),
            }

            for path in bench_dir.glob(f"baseline_*{ext}"):
                if should_ignore_benchmark_candidate(path):
                    continue
                key = path.stem.replace("baseline_", "", 1)
                groups["baseline"][key].append(path)

            for path in bench_dir.glob(f"optimized_*{ext}"):
                if should_ignore_benchmark_candidate(path):
                    continue
                key = path.stem.replace("optimized_", "", 1)
                groups["optimized"][key].append(path)

            baseline_keys = set(groups["baseline"].keys())
            optimized_keys = set(groups["optimized"].keys())

            for baseline_key in baseline_keys:
                if not any(_matches_prefix(opt_key, baseline_key) for opt_key in optimized_keys):
                    orphan_baselines.extend(groups["baseline"][baseline_key])

            for optimized_key in optimized_keys:
                if not any(_matches_prefix(optimized_key, baseline_key) for baseline_key in baseline_keys):
                    orphan_optimized.extend(groups["optimized"][optimized_key])

    if not orphan_baselines and not orphan_optimized:
        return

    lines: List[str] = []
    lines.append("Orphan baseline_/optimized_ files found in harness-discoverable benchmark directories.")
    lines.append("These names are reserved for paired benchmarks; orphans are silently skipped by discovery.")

    if orphan_baselines:
        lines.append("")
        lines.append(f"Orphan baselines ({len(orphan_baselines)}):")
        for path in sorted(orphan_baselines):
            lines.append(f"  - {path.relative_to(REPO_ROOT)}")

    if orphan_optimized:
        lines.append("")
        lines.append(f"Orphan optimized ({len(orphan_optimized)}):")
        for path in sorted(orphan_optimized):
            lines.append(f"  - {path.relative_to(REPO_ROOT)}")

    raise AssertionError("\n".join(lines))
