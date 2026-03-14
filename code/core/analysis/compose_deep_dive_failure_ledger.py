#!/usr/bin/env python3
"""Compose a final deep-dive failure ledger from an original run plus rechecks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_results(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id_from_results_path(path: Path) -> str:
    try:
        return path.resolve().parents[1].name
    except IndexError:
        return path.stem


def _classify_benchmark(benchmark: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, str]] = []
    best_speedup: Optional[float] = None

    for profiler, status in (benchmark.get("baseline_profiler_statuses") or {}).items():
        if status not in {"succeeded", "skipped"}:
            issues.append({"kind": "failed_profiler", "detail": f"baseline:{profiler}:{status}"})

    for optimization in benchmark.get("optimizations", []):
        opt_status = optimization.get("status")
        technique = optimization.get("technique") or optimization.get("file") or "<unknown>"
        if opt_status == "succeeded":
            speedup = optimization.get("speedup")
            if isinstance(speedup, (int, float)):
                best_speedup = max(float(speedup), best_speedup or float(speedup))
        elif opt_status == "failed_verification":
            issues.append({"kind": "failed_verification", "detail": f"{technique}:{opt_status}"})
        elif opt_status and opt_status.startswith("failed"):
            issues.append({"kind": "failed_error", "detail": f"{technique}:{opt_status}"})

        for profiler, status in (optimization.get("optimized_profiler_statuses") or {}).items():
            if status not in {"succeeded", "skipped"}:
                issues.append({"kind": "failed_profiler", "detail": f"{technique}:{profiler}:{status}"})

    if issues:
        kinds = [issue["kind"] for issue in issues]
        if "failed_verification" in kinds:
            overall = "failed_verification"
        elif "failed_profiler" in kinds:
            overall = "failed_profiler"
        else:
            overall = "failed_error"
    else:
        overall = "succeeded"

    return {
        "status": overall,
        "best_speedup": best_speedup,
        "issues": issues,
    }


def _extract_targets(results_path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_results(results_path)
    run_id = _run_id_from_results_path(results_path)
    extracted: Dict[str, Dict[str, Any]] = {}
    for chapter in payload.get("results", []):
        chapter_name = chapter.get("chapter") or "<unknown>"
        for benchmark in chapter.get("benchmarks", []):
            example = benchmark.get("example") or "<unknown>"
            target = f"{chapter_name}:{example}"
            summary = _classify_benchmark(benchmark)
            extracted[target] = {
                "target": target,
                "chapter": chapter_name,
                "example": example,
                "run_id": run_id,
                **summary,
            }
    return extracted


def compose_failure_ledger(
    *,
    original_results_json: Path,
    recheck_results_json: Iterable[Path],
) -> Dict[str, Any]:
    original_targets = _extract_targets(original_results_json)
    original_failures = {
        target: row
        for target, row in original_targets.items()
        if row["status"] != "succeeded"
    }

    latest_by_target = dict(original_failures)
    recheck_run_ids: List[str] = []
    for recheck_path in recheck_results_json:
        recheck_run_ids.append(_run_id_from_results_path(recheck_path))
        for target, row in _extract_targets(recheck_path).items():
            if target in latest_by_target:
                latest_by_target[target] = row

    rows: List[Dict[str, Any]] = []
    for target in sorted(original_failures):
        original = original_failures[target]
        latest = latest_by_target[target]
        rows.append(
            {
                "target": target,
                "chapter": original["chapter"],
                "example": original["example"],
                "original_status": original["status"],
                "original_issues": original["issues"],
                "latest_status": latest["status"],
                "latest_issues": latest["issues"],
                "latest_run_id": latest["run_id"],
                "best_speedup": latest.get("best_speedup"),
                "resolved": latest["status"] == "succeeded",
            }
        )

    summary = {
        "original_run_id": _run_id_from_results_path(original_results_json),
        "recheck_run_ids": recheck_run_ids,
        "total_original_failures": len(rows),
        "resolved_count": sum(1 for row in rows if row["resolved"]),
        "unresolved_count": sum(1 for row in rows if not row["resolved"]),
    }
    return {"summary": summary, "rows": rows}


def _markdown_for_ledger(ledger: Dict[str, Any]) -> str:
    summary = ledger["summary"]
    lines = [
        "# Final Deep-Dive Failure Ledger",
        "",
        f"- Original run: `{summary['original_run_id']}`",
        f"- Rechecks: `{', '.join(summary['recheck_run_ids'])}`",
        f"- Total original failures: `{summary['total_original_failures']}`",
        f"- Resolved: `{summary['resolved_count']}`",
        f"- Unresolved: `{summary['unresolved_count']}`",
        "",
        "| Target | Original | Latest | Latest Run | Best Speedup | Resolved |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in ledger["rows"]:
        speedup = row["best_speedup"]
        speedup_text = "" if speedup is None else f"{float(speedup):.3f}x"
        lines.append(
            f"| `{row['target']}` | `{row['original_status']}` | `{row['latest_status']}` | "
            f"`{row['latest_run_id']}` | `{speedup_text}` | `{row['resolved']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def write_failure_ledger(
    *,
    original_results_json: Path,
    recheck_results_json: Iterable[Path],
    output_dir: Path,
) -> Dict[str, Path]:
    ledger = compose_failure_ledger(
        original_results_json=original_results_json,
        recheck_results_json=recheck_results_json,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "final_failure_ledger.json"
    md_path = output_dir / "final_failure_ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_for_ledger(ledger), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a final deep-dive failure ledger from an original run plus rechecks.")
    parser.add_argument("--original-results-json", type=Path, required=True, help="Original deep-dive benchmark_test_results.json")
    parser.add_argument("--recheck-results-json", type=Path, action="append", required=True, help="Recheck benchmark_test_results.json (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for final_failure_ledger.{json,md}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_failure_ledger(
        original_results_json=args.original_results_json,
        recheck_results_json=args.recheck_results_json,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
