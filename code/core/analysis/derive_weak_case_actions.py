#!/usr/bin/env python3
"""Derive an auditable action plan for weak/flat deep-dive review candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


HIGH_PRIORITY_ACTIONS = {
    "family_level_investigation_before_blessing",
    "investigate_missing_optimized_win",
    "hold_expectations_and_reframe_story",
}


def _load_rows(path: Path) -> Tuple[str | None, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = list(data.get("rows", []))
        return data.get("run_id"), rows
    if isinstance(data, list):
        return None, list(data)
    raise TypeError(f"Unsupported review candidate payload type: {type(data)!r}")


def classify_weak_case_actions(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    family_counts = Counter(row.get("example") for row in materialized)
    classified: List[Dict[str, Any]] = []

    for row in materialized:
        goal = row.get("goal") or "speed"
        bucket = row.get("bucket") or "unknown"
        technique = row.get("technique") or ""
        best_opt_ms = row.get("best_opt_ms")
        family_count = family_counts.get(row.get("example"), 1)

        if goal != "speed":
            action = "treat_as_non_speed_example"
            note = "Do not score this example as a speed story; evaluate it against its declared non-speed goal."
        elif family_count >= 3:
            action = "family_level_investigation_before_blessing"
            note = (
                f"This weak pattern repeats across {family_count} chapters; investigate the shared benchmark family "
                "before blessing any instance."
            )
        elif best_opt_ms is None or not technique:
            action = "investigate_missing_optimized_win"
            note = "No credible optimized winner is recorded; verify whether the optimization path is ineffective or not actually landing."
        elif bucket == "flat_or_negative":
            action = "hold_expectations_and_reframe_story"
            note = "Do not refresh expectations upward or present this as a strong win; either improve the benchmark or demote the narrative."
        else:
            action = "qualify_as_small_or_contextual_win"
            note = "Keep the example, but describe it as a small or context-dependent win rather than a headline optimization."

        classified_row = dict(row)
        classified_row["action"] = action
        classified_row["priority"] = "high" if action in HIGH_PRIORITY_ACTIONS else "medium"
        classified_row["family_count"] = family_count
        classified_row["note"] = note
        classified.append(classified_row)

    classified.sort(key=lambda row: (row.get("priority") != "high", row.get("chapter") or "", row.get("example") or ""))
    return classified


def _rows_to_markdown(rows: List[Dict[str, Any]], *, run_id: str | None) -> str:
    bucket_counts = Counter(row.get("bucket") for row in rows)
    action_counts = Counter(row.get("action") for row in rows)
    lines = [
        "# Deep-Dive Weak Case Actions",
        "",
        f"Run ID: `{run_id or 'unknown'}`",
        "",
        "## Summary",
        "",
        f"- Total weak/review candidates: `{len(rows)}`",
        f"- Bucket counts: `{dict(bucket_counts)}`",
        f"- Action counts: `{dict(action_counts)}`",
        "",
        "## Policy",
        "",
        "- Weak cases are not auto-blessed.",
        "- Expectation refreshes stay on hold for `family_level_investigation_before_blessing`, `investigate_missing_optimized_win`, and `hold_expectations_and_reframe_story`.",
        "- `qualify_as_small_or_contextual_win` cases may remain in the repo, but should not be narrated as headline optimization wins.",
        "- `treat_as_non_speed_example` cases should be evaluated against their declared non-speed goal.",
        "",
        "## Rows",
        "",
        "| Target | Bucket | Action | Priority | Note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        target = f"{row.get('chapter')}:{row.get('example')}"
        lines.append(
            f"| `{target}` | `{row.get('bucket')}` | `{row.get('action')}` | `{row.get('priority')}` | {row.get('note')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(*, review_candidates_json: Path, output_dir: Path | None = None) -> Dict[str, Path]:
    run_id, rows = _load_rows(review_candidates_json)
    classified = classify_weak_case_actions(rows)
    out_dir = output_dir or review_candidates_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "total": len(classified),
        "bucket_counts": dict(Counter(row.get("bucket") for row in classified)),
        "action_counts": dict(Counter(row.get("action") for row in classified)),
        "rows": classified,
    }
    json_path = out_dir / "deep_dive_weak_case_actions.json"
    md_path = out_dir / "deep_dive_weak_case_actions.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_rows_to_markdown(classified, run_id=run_id), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive action buckets for weak deep-dive review candidates.")
    parser.add_argument("--review-candidates-json", type=Path, required=True, help="Path to deep_dive_review_candidates_refined.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to the input parent directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(
        review_candidates_json=args.review_candidates_json,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
