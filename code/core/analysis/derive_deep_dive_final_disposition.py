#!/usr/bin/env python3
"""Merge deep-dive rerun outcomes and weak-case actions into a final disposition ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target(chapter: str | None, example: str | None) -> str:
    return f"{chapter}:{example}"


def _root_cause_map(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        category = row.get("category")
        note = row.get("root_cause") or row.get("next_action")
        for target in row.get("targets", []):
            mapping[str(target)] = {
                "root_cause_category": category,
                "root_cause_note": note,
            }
    return mapping


def _failure_status(failure: Dict[str, Any]) -> Any:
    return failure.get("final_status", failure.get("latest_status"))


def _failure_note(failure: Dict[str, Any]) -> str | None:
    if "final_error" in failure and failure.get("final_error"):
        return str(failure["final_error"])
    issues = failure.get("latest_issues") or []
    if issues:
        rendered = []
        for issue in issues:
            kind = issue.get("kind") or "issue"
            detail = issue.get("detail") or ""
            rendered.append(f"{kind}:{detail}" if detail else str(kind))
        return "; ".join(rendered)
    return None


def _failure_summary(final_failure_ledger: Dict[str, Any]) -> Dict[str, Any]:
    summary = final_failure_ledger.get("summary")
    if isinstance(summary, dict):
        return summary
    return final_failure_ledger


def classify_rows(
    *,
    final_failure_ledger: Dict[str, Any],
    weak_actions: Dict[str, Any],
    weak_root_causes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    root_causes = _root_cause_map(weak_root_causes.get("rows", []))
    action_rows = list(weak_actions.get("rows", []))
    action_map = {
        _target(row.get("chapter"), row.get("example")): dict(row)
        for row in action_rows
    }
    failure_rows = list(final_failure_ledger.get("rows", []))
    failure_map = {
        _target(row.get("chapter"), row.get("example")): dict(row)
        for row in failure_rows
    }

    all_targets = sorted(set(action_map) | set(failure_map))
    rows: List[Dict[str, Any]] = []

    for target in all_targets:
        chapter, example = target.split(":", 1)
        failure = failure_map.get(target)
        action = action_map.get(target)
        cause = root_causes.get(target, {})

        row: Dict[str, Any] = {
            "target": target,
            "chapter": chapter,
            "example": example,
            "failure_status": None if not failure else _failure_status(failure),
            "resolved_failure": None if not failure else failure.get("resolved"),
            "best_speedup": None if not failure else failure.get("best_speedup"),
            "weak_bucket": None if not action else action.get("bucket"),
            "weak_action": None if not action else action.get("action"),
            "root_cause_category": cause.get("root_cause_category"),
            "root_cause_note": cause.get("root_cause_note"),
        }

        if failure and not failure.get("resolved"):
            row["disposition"] = "unresolved_failure_blocker"
            row["expectation_decision"] = "block"
            row["narrative_decision"] = "block"
            row["note"] = _failure_note(failure) or "Still unresolved after clean reruns."
        elif action:
            weak_action = action.get("action")
            if weak_action == "family_level_investigation_before_blessing":
                row["disposition"] = "hold_expectations_family_investigation"
                row["expectation_decision"] = "hold"
                row["narrative_decision"] = "reframe"
                row["note"] = cause.get("root_cause_note") or action.get("note")
            elif weak_action == "investigate_missing_optimized_win":
                row["disposition"] = "hold_expectations_hardware_or_capability_gated"
                row["expectation_decision"] = "hold"
                row["narrative_decision"] = "reframe"
                row["note"] = cause.get("root_cause_note") or action.get("note")
            elif weak_action == "hold_expectations_and_reframe_story":
                row["disposition"] = "hold_expectations_reframe_narrative"
                row["expectation_decision"] = "hold"
                row["narrative_decision"] = "reframe"
                row["note"] = action.get("note")
            elif weak_action == "qualify_as_small_or_contextual_win":
                row["disposition"] = "refresh_with_qualified_narrative"
                row["expectation_decision"] = "refresh"
                row["narrative_decision"] = "qualify"
                row["note"] = action.get("note")
            elif weak_action == "treat_as_non_speed_example":
                row["disposition"] = "evaluate_non_speed_goal"
                row["expectation_decision"] = "goal_specific"
                row["narrative_decision"] = "goal_specific"
                row["note"] = action.get("note")
            else:
                row["disposition"] = "manual_review"
                row["expectation_decision"] = "manual_review"
                row["narrative_decision"] = "manual_review"
                row["note"] = action.get("note") or "Unknown weak-case action."
        else:
            row["disposition"] = "refresh_expectations"
            row["expectation_decision"] = "refresh"
            row["narrative_decision"] = "keep"
            row["note"] = "Resolved on clean reruns and not flagged as a weak-case holdout."

        rows.append(row)

    rows.sort(key=lambda item: (item["chapter"], item["example"]))
    return rows


def _to_markdown(
    rows: List[Dict[str, Any]],
    *,
    original_run_id: str | None,
    failure_recheck_run_ids: List[str],
    weak_actions_run_id: str | None,
) -> str:
    disposition_counts = Counter(row["disposition"] for row in rows)
    lines = [
        "# Deep-Dive Final Disposition",
        "",
        f"- Original run: `{original_run_id or 'unknown'}`",
        f"- Failure rechecks: `{', '.join(failure_recheck_run_ids) or 'none'}`",
        f"- Weak-case source run: `{weak_actions_run_id or 'unknown'}`",
        f"- Total targets in disposition ledger: `{len(rows)}`",
        f"- Disposition counts: `{dict(disposition_counts)}`",
        "",
        "| Target | Disposition | Expectations | Narrative | Failure Status | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target']}` | `{row['disposition']}` | `{row['expectation_decision']}` | "
            f"`{row['narrative_decision']}` | `{row['failure_status'] or 'n/a'}` | {row['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    final_failure_ledger_json: Path,
    weak_actions_json: Path,
    weak_root_causes_json: Path,
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    final_failure_ledger = _load_json(final_failure_ledger_json)
    weak_actions = _load_json(weak_actions_json)
    weak_root_causes = _load_json(weak_root_causes_json)
    failure_summary = _failure_summary(final_failure_ledger)

    rows = classify_rows(
        final_failure_ledger=final_failure_ledger,
        weak_actions=weak_actions,
        weak_root_causes=weak_root_causes,
    )
    out_dir = output_dir or final_failure_ledger_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "original_run_id": failure_summary.get("original_run_id"),
        "failure_recheck_run_ids": failure_summary.get("recheck_run_ids", []),
        "weak_actions_run_id": weak_actions.get("run_id"),
        "total": len(rows),
        "failure_summary": {
            "total_original_failures": failure_summary.get("total_original_failures"),
            "resolved_count": failure_summary.get("resolved_count"),
            "unresolved_count": failure_summary.get("unresolved_count"),
        },
        "disposition_counts": dict(Counter(row["disposition"] for row in rows)),
        "rows": rows,
    }

    json_path = out_dir / "deep_dive_final_disposition.json"
    md_path = out_dir / "deep_dive_final_disposition.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(
        _to_markdown(
            rows,
            original_run_id=payload.get("original_run_id"),
            failure_recheck_run_ids=list(payload.get("failure_recheck_run_ids", [])),
            weak_actions_run_id=payload.get("weak_actions_run_id"),
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge deep-dive rerun and weak-case outcomes into a final disposition ledger.")
    parser.add_argument("--final-failure-ledger-json", type=Path, required=True, help="Path to final_failure_ledger.json")
    parser.add_argument("--weak-actions-json", type=Path, required=True, help="Path to deep_dive_weak_case_actions.json")
    parser.add_argument("--weak-root-causes-json", type=Path, required=True, help="Path to deep_dive_weak_case_root_causes.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to the final failure ledger directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(
        final_failure_ledger_json=args.final_failure_ledger_json,
        weak_actions_json=args.weak_actions_json,
        weak_root_causes_json=args.weak_root_causes_json,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
