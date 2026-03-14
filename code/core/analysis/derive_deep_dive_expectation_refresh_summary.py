#!/usr/bin/env python3
"""Derive a deep-dive expectation refresh summary from disposition and apply ledgers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


APPLIED_STATUSES = {"updated", "improved", "regressed", "unchanged"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(
    *,
    final_disposition: Dict[str, Any],
    apply_summaries: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    approved_targets = {
        str(row.get("target"))
        for row in final_disposition.get("rows", [])
        if row.get("expectation_decision") == "refresh"
    }

    latest_record_by_target: Dict[str, Dict[str, Any]] = {}
    source_runs: List[Dict[str, Any]] = []
    for summary in apply_summaries:
        source_runs.append(
            {
                "source_results_json": summary.get("results_json"),
                "counts": summary.get("counts", {}),
                "updated_files": summary.get("updated_files", []),
            }
        )
        for record in summary.get("records", []):
            target = str(record.get("target"))
            status = str(record.get("status"))
            if target in approved_targets and status in APPLIED_STATUSES:
                latest_record_by_target[target] = {
                    "target": target,
                    "expectation_file": record.get("expectation_file"),
                    "status": status,
                    "message": record.get("message"),
                }

    applied_records = [latest_record_by_target[target] for target in sorted(latest_record_by_target)]
    applied_counts = dict(Counter(str(record.get("status")) for record in applied_records))

    return {
        "original_run_id": final_disposition.get("original_run_id"),
        "failure_recheck_run_ids": final_disposition.get("failure_recheck_run_ids", []),
        "approved_refresh_count": len(approved_targets),
        "applied_record_count": len(applied_records),
        "applied_counts": applied_counts,
        "source_runs": source_runs,
        "applied_records": applied_records,
        "missing_targets": sorted(approved_targets - set(latest_record_by_target)),
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Deep-Dive Expectation Refresh Summary",
        "",
        f"- Original deep-dive run: `{payload.get('original_run_id') or 'unknown'}`",
        f"- Failure rechecks: `{', '.join(payload.get('failure_recheck_run_ids', []))}`",
        f"- Approved expectation refresh targets: `{payload.get('approved_refresh_count', 0)}`",
        f"- Applied records: `{payload.get('applied_record_count', 0)}`",
        f"- Applied counts: `{payload.get('applied_counts', {})}`",
        "",
        "## Source Runs",
        "",
    ]
    for source in payload.get("source_runs", []):
        lines.append(
            f"- `{source.get('source_results_json')}` -> counts `{source.get('counts', {})}`"
        )
    lines.extend(["", "## Applied Records", "", "| Target | File | Status | Message |", "| --- | --- | --- | --- |"])
    for record in payload.get("applied_records", []):
        lines.append(
            f"| `{record.get('target')}` | `{record.get('expectation_file')}` | "
            f"`{record.get('status')}` | {record.get('message') or ''} |"
        )
    missing_targets = payload.get("missing_targets", [])
    if missing_targets:
        lines.extend(["", "## Missing Approved Targets", ""])
        for target in missing_targets:
            lines.append(f"- `{target}`")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    final_disposition_json: Path,
    apply_summary_jsons: Iterable[Path],
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    final_disposition = _load_json(final_disposition_json)
    apply_summaries = [_load_json(path) for path in apply_summary_jsons]
    payload = build_summary(final_disposition=final_disposition, apply_summaries=apply_summaries)
    out_dir = output_dir or final_disposition_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "deep_dive_expectation_refresh_summary.json"
    md_path = out_dir / "deep_dive_expectation_refresh_summary.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a deep-dive expectation refresh summary from disposition and apply ledgers.")
    parser.add_argument("--final-disposition-json", type=Path, required=True, help="Path to deep_dive_final_disposition.json")
    parser.add_argument("--apply-summary-json", type=Path, action="append", required=True, help="Refresh apply summary JSON (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to the disposition directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(
        final_disposition_json=args.final_disposition_json,
        apply_summary_jsons=args.apply_summary_json,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
