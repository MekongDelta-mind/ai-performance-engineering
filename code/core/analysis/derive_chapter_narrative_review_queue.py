#!/usr/bin/env python3
"""Derive a chapter narrative review queue from the final deep-dive disposition."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_queue(final_disposition: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(final_disposition.get("rows", []))
    queued = [row for row in rows if row.get("narrative_decision") not in {None, "keep"}]
    queued.sort(key=lambda row: (str(row.get("chapter")), str(row.get("example"))))

    by_chapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in queued:
        by_chapter[str(row.get("chapter"))].append(
            {
                "target": row.get("target"),
                "disposition": row.get("disposition"),
                "expectation_decision": row.get("expectation_decision"),
                "narrative_decision": row.get("narrative_decision"),
                "note": row.get("note"),
            }
        )

    counts = dict(Counter(str(row.get("narrative_decision")) for row in queued))
    counts["total"] = len(queued)

    return {
        "original_run_id": final_disposition.get("original_run_id"),
        "counts": counts,
        "by_chapter": dict(sorted(by_chapter.items())),
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Chapter Narrative Review Queue",
        "",
        f"- Original deep-dive run: `{payload.get('original_run_id') or 'unknown'}`",
        f"- Counts: `{payload.get('counts', {})}`",
        "",
    ]
    for chapter, rows in payload.get("by_chapter", {}).items():
        lines.extend(
            [
                f"## {chapter}",
                "",
                "| Target | Narrative | Expectations | Disposition | Note |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row.get('target')}` | `{row.get('narrative_decision')}` | "
                f"`{row.get('expectation_decision')}` | `{row.get('disposition')}` | {row.get('note') or ''} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_outputs(*, final_disposition_json: Path, output_dir: Path | None = None) -> Dict[str, Path]:
    final_disposition = _load_json(final_disposition_json)
    payload = build_queue(final_disposition)
    out_dir = output_dir or final_disposition_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "chapter_narrative_review_queue.json"
    md_path = out_dir / "chapter_narrative_review_queue.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a chapter narrative review queue from a final deep-dive disposition.")
    parser.add_argument("--final-disposition-json", type=Path, required=True, help="Path to deep_dive_final_disposition.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to the disposition directory)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = write_outputs(final_disposition_json=args.final_disposition_json, output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
