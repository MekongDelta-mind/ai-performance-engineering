"""History index helpers for canonical benchmark suites."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, Optional

from core.benchmark.suites.tier1 import Tier1SuiteDefinition


def load_history_index(index_path: Path) -> Dict[str, Any]:
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return {"suite_name": "tier1", "runs": []}


def update_history_index(
    *,
    history_root: Path,
    suite: Tier1SuiteDefinition,
    summary: Dict[str, Any],
    summary_path: Path,
    regression_summary_path: Path,
    regression_json_path: Optional[Path] = None,
    trend_snapshot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    history_root.mkdir(parents=True, exist_ok=True)
    index_path = history_root / "index.json"
    index = load_history_index(index_path)

    entry = {
        "run_id": summary["run_id"],
        "generated_at": summary.get("generated_at"),
        "summary_path": str(summary_path),
        "regression_summary_path": str(regression_summary_path),
        "regression_json_path": str(regression_json_path) if regression_json_path else None,
        "trend_snapshot_path": str(trend_snapshot_path) if trend_snapshot_path else None,
        "avg_speedup": summary.get("summary", {}).get("avg_speedup", 0.0),
        "median_speedup": summary.get("summary", {}).get("median_speedup", 0.0),
        "geomean_speedup": summary.get("summary", {}).get("geomean_speedup", 0.0),
        "representative_speedup": summary.get("summary", {}).get("representative_speedup", 0.0),
        "max_speedup": summary.get("summary", {}).get("max_speedup", 0.0),
        "succeeded": summary.get("summary", {}).get("succeeded", 0),
        "failed": summary.get("summary", {}).get("failed", 0),
        "skipped": summary.get("summary", {}).get("skipped", 0),
        "missing": summary.get("summary", {}).get("missing", 0),
    }

    runs = [existing for existing in index.get("runs", []) if existing.get("run_id") != entry["run_id"]]
    runs.append(entry)
    runs.sort(key=lambda item: str(item.get("generated_at") or item.get("run_id") or ""))

    updated = {
        "suite_name": suite.name,
        "suite_version": suite.version,
        "history_root": str(history_root),
        "runs": runs,
    }
    index_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return updated
