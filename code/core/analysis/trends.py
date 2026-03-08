"""Trend snapshots for canonical benchmark suite history."""

from __future__ import annotations

from typing import Any, Dict, List


def build_trend_snapshot(index: Dict[str, Any]) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = list(index.get("runs", []))
    by_date = [
        {
            "run_id": run.get("run_id"),
            "generated_at": run.get("generated_at"),
            "avg_speedup": run.get("avg_speedup", 0.0),
            "median_speedup": run.get("median_speedup", 0.0),
            "geomean_speedup": run.get("geomean_speedup", 0.0),
            "representative_speedup": run.get("representative_speedup", run.get("geomean_speedup", 0.0)),
            "max_speedup": run.get("max_speedup", 0.0),
            "succeeded": run.get("succeeded", 0),
            "failed": run.get("failed", 0),
            "skipped": run.get("skipped", 0),
            "missing": run.get("missing", 0),
        }
        for run in runs
    ]
    if by_date:
        avg_speedup = sum(item["avg_speedup"] for item in by_date) / len(by_date)
        median_speedup = sum(item["median_speedup"] for item in by_date) / len(by_date)
        geomean_speedup = sum(item["geomean_speedup"] for item in by_date) / len(by_date)
        representative_speedup = sum(item["representative_speedup"] for item in by_date) / len(by_date)
        max_speedup = max(item["max_speedup"] for item in by_date)
    else:
        avg_speedup = 0.0
        median_speedup = 0.0
        geomean_speedup = 0.0
        representative_speedup = 0.0
        max_speedup = 0.0

    return {
        "suite_name": index.get("suite_name", "tier1"),
        "run_count": len(by_date),
        "history": by_date,
        "avg_speedup": avg_speedup,
        "avg_median_speedup": median_speedup,
        "avg_geomean_speedup": geomean_speedup,
        "representative_speedup": representative_speedup,
        "best_speedup_seen": max_speedup,
        "latest_run_id": by_date[-1]["run_id"] if by_date else None,
    }
