"""Regression summaries for canonical benchmark suites."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _target_map(summary: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not summary:
        return {}
    return {
        str(target.get("target")): target
        for target in summary.get("targets", [])
        if target.get("target")
    }


def compare_suite_summaries(
    current: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    *,
    speedup_regression_threshold_pct: float = 5.0,
    memory_regression_threshold_points: float = 5.0,
) -> Dict[str, Any]:
    if baseline is None:
        return {
            "baseline_run_id": None,
            "current_run_id": current.get("run_id"),
            "regressions": [],
            "improvements": [],
            "new_targets": list(current.get("targets", [])),
            "missing_targets": [],
        }

    current_map = _target_map(current)
    baseline_map = _target_map(baseline)
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []

    for target_name, current_target in current_map.items():
        previous = baseline_map.get(target_name)
        if previous is None:
            continue

        current_status = str(current_target.get("status", "unknown"))
        previous_status = str(previous.get("status", "unknown"))
        if previous_status == "succeeded" and current_status != "succeeded":
            regressions.append(
                {
                    "target": target_name,
                    "reason": "status",
                    "before": previous_status,
                    "after": current_status,
                }
            )
            continue
        if previous_status != "succeeded" and current_status == "succeeded":
            improvements.append(
                {
                    "target": target_name,
                    "reason": "status",
                    "before": previous_status,
                    "after": current_status,
                }
            )

        current_speedup = float(current_target.get("best_speedup", 0.0) or 0.0)
        previous_speedup = float(previous.get("best_speedup", 0.0) or 0.0)
        if current_speedup > 0 and previous_speedup > 0:
            delta_pct = ((current_speedup - previous_speedup) / previous_speedup) * 100.0
            payload = {
                "target": target_name,
                "reason": "speedup",
                "before": previous_speedup,
                "after": current_speedup,
                "delta_pct": delta_pct,
            }
            if delta_pct <= -speedup_regression_threshold_pct:
                regressions.append(payload)
            elif delta_pct >= speedup_regression_threshold_pct:
                improvements.append(payload)

        current_memory = float(current_target.get("best_memory_savings_pct", 0.0) or 0.0)
        previous_memory = float(previous.get("best_memory_savings_pct", 0.0) or 0.0)
        memory_delta = current_memory - previous_memory
        if memory_delta <= -memory_regression_threshold_points:
            regressions.append(
                {
                    "target": target_name,
                    "reason": "memory_savings",
                    "before": previous_memory,
                    "after": current_memory,
                    "delta_points": memory_delta,
                }
            )
        elif memory_delta >= memory_regression_threshold_points:
            improvements.append(
                {
                    "target": target_name,
                    "reason": "memory_savings",
                    "before": previous_memory,
                    "after": current_memory,
                    "delta_points": memory_delta,
                }
            )

    return {
        "baseline_run_id": baseline.get("run_id"),
        "current_run_id": current.get("run_id"),
        "regressions": regressions,
        "improvements": improvements,
        "new_targets": [current_map[name] for name in sorted(current_map.keys() - baseline_map.keys())],
        "missing_targets": [baseline_map[name] for name in sorted(baseline_map.keys() - current_map.keys())],
    }


def render_regression_summary(
    current: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    comparison: Optional[Dict[str, Any]] = None,
) -> str:
    comparison = comparison or compare_suite_summaries(current, baseline)
    lines = [
        "# Tier-1 Regression Summary",
        "",
        f"- Current run: `{current.get('run_id')}`",
        f"- Baseline run: `{comparison.get('baseline_run_id') or 'none'}`",
        "",
        "## Summary",
        "",
        f"- Regressions: {len(comparison.get('regressions', []))}",
        f"- Improvements: {len(comparison.get('improvements', []))}",
        f"- New targets: {len(comparison.get('new_targets', []))}",
        f"- Missing targets: {len(comparison.get('missing_targets', []))}",
        "",
    ]

    def _render_rows(title: str, rows: List[Dict[str, Any]]) -> None:
        lines.extend([f"## {title}", "", "| Target | Reason | Before | After | Delta |", "| --- | --- | ---: | ---: | ---: |"])
        for row in rows:
            delta = row.get("delta_pct")
            if delta is None:
                delta = row.get("delta_points")
            delta_text = "" if delta is None else f"{float(delta):+.2f}"
            before = row.get("before", "")
            after = row.get("after", "")
            if isinstance(before, float):
                before = f"{before:.3f}"
            if isinstance(after, float):
                after = f"{after:.3f}"
            lines.append(f"| `{row.get('target')}` | {row.get('reason')} | {before} | {after} | {delta_text} |")
        lines.append("")

    if comparison.get("regressions"):
        _render_rows("Regressions", comparison["regressions"])
    if comparison.get("improvements"):
        _render_rows("Improvements", comparison["improvements"])

    if comparison.get("new_targets"):
        lines.extend(["## New Targets", ""])
        for target in comparison["new_targets"]:
            lines.append(f"- `{target.get('target')}` ({target.get('category')})")
        lines.append("")

    if comparison.get("missing_targets"):
        lines.extend(["## Missing Targets", ""])
        for target in comparison["missing_targets"]:
            lines.append(f"- `{target.get('target')}` ({target.get('category')})")
        lines.append("")

    if baseline is None:
        lines.extend([
            "## Notes",
            "",
            "No previous canonical tier-1 summary was available, so this run becomes the initial history anchor.",
            "",
        ])

    return "\n".join(lines).rstrip() + "\n"
