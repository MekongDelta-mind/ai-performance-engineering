from __future__ import annotations

import json

from core.analysis.derive_deep_dive_final_disposition import classify_rows, write_outputs


def test_classify_rows_merges_failures_and_weak_actions() -> None:
    final_failure_ledger = {
        "original_run_id": "orig",
        "recheck_run_ids": ["r1"],
        "rows": [
            {
                "chapter": "ch01",
                "example": "broken_but_fixed",
                "resolved": True,
                "latest_status": "succeeded",
                "latest_issues": [],
                "best_speedup": 2.0,
            },
            {
                "chapter": "ch02",
                "example": "still_broken",
                "resolved": False,
                "latest_status": "failed_profiler",
                "latest_issues": [{"kind": "failed_profiler", "detail": "baseline:nsys:failed"}],
                "best_speedup": None,
            },
        ],
    }
    weak_actions = {
        "run_id": "weak",
        "rows": [
            {
                "chapter": "ch03",
                "example": "family_case",
                "bucket": "flat_or_negative",
                "action": "family_level_investigation_before_blessing",
                "note": "family hold",
            },
            {
                "chapter": "ch04",
                "example": "small_win",
                "bucket": "weak_positive",
                "action": "qualify_as_small_or_contextual_win",
                "note": "small/contextual",
            },
            {
                "chapter": "ch05",
                "example": "non_speed",
                "bucket": "non_speed_goal",
                "action": "treat_as_non_speed_example",
                "note": "not a speed target",
            },
        ],
    }
    weak_root_causes = {
        "rows": [
            {
                "category": "repeated_family",
                "targets": ["ch03:family_case"],
                "root_cause": "shared weak family",
            }
        ]
    }

    rows = classify_rows(
        final_failure_ledger=final_failure_ledger,
        weak_actions=weak_actions,
        weak_root_causes=weak_root_causes,
    )
    by_target = {row["target"]: row for row in rows}

    assert by_target["ch01:broken_but_fixed"]["disposition"] == "refresh_expectations"
    assert by_target["ch02:still_broken"]["disposition"] == "unresolved_failure_blocker"
    assert by_target["ch02:still_broken"]["failure_status"] == "failed_profiler"
    assert "baseline:nsys:failed" in by_target["ch02:still_broken"]["note"]
    assert by_target["ch03:family_case"]["disposition"] == "hold_expectations_family_investigation"
    assert by_target["ch03:family_case"]["note"] == "shared weak family"
    assert by_target["ch04:small_win"]["disposition"] == "refresh_with_qualified_narrative"
    assert by_target["ch05:non_speed"]["disposition"] == "evaluate_non_speed_goal"


def test_write_outputs_writes_json_and_markdown(tmp_path) -> None:
    final_failure_path = tmp_path / "final_failure_ledger.json"
    weak_actions_path = tmp_path / "deep_dive_weak_case_actions.json"
    weak_root_causes_path = tmp_path / "deep_dive_weak_case_root_causes.json"

    final_failure_path.write_text(
        json.dumps(
            {
                "original_run_id": "orig",
                "recheck_run_ids": ["recheck"],
                "rows": [
                    {
                        "chapter": "ch01",
                        "example": "fixed",
                        "resolved": True,
                        "latest_status": "succeeded",
                        "latest_issues": [],
                        "best_speedup": 1.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    weak_actions_path.write_text(json.dumps({"run_id": "weak", "rows": []}), encoding="utf-8")
    weak_root_causes_path.write_text(json.dumps({"generated_on": "today", "rows": []}), encoding="utf-8")

    outputs = write_outputs(
        final_failure_ledger_json=final_failure_path,
        weak_actions_json=weak_actions_path,
        weak_root_causes_json=weak_root_causes_path,
        output_dir=tmp_path,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["disposition_counts"]["refresh_expectations"] == 1
    assert payload["original_run_id"] == "orig"
    assert payload["failure_recheck_run_ids"] == ["recheck"]
    assert payload["failure_summary"] == {
        "total_original_failures": None,
        "resolved_count": None,
        "unresolved_count": None,
    }
    assert "Deep-Dive Final Disposition" in outputs["markdown"].read_text(encoding="utf-8")


def test_write_outputs_reads_summary_wrapped_failure_ledger(tmp_path) -> None:
    final_failure_path = tmp_path / "final_failure_ledger.json"
    weak_actions_path = tmp_path / "deep_dive_weak_case_actions.json"
    weak_root_causes_path = tmp_path / "deep_dive_weak_case_root_causes.json"

    final_failure_path.write_text(
        json.dumps(
            {
                "summary": {
                    "original_run_id": "orig_summary",
                    "recheck_run_ids": ["r1", "r2"],
                    "total_original_failures": 18,
                    "resolved_count": 17,
                    "unresolved_count": 1,
                },
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    weak_actions_path.write_text(json.dumps({"run_id": "weak", "rows": []}), encoding="utf-8")
    weak_root_causes_path.write_text(json.dumps({"generated_on": "today", "rows": []}), encoding="utf-8")

    outputs = write_outputs(
        final_failure_ledger_json=final_failure_path,
        weak_actions_json=weak_actions_path,
        weak_root_causes_json=weak_root_causes_path,
        output_dir=tmp_path,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["original_run_id"] == "orig_summary"
    assert payload["failure_recheck_run_ids"] == ["r1", "r2"]
    assert payload["failure_summary"] == {
        "total_original_failures": 18,
        "resolved_count": 17,
        "unresolved_count": 1,
    }
