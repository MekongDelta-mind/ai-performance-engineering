from pathlib import Path

from core.analysis.derive_deep_dive_expectation_refresh_summary import build_summary, write_outputs


def test_build_summary_filters_to_refresh_targets_and_dedupes_latest() -> None:
    disposition = {
        "original_run_id": "orig",
        "failure_recheck_run_ids": ["r1", "r2"],
        "rows": [
            {"target": "ch01:a", "expectation_decision": "refresh"},
            {"target": "ch01:b", "expectation_decision": "hold"},
            {"target": "ch02:c", "expectation_decision": "refresh"},
        ],
    }
    apply_summaries = [
        {
            "results_json": "run_a/results.json",
            "counts": {"applied": 1},
            "updated_files": ["ch01/expectations_b200.json"],
            "records": [
                {"target": "ch01:a", "expectation_file": "ch01/expectations_b200.json", "status": "updated", "message": "first"},
                {"target": "ch01:b", "expectation_file": "ch01/expectations_b200.json", "status": "updated", "message": "ignored"},
            ],
        },
        {
            "results_json": "run_b/results.json",
            "counts": {"applied": 2},
            "updated_files": ["ch02/expectations_b200.json"],
            "records": [
                {"target": "ch01:a", "expectation_file": "ch01/expectations_b200.json", "status": "unchanged", "message": "latest"},
                {"target": "ch02:c", "expectation_file": "ch02/expectations_b200.json", "status": "improved", "message": "good"},
            ],
        },
    ]

    payload = build_summary(final_disposition=disposition, apply_summaries=apply_summaries)
    assert payload["approved_refresh_count"] == 2
    assert payload["applied_record_count"] == 2
    assert payload["applied_counts"] == {"unchanged": 1, "improved": 1}
    assert payload["missing_targets"] == []
    assert [record["target"] for record in payload["applied_records"]] == ["ch01:a", "ch02:c"]


def test_write_outputs(tmp_path: Path) -> None:
    disposition_path = tmp_path / "deep_dive_final_disposition.json"
    disposition_path.write_text(
        '{"original_run_id":"orig","failure_recheck_run_ids":["r1"],"rows":[{"target":"ch01:a","expectation_decision":"refresh"}]}',
        encoding="utf-8",
    )
    apply_path = tmp_path / "refresh_apply.json"
    apply_path.write_text(
        '{"results_json":"run/results.json","counts":{"applied":1},"updated_files":["ch01/expectations_b200.json"],"records":[{"target":"ch01:a","expectation_file":"ch01/expectations_b200.json","status":"updated","message":"ok"}]}',
        encoding="utf-8",
    )

    outputs = write_outputs(final_disposition_json=disposition_path, apply_summary_jsons=[apply_path])
    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    assert "Approved expectation refresh targets" in outputs["markdown"].read_text(encoding="utf-8")
