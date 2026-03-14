from pathlib import Path

from core.analysis.derive_chapter_narrative_review_queue import build_queue, write_outputs


def test_build_queue_filters_keep_rows() -> None:
    disposition = {
        "original_run_id": "run123",
        "rows": [
            {
                "target": "ch01:foo",
                "chapter": "ch01",
                "example": "foo",
                "narrative_decision": "keep",
                "expectation_decision": "refresh",
                "disposition": "refresh_expectations",
                "note": "green",
            },
            {
                "target": "ch02:bar",
                "chapter": "ch02",
                "example": "bar",
                "narrative_decision": "reframe",
                "expectation_decision": "hold",
                "disposition": "hold_expectations_reframe_narrative",
                "note": "weak",
            },
            {
                "target": "ch02:baz",
                "chapter": "ch02",
                "example": "baz",
                "narrative_decision": "qualify",
                "expectation_decision": "refresh",
                "disposition": "refresh_with_qualified_narrative",
                "note": "small win",
            },
        ],
    }

    payload = build_queue(disposition)
    assert payload["original_run_id"] == "run123"
    assert payload["counts"] == {"reframe": 1, "qualify": 1, "total": 2}
    assert list(payload["by_chapter"]) == ["ch02"]
    assert [row["target"] for row in payload["by_chapter"]["ch02"]] == ["ch02:bar", "ch02:baz"]


def test_write_outputs(tmp_path: Path) -> None:
    disposition_path = tmp_path / "deep_dive_final_disposition.json"
    disposition_path.write_text(
        '{"original_run_id":"run123","rows":[{"target":"ch02:bar","chapter":"ch02","example":"bar","narrative_decision":"reframe","expectation_decision":"hold","disposition":"hold_expectations_reframe_narrative","note":"weak"}]}',
        encoding="utf-8",
    )

    outputs = write_outputs(final_disposition_json=disposition_path)
    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    assert "Chapter Narrative Review Queue" in outputs["markdown"].read_text(encoding="utf-8")
