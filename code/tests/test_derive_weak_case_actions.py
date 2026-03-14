import json
from pathlib import Path

from core.analysis.derive_weak_case_actions import classify_weak_case_actions, write_outputs


def test_classify_weak_case_actions_assigns_expected_buckets():
    rows = [
        {"chapter": "ch01", "example": "family_case", "goal": "speed", "bucket": "flat_or_negative", "best_opt_ms": 1.2, "technique": "opt"},
        {"chapter": "ch02", "example": "family_case", "goal": "speed", "bucket": "flat_or_negative", "best_opt_ms": 1.1, "technique": "opt"},
        {"chapter": "ch03", "example": "family_case", "goal": "speed", "bucket": "flat_or_negative", "best_opt_ms": 1.0, "technique": "opt"},
        {"chapter": "ch04", "example": "no_winner", "goal": "speed", "bucket": "flat_or_negative", "best_opt_ms": None, "technique": ""},
        {"chapter": "ch05", "example": "flat_case", "goal": "speed", "bucket": "flat_or_negative", "best_opt_ms": 2.0, "technique": "opt"},
        {"chapter": "ch06", "example": "weak_case", "goal": "speed", "bucket": "weak_positive", "best_opt_ms": 3.0, "technique": "opt"},
        {"chapter": "ch07", "example": "memory_story", "goal": "memory", "bucket": "non_speed_goal", "best_opt_ms": 4.0, "technique": "opt"},
    ]

    classified = classify_weak_case_actions(rows)
    by_target = {(row["chapter"], row["example"]): row for row in classified}

    assert by_target[("ch01", "family_case")]["action"] == "family_level_investigation_before_blessing"
    assert by_target[("ch04", "no_winner")]["action"] == "investigate_missing_optimized_win"
    assert by_target[("ch05", "flat_case")]["action"] == "hold_expectations_and_reframe_story"
    assert by_target[("ch06", "weak_case")]["action"] == "qualify_as_small_or_contextual_win"
    assert by_target[("ch07", "memory_story")]["action"] == "treat_as_non_speed_example"


def test_write_outputs_materializes_summary_and_markdown(tmp_path: Path):
    review_json = tmp_path / "deep_dive_review_candidates_refined.json"
    review_json.write_text(
        json.dumps(
            {
                "run_id": "run_123",
                "count": 2,
                "rows": [
                    {
                        "chapter": "ch01",
                        "example": "foo",
                        "goal": "speed",
                        "bucket": "flat_or_negative",
                        "best_opt_ms": 1.0,
                        "technique": "optimized_foo",
                    },
                    {
                        "chapter": "ch02",
                        "example": "bar",
                        "goal": "memory",
                        "bucket": "non_speed_goal",
                        "best_opt_ms": 2.0,
                        "technique": "optimized_bar",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = write_outputs(review_candidates_json=review_json, output_dir=tmp_path / "out")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_123"
    assert payload["total"] == 2
    assert payload["action_counts"]["hold_expectations_and_reframe_story"] == 1
    assert payload["action_counts"]["treat_as_non_speed_example"] == 1

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "Weak cases are not auto-blessed." in markdown
    assert "`ch01:foo`" in markdown
