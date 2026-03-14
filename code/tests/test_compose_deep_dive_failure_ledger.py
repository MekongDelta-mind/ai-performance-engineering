import json
from pathlib import Path

from core.analysis.compose_deep_dive_failure_ledger import (
    compose_failure_ledger,
    write_failure_ledger,
)


def _write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_compose_failure_ledger_marks_resolved_and_unresolved_targets(tmp_path: Path) -> None:
    original = tmp_path / "artifacts" / "runs" / "original" / "results" / "benchmark_test_results.json"
    recheck_a = tmp_path / "artifacts" / "runs" / "recheck_a" / "results" / "benchmark_test_results.json"
    recheck_b = tmp_path / "artifacts" / "runs" / "recheck_b" / "results" / "benchmark_test_results.json"

    _write_results(
        original,
        {
            "results": [
                {
                    "chapter": "ch09",
                    "benchmarks": [
                        {
                            "example": "compute_bound",
                            "baseline_profiler_statuses": {"nsys": "failed"},
                            "optimizations": [],
                        },
                        {
                            "example": "cublas_gemm_fp4_perchannel",
                            "baseline_profiler_statuses": {"nsys": "succeeded"},
                            "optimizations": [
                                {
                                    "technique": "optimized_cublas_gemm_fp4_perchannel",
                                    "status": "failed_profiler",
                                    "optimized_profiler_statuses": {"nsys": "failed"},
                                }
                            ],
                        },
                    ],
                }
            ]
        },
    )
    _write_results(
        recheck_a,
        {
            "results": [
                {
                    "chapter": "ch09",
                    "benchmarks": [
                        {
                            "example": "compute_bound",
                            "baseline_profiler_statuses": {"nsys": "succeeded"},
                            "optimizations": [
                                {
                                    "technique": "optimized_compute_bound",
                                    "status": "succeeded",
                                    "speedup": 1.81,
                                    "optimized_profiler_statuses": {"nsys": "succeeded"},
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    )
    _write_results(
        recheck_b,
        {
            "results": [
                {
                    "chapter": "ch09",
                    "benchmarks": [
                        {
                            "example": "cublas_gemm_fp4_perchannel",
                            "baseline_profiler_statuses": {"nsys": "succeeded"},
                            "optimizations": [
                                {
                                    "technique": "optimized_cublas_gemm_fp4_perchannel",
                                    "status": "failed_profiler",
                                    "optimized_profiler_statuses": {"nsys": "failed"},
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    )

    ledger = compose_failure_ledger(
        original_results_json=original,
        recheck_results_json=[recheck_a, recheck_b],
    )

    assert ledger["summary"]["original_run_id"] == "original"
    assert ledger["summary"]["recheck_run_ids"] == ["recheck_a", "recheck_b"]
    assert ledger["summary"]["total_original_failures"] == 2
    assert ledger["summary"]["resolved_count"] == 1
    assert ledger["summary"]["unresolved_count"] == 1

    rows = {row["target"]: row for row in ledger["rows"]}
    assert rows["ch09:compute_bound"]["resolved"] is True
    assert rows["ch09:compute_bound"]["latest_status"] == "succeeded"
    assert rows["ch09:compute_bound"]["best_speedup"] == 1.81
    assert rows["ch09:cublas_gemm_fp4_perchannel"]["resolved"] is False
    assert rows["ch09:cublas_gemm_fp4_perchannel"]["latest_status"] == "failed_profiler"
    assert rows["ch09:cublas_gemm_fp4_perchannel"]["latest_run_id"] == "recheck_b"


def test_write_failure_ledger_materializes_json_and_markdown(tmp_path: Path) -> None:
    original = tmp_path / "artifacts" / "runs" / "original" / "results" / "benchmark_test_results.json"
    recheck = tmp_path / "artifacts" / "runs" / "recheck" / "results" / "benchmark_test_results.json"

    _write_results(
        original,
        {
            "results": [
                {
                    "chapter": "ch10",
                    "benchmarks": [
                        {
                            "example": "foo",
                            "baseline_profiler_statuses": {"nsys": "failed"},
                            "optimizations": [],
                        }
                    ],
                }
            ]
        },
    )
    _write_results(
        recheck,
        {
            "results": [
                {
                    "chapter": "ch10",
                    "benchmarks": [
                        {
                            "example": "foo",
                            "baseline_profiler_statuses": {"nsys": "succeeded"},
                            "optimizations": [
                                {
                                    "technique": "optimized_foo",
                                    "status": "succeeded",
                                    "speedup": 1.5,
                                    "optimized_profiler_statuses": {"nsys": "succeeded"},
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    )

    outputs = write_failure_ledger(
        original_results_json=original,
        recheck_results_json=[recheck],
        output_dir=tmp_path / "out",
    )

    json_payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert json_payload["summary"]["resolved_count"] == 1

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "Final Deep-Dive Failure Ledger" in markdown
    assert "`ch10:foo`" in markdown
