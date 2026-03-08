from __future__ import annotations

import json
from pathlib import Path

import core.benchmark.bench_commands as bench_commands
from core.analysis.history_index import update_history_index
from core.analysis.regressions import compare_suite_summaries
from core.analysis.trends import build_trend_snapshot
from core.benchmark.suites.tier1 import build_tier1_suite_summary, load_tier1_suite


def _write_result_payload(path: Path, *, block_scaling_speedup: float, flash_speedup: float) -> None:
    payload = {
        "timestamp": "2026-03-08T00:00:00Z",
        "results": [
            {
                "chapter": "labs_block_scaling",
                "benchmarks": [
                    {
                        "example": "block_scaling",
                        "status": "succeeded",
                        "baseline_time_ms": 0.1074,
                        "best_speedup": block_scaling_speedup,
                        "best_optimization": "hardware_block_scaled",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 512.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "labs/block_scaling/baseline_block_scaling.py",
                        "nsys_rep": "artifacts/block_scaling.nsys-rep",
                    }
                ],
            },
            {
                "chapter": "labs_flashattention4",
                "benchmarks": [
                    {
                        "example": "flashattention4_alibi",
                        "status": "succeeded",
                        "baseline_time_ms": 5.5622,
                        "best_speedup": flash_speedup,
                        "best_optimization": "flashattention4_alibi",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 1024.0,
                        "best_memory_savings_pct": 18.5,
                        "baseline_file": "labs/flashattention4/baseline_flashattention4.py",
                        "ncu_json": "artifacts/flashattention4_alibi.json",
                    }
                ],
            },
            {
                "chapter": "labs_persistent_decode",
                "benchmarks": [
                    {
                        "example": "persistent_decode",
                        "status": "succeeded",
                        "baseline_time_ms": 1.4107,
                        "best_speedup": 11.93,
                        "best_optimization": "graphs",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 256.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "labs/persistent_decode/baseline_persistent_decode.py",
                    }
                ],
            },
            {
                "chapter": "labs_kv_optimization",
                "benchmarks": [
                    {
                        "example": "kv_standard",
                        "status": "succeeded",
                        "baseline_time_ms": 1585.6,
                        "best_speedup": 1.58,
                        "best_optimization": "kv_standard",
                        "optimization_goal": "memory",
                        "baseline_memory_mb": 32140.0,
                        "best_memory_savings_pct": 49.7,
                        "baseline_file": "labs/kv_optimization/baseline_kv_standard.py",
                    }
                ],
            },
            {
                "chapter": "ch04",
                "benchmarks": [
                    {
                        "example": "gradient_fusion",
                        "status": "succeeded",
                        "baseline_time_ms": 1.0,
                        "best_speedup": 1.21,
                        "best_optimization": "gradient_fusion",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 64.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "ch04/baseline_gradient_fusion.py",
                    }
                ],
            },
            {
                "chapter": "labs_real_world_models",
                "benchmarks": [
                    {
                        "example": "llama_3_1_8b",
                        "status": "succeeded",
                        "baseline_time_ms": 13.143,
                        "best_speedup": 2.49,
                        "best_optimization": "optimized_llama_3_1_8b",
                        "optimization_goal": "performance",
                        "baseline_memory_mb": 4096.0,
                        "best_memory_savings_pct": 0.0,
                        "baseline_file": "labs/real_world_models/baseline_llama_3_1_8b.py",
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_tier1_suite_summary_and_history_artifacts(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    result_json = tmp_path / "results.json"
    _write_result_payload(result_json, block_scaling_speedup=1.45, flash_speedup=14.45)

    summary = build_tier1_suite_summary(result_json, suite, run_id="tier1_run_a")

    assert summary["suite_name"] == "tier1"
    assert summary["summary"]["target_count"] == 6
    assert summary["summary"]["succeeded"] == 6
    assert summary["summary"]["missing"] == 0
    assert summary["summary"]["median_speedup"] > 0
    assert summary["summary"]["representative_speedup"] == summary["summary"]["geomean_speedup"]

    block_scaling = next(target for target in summary["targets"] if target["key"] == "block_scaling")
    assert block_scaling["best_speedup"] == 1.45
    assert block_scaling["artifacts"]["nsys_rep"] == "artifacts/block_scaling.nsys-rep"

    summary_path = tmp_path / "summary.json"
    regression_md_path = tmp_path / "regression_summary.md"
    regression_json_path = tmp_path / "regression_summary.json"
    trend_snapshot_path = tmp_path / "trend_snapshot.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    regression_md_path.write_text("# placeholder\n", encoding="utf-8")
    regression_json_path.write_text("{}", encoding="utf-8")

    updated_index = update_history_index(
        history_root=tmp_path / "history",
        suite=suite,
        summary=summary,
        summary_path=summary_path,
        regression_summary_path=regression_md_path,
        regression_json_path=regression_json_path,
        trend_snapshot_path=trend_snapshot_path,
    )

    assert updated_index["suite_name"] == "tier1"
    assert len(updated_index["runs"]) == 1
    assert updated_index["runs"][0]["regression_json_path"] == str(regression_json_path)
    assert updated_index["runs"][0]["median_speedup"] == summary["summary"]["median_speedup"]
    assert updated_index["runs"][0]["representative_speedup"] == summary["summary"]["representative_speedup"]

    trend = build_trend_snapshot(updated_index)
    assert trend["run_count"] == 1
    assert trend["latest_run_id"] == "tier1_run_a"
    assert trend["best_speedup_seen"] == summary["summary"]["max_speedup"]
    assert trend["representative_speedup"] == summary["summary"]["representative_speedup"]
    assert trend["avg_median_speedup"] == summary["summary"]["median_speedup"]


def test_compare_suite_summaries_detects_speedup_regression_and_new_targets(tmp_path: Path) -> None:
    suite = load_tier1_suite()
    baseline_json = tmp_path / "baseline.json"
    current_json = tmp_path / "current.json"
    _write_result_payload(baseline_json, block_scaling_speedup=1.60, flash_speedup=12.0)
    _write_result_payload(current_json, block_scaling_speedup=1.45, flash_speedup=14.45)

    baseline_summary = build_tier1_suite_summary(baseline_json, suite, run_id="tier1_old")
    current_summary = build_tier1_suite_summary(current_json, suite, run_id="tier1_new")

    comparison = compare_suite_summaries(current_summary, baseline_summary)

    assert comparison["baseline_run_id"] == "tier1_old"
    assert any(
        row["target"] == "labs/block_scaling:block_scaling" and row["reason"] == "speedup"
        for row in comparison["regressions"]
    )
    assert any(
        row["target"] == "labs/flashattention4:flashattention4_alibi" and row["reason"] == "speedup"
        for row in comparison["improvements"]
    )


def test_tier1_doc_mentions_current_targets_and_artifacts() -> None:
    suite = load_tier1_suite()
    doc_path = Path("docs/tier1_benchmark_suite.md")
    text = doc_path.read_text(encoding="utf-8")

    assert "## Current Tier-1 Targets" in text
    assert "## Artifact Contract" in text
    assert "`artifacts/history/tier1/index.json`" in text
    assert "`artifacts/history/tier1/<run_id>/summary.json`" in text

    for target in suite.targets:
        assert f"`{target.target}`" in text


def test_execute_benchmarks_defaults_bench_root_to_repo_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bench_commands, "BENCHMARK_AVAILABLE", False)
    monkeypatch.setattr(bench_commands, "TEST_FUNCTIONS_AVAILABLE", False)

    result = bench_commands._execute_benchmarks(
        targets=["ch04:gradient_fusion"],
        bench_root=None,
        output_format="json",
        profile_type="none",
        artifacts_dir=str(tmp_path / "artifacts"),
        run_id="tier1_default_root_smoke",
        exit_on_failure=False,
    )

    assert Path(result["bench_root"]) == Path(bench_commands.__file__).resolve().parents[2]
    assert result["run_id"] == "tier1_default_root_smoke"
    assert result["error"] == "Benchmark dependencies missing"
