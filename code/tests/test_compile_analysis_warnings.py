from pathlib import Path

from core.compile_analysis import load_compile_analysis


def test_compile_analysis_surfaces_invalid_saved_report_warning(tmp_path: Path) -> None:
    report_path = tmp_path / "compile_report_demo.json"
    report_path.write_text("{bad", encoding="utf-8")

    result = load_compile_analysis(tmp_path)

    assert result["warnings"]
    assert str(report_path) in result["warnings"][0]
    assert result["artifact_path"] == str(report_path)
    assert result["has_real_data"] is False


def test_compile_analysis_preserves_benchmark_summary_after_report_warning(tmp_path: Path) -> None:
    report_path = tmp_path / "compile_report_demo.json"
    report_path.write_text("[]", encoding="utf-8")

    result = load_compile_analysis(
        tmp_path,
        benchmarks=[
            {
                "name": "torch_compile_gemm",
                "chapter": "ch14",
                "speedup": 1.7,
                "baseline_time_ms": 10.0,
                "optimized_time_ms": 5.88,
                "optimizations": ["torch_compile"],
            }
        ],
    )

    assert result["warnings"]
    assert "expected JSON object, got list" in result["warnings"][0]
    assert result["has_real_data"] is True
    assert result["compile_benchmarks"]
    assert result["recommendations"]
