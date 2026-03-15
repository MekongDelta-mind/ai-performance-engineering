from pathlib import Path

from core import ncu_analysis


def test_ncu_analysis_surfaces_source_extraction_warning(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "artifacts" / "runs" / "demo" / "capture.ncu-rep"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("rep", encoding="utf-8")

    monkeypatch.setattr(
        "core.profile_insights._extract_ncu_sources",
        lambda path: (_ for _ in ()).throw(RuntimeError(f"source boom for {path.name}")),
    )
    monkeypatch.setattr(
        "core.profile_insights._extract_ncu_disassembly",
        lambda path: (_ for _ in ()).throw(RuntimeError(f"disassembly boom for {path.name}")),
    )

    result = ncu_analysis.load_ncu_deepdive(tmp_path)

    assert result["available"] is True
    assert result["warnings"]
    assert "Failed to extract NCU sources/disassembly" in result["warnings"][0]
    assert result["latest_file_path"] == str(report_path)


def test_ncu_analysis_surfaces_csv_parse_warning(tmp_path: Path) -> None:
    csv_path = tmp_path / "artifacts" / "runs" / "demo" / "capture_ncu.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(b"\xff\xfe\x00")

    result = ncu_analysis.load_ncu_deepdive(tmp_path)

    assert result["available"] is True
    assert result["parse_error"]
    assert result["warnings"]
    assert result["latest_file_path"] == str(csv_path)
