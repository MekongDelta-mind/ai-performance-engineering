import json
from pathlib import Path

from core import profile_artifacts
from core.perf_core_base import PerformanceCoreBase


def _profiles_dir(root: Path) -> Path:
    path = root / "artifacts" / "runs" / "demo_run" / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_flame_graph_data_surfaces_invalid_trace_warning(tmp_path: Path) -> None:
    trace_path = _profiles_dir(tmp_path) / "chrome_trace.json"
    trace_path.write_text("{not-json", encoding="utf-8")

    data = profile_artifacts.load_flame_graph_data(tmp_path)

    assert data["error"]
    assert data["warnings"]
    assert str(trace_path) in data["warnings"][0]
    assert data["trace_path"] == str(trace_path)


def test_memory_timeline_surfaces_unsupported_artifact_format(tmp_path: Path) -> None:
    memory_path = _profiles_dir(tmp_path) / "gpu_memory_profile.pickle"
    memory_path.write_bytes(b"pickle-bytes")

    data = profile_artifacts.load_memory_timeline(tmp_path)

    assert data["has_real_data"] is False
    assert data["error"]
    assert data["warnings"]
    assert "Unsupported memory timeline artifact format" in data["warnings"][0]
    assert data["artifact_path"] == str(memory_path)


def test_cpu_gpu_timeline_ignores_malformed_events_with_warning(tmp_path: Path) -> None:
    trace_path = _profiles_dir(tmp_path) / "timeline.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "ts": 1000, "dur": 500, "name": "cpu_op", "cat": "python"},
                    "bad-event",
                ]
            }
        ),
        encoding="utf-8",
    )

    data = profile_artifacts.load_cpu_gpu_timeline(tmp_path)

    assert len(data["cpu"]) == 1
    assert data["summary"]["total_time_ms"] == 0.5
    assert data["warnings"]
    assert "Ignored 1 malformed trace event" in data["warnings"][0]
    assert data["trace_path"] == str(trace_path)


def test_hta_analysis_surfaces_non_object_payload_warning(tmp_path: Path) -> None:
    hta_path = tmp_path / "analysis" / "hta_report_demo.json"
    hta_path.parent.mkdir(parents=True, exist_ok=True)
    hta_path.write_text("[]", encoding="utf-8")

    data = profile_artifacts.load_hta_analysis(tmp_path)

    assert data["error"]
    assert data["warnings"]
    assert "expected JSON object, got list" in data["warnings"][0]
    assert data["artifact_path"] == str(hta_path)


def test_perf_core_torch_profiler_preserves_metadata_warning(tmp_path: Path) -> None:
    profile_dir = tmp_path / "captures" / "torch"
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_path = profile_dir / "torch_profile_summary.json"
    metadata_path = profile_dir / "metadata.json"
    summary_path.write_text(json.dumps({"top_ops": [{"name": "matmul"}], "mode": "full"}), encoding="utf-8")
    metadata_path.write_text("[]", encoding="utf-8")

    core = PerformanceCoreBase(bench_root=tmp_path)
    data = core.get_torch_profiler()

    assert data["top_ops"] == [{"name": "matmul"}]
    assert data["warnings"]
    assert "torch profiler metadata" in data["warnings"][0]
    assert data["metadata_error"]
    assert data["summary_path"] == str(summary_path)
    assert data["metadata_path"] == str(metadata_path)
