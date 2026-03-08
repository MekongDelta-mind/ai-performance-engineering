#!/usr/bin/env python3
"""
Unit tests for the HTA (Holistic Trace Analysis) Analyzer.

Tests cover:
- Trace parsing (PyTorch profiler JSON format)
- Kernel statistics extraction
- Critical path analysis
- Idle time calculation
- Communication/computation overlap
- Trace comparison
"""

import json
import tempfile
from pathlib import Path

import pytest

from core.analysis.hta_analyzer import (
    HTAAnalyzer,
    HTATraceAnalysis,
    HTAComparison,
    KernelStats,
    CriticalPathSegment,
    IdleTimeBreakdown,
    CommCompOverlap,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_trace_events():
    """Sample PyTorch profiler trace events."""
    return {
        "traceEvents": [
            {
                "name": "matmul_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 1000,
                "dur": 5000,
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "matmul_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 7000,
                "dur": 4500,
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "softmax_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 12000,
                "dur": 1000,
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "cudaMemcpyAsync",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 14000,
                "dur": 2000,
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "nccl_AllReduce",
                "cat": "nccl",
                "ph": "X",
                "ts": 16500,
                "dur": 3000,
                "pid": 0,
                "tid": 0,
            },
        ]
    }


@pytest.fixture
def optimized_trace_events():
    """Sample optimized trace with faster kernels."""
    return {
        "traceEvents": [
            {
                "name": "matmul_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 1000,
                "dur": 2500,  # 2x faster
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "matmul_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 4000,
                "dur": 2200,
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "softmax_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 6500,
                "dur": 500,  # 2x faster
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "cudaMemcpyAsync",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 7500,
                "dur": 1000,  # 2x faster
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "nccl_AllReduce",
                "cat": "nccl",
                "ph": "X",
                "ts": 6500,  # Overlaps with softmax
                "dur": 2000,
                "pid": 0,
                "tid": 1,  # Different stream
            },
        ]
    }


@pytest.fixture
def baseline_trace_file(sample_trace_events, tmp_path):
    """Create a temporary baseline trace file."""
    path = tmp_path / "baseline_trace.json"
    path.write_text(json.dumps(sample_trace_events, indent=2))
    return path


@pytest.fixture
def optimized_trace_file(optimized_trace_events, tmp_path):
    """Create a temporary optimized trace file."""
    path = tmp_path / "optimized_trace.json"
    path.write_text(json.dumps(optimized_trace_events, indent=2))
    return path


# ============================================================================
# Unit Tests: HTAAnalyzer
# ============================================================================

class TestHTAAnalyzer:
    """Tests for the HTAAnalyzer class."""
    
    def test_analyzer_creation(self):
        """Test that analyzer can be created."""
        analyzer = HTAAnalyzer()
        assert analyzer is not None
    
    def test_is_available_property(self):
        """Test is_available property."""
        analyzer = HTAAnalyzer()
        # Should be a boolean
        assert isinstance(analyzer.is_available, bool)


class TestAnalyzeTrace:
    """Tests for single trace analysis."""
    
    def test_analyze_valid_trace(self, baseline_trace_file):
        """Test analysis of a valid trace file."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        assert result is not None
        assert isinstance(result, HTATraceAnalysis)
        assert result.trace_path == str(baseline_trace_file)
    
    def test_kernel_count(self, baseline_trace_file):
        """Test that kernel count is correct."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        # Should have 5 kernel events
        assert result.kernel_count == 5
    
    def test_unique_kernel_count(self, baseline_trace_file):
        """Test unique kernel count."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        # Should have 4 unique kernel names
        # (matmul_kernel, softmax_kernel, cudaMemcpyAsync, nccl_AllReduce)
        assert result.unique_kernel_count >= 4
    
    def test_top_kernels(self, baseline_trace_file):
        """Test top kernels extraction."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file, top_k=3)
        
        assert len(result.top_kernels) <= 3
        assert all(isinstance(k, KernelStats) for k in result.top_kernels)
        
        # First kernel should have highest total time
        if len(result.top_kernels) >= 2:
            assert result.top_kernels[0].total_time_us >= result.top_kernels[1].total_time_us
    
    def test_missing_file(self, tmp_path):
        """Test handling of missing file."""
        analyzer = HTAAnalyzer()
        missing = tmp_path / "does_not_exist.json"
        result = analyzer.analyze_trace(missing)
        assert result is None
    
    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON."""
        analyzer = HTAAnalyzer()
        invalid = tmp_path / "invalid.json"
        invalid.write_text("not valid json {{{")
        result = analyzer.analyze_trace(invalid)
        assert result is None
    
    def test_empty_trace(self, tmp_path):
        """Test handling of empty trace."""
        analyzer = HTAAnalyzer()
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"traceEvents": []}))
        result = analyzer.analyze_trace(empty)
        assert result is None


class TestKernelStats:
    """Tests for kernel statistics extraction."""
    
    def test_kernel_total_time(self, baseline_trace_file):
        """Test kernel total time calculation."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        # Find matmul_kernel stats
        matmul = next((k for k in result.top_kernels if k.name == "matmul_kernel"), None)
        if matmul:
            # Should be 5000 + 4500 = 9500 us
            assert matmul.total_time_us == pytest.approx(9500, rel=0.1)
            assert matmul.count == 2
    
    def test_kernel_average_time(self, baseline_trace_file):
        """Test kernel average time calculation."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        matmul = next((k for k in result.top_kernels if k.name == "matmul_kernel"), None)
        if matmul:
            # Average: (5000 + 4500) / 2 = 4750 us
            assert matmul.avg_time_us == pytest.approx(4750, rel=0.1)
    
    def test_kernel_min_max(self, baseline_trace_file):
        """Test kernel min/max times."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        matmul = next((k for k in result.top_kernels if k.name == "matmul_kernel"), None)
        if matmul:
            assert matmul.min_time_us == pytest.approx(4500, rel=0.1)
            assert matmul.max_time_us == pytest.approx(5000, rel=0.1)


class TestCriticalPath:
    """Tests for critical path analysis."""
    
    def test_critical_path_exists(self, baseline_trace_file):
        """Test that critical path is extracted."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        assert result.critical_path is not None
        assert len(result.critical_path) > 0
    
    def test_critical_path_segments(self, baseline_trace_file):
        """Test critical path segment structure."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        for segment in result.critical_path:
            assert isinstance(segment, CriticalPathSegment)
            assert segment.duration_us >= 0
            assert segment.category in ("compute", "memory", "communication", "idle")
    
    def test_critical_path_categories(self, baseline_trace_file):
        """Test that different categories are detected."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        categories = {seg.category for seg in result.critical_path}
        # Should have at least compute and memory
        assert "compute" in categories or "memory" in categories


class TestIdleTime:
    """Tests for idle time analysis."""
    
    def test_idle_breakdown_exists(self, baseline_trace_file):
        """Test that idle breakdown is calculated."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        assert result.idle_breakdown is not None
        assert isinstance(result.idle_breakdown, IdleTimeBreakdown)
    
    def test_idle_time_calculation(self, baseline_trace_file):
        """Test idle time between events."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        # There should be some idle time between kernels
        # e.g., gap between ts=6000 (end of first matmul) and ts=7000 (start of second)
        assert result.idle_breakdown.total_idle_us >= 0
    
    def test_idle_percentage(self, baseline_trace_file):
        """Test idle percentage is reasonable."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        assert 0 <= result.idle_breakdown.idle_percentage <= 100


class TestCommCompOverlap:
    """Tests for communication/computation overlap."""
    
    def test_overlap_exists(self, baseline_trace_file):
        """Test that overlap analysis exists."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(baseline_trace_file)
        
        assert result.comm_comp_overlap is not None
        assert isinstance(result.comm_comp_overlap, CommCompOverlap)
    
    def test_overlap_with_overlapping_events(self, optimized_trace_file):
        """Test overlap detection with overlapping events."""
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(optimized_trace_file)
        
        # nccl_AllReduce (ts=6500, dur=2000) overlaps with softmax (ts=6500, dur=500)
        # Should detect some overlap
        cc = result.comm_comp_overlap
        assert cc.total_comm_us > 0


# ============================================================================
# Tests for Trace Comparison
# ============================================================================

class TestCompareTraces:
    """Tests for trace comparison functionality."""
    
    def test_compare_valid_traces(self, baseline_trace_file, optimized_trace_file):
        """Test comparison of two valid traces."""
        analyzer = HTAAnalyzer()
        comparison = analyzer.compare_traces(baseline_trace_file, optimized_trace_file)
        
        assert comparison is not None
        assert isinstance(comparison, HTAComparison)
    
    def test_total_speedup(self, baseline_trace_file, optimized_trace_file):
        """Test total speedup calculation."""
        analyzer = HTAAnalyzer()
        comparison = analyzer.compare_traces(baseline_trace_file, optimized_trace_file)
        
        # Optimized trace should be faster
        assert comparison.total_speedup > 1.0
    
    def test_kernel_speedups(self, baseline_trace_file, optimized_trace_file):
        """Test per-kernel speedup tracking."""
        analyzer = HTAAnalyzer()
        comparison = analyzer.compare_traces(baseline_trace_file, optimized_trace_file)
        
        assert comparison.kernel_speedups is not None
        # matmul_kernel should show speedup
        if "matmul_kernel" in comparison.kernel_speedups:
            assert comparison.kernel_speedups["matmul_kernel"] > 1.0
    
    def test_insights_generated(self, baseline_trace_file, optimized_trace_file):
        """Test that insights are generated."""
        analyzer = HTAAnalyzer()
        comparison = analyzer.compare_traces(baseline_trace_file, optimized_trace_file)
        
        assert comparison.insights is not None
        assert len(comparison.insights) > 0
    
    def test_compare_with_missing_baseline(self, optimized_trace_file, tmp_path):
        """Test comparison with missing baseline."""
        analyzer = HTAAnalyzer()
        missing = tmp_path / "missing.json"
        comparison = analyzer.compare_traces(missing, optimized_trace_file)
        assert comparison is None
    
    def test_compare_with_missing_optimized(self, baseline_trace_file, tmp_path):
        """Test comparison with missing optimized trace."""
        analyzer = HTAAnalyzer()
        missing = tmp_path / "missing.json"
        comparison = analyzer.compare_traces(baseline_trace_file, missing)
        assert comparison is None


# ============================================================================
# Tests for Report Generation
# ============================================================================

class TestReportGeneration:
    """Tests for markdown report generation."""
    
    def test_single_trace_report(self, baseline_trace_file):
        """Test single trace markdown report."""
        analyzer = HTAAnalyzer()
        analysis = analyzer.analyze_trace(baseline_trace_file)
        
        markdown = analyzer.generate_markdown_report(analysis)
        
        assert "# HTA Trace Analysis Report" in markdown
        assert "Top Kernels" in markdown
        assert "Critical Path Analysis" in markdown
        assert "Idle Time Breakdown" in markdown
    
    def test_comparison_report(self, baseline_trace_file, optimized_trace_file):
        """Test comparison markdown report."""
        analyzer = HTAAnalyzer()
        comparison = analyzer.compare_traces(baseline_trace_file, optimized_trace_file)
        
        markdown = analyzer.generate_comparison_markdown(comparison)
        
        assert "# HTA Trace Comparison Report" in markdown
        assert "Summary" in markdown
        assert "Insights" in markdown
        assert "Speedup" in markdown.lower() or "speedup" in markdown
    
    def test_report_contains_trace_path(self, baseline_trace_file):
        """Test that report contains trace path."""
        analyzer = HTAAnalyzer()
        analysis = analyzer.analyze_trace(baseline_trace_file)
        markdown = analyzer.generate_markdown_report(analysis)
        
        assert str(baseline_trace_file.name) in markdown or "baseline" in markdown.lower()


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_single_event_trace(self, tmp_path):
        """Test trace with single event."""
        trace = tmp_path / "single.json"
        trace.write_text(json.dumps({
            "traceEvents": [
                {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 0, "dur": 1000}
            ]
        }))
        
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(trace)
        
        assert result is not None
        assert result.kernel_count == 1
    
    def test_trace_with_no_duration(self, tmp_path):
        """Test trace with events missing duration."""
        trace = tmp_path / "no_dur.json"
        trace.write_text(json.dumps({
            "traceEvents": [
                {"name": "instant_event", "cat": "marker", "ph": "i", "ts": 0},
                {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 100, "dur": 1000},
            ]
        }))
        
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(trace)
        
        # Should handle gracefully
        assert result is not None
    
    def test_trace_as_list(self, tmp_path):
        """Test trace in list format (alternative format)."""
        trace = tmp_path / "list.json"
        trace.write_text(json.dumps([
            {"name": "kernel", "cat": "kernel", "ph": "X", "ts": 0, "dur": 1000}
        ]))
        
        analyzer = HTAAnalyzer()
        result = analyzer.analyze_trace(trace)
        
        assert result is not None
    
    def test_identical_traces(self, baseline_trace_file):
        """Test comparison of identical traces."""
        analyzer = HTAAnalyzer()
        comparison = analyzer.compare_traces(baseline_trace_file, baseline_trace_file)
        
        assert comparison is not None
        # Speedup should be ~1.0
        assert comparison.total_speedup == pytest.approx(1.0, rel=0.1)


# ============================================================================
# Tests for Data Classes
# ============================================================================

class TestDataClasses:
    """Tests for data class structures."""
    
    def test_kernel_stats_creation(self):
        """Test KernelStats can be created."""
        stats = KernelStats(
            name="test_kernel",
            count=10,
            total_time_us=5000.0,
            avg_time_us=500.0,
            min_time_us=400.0,
            max_time_us=600.0,
            std_dev_us=50.0,
            percentage_of_total=25.0,
        )
        assert stats.name == "test_kernel"
        assert stats.count == 10
    
    def test_critical_path_segment_creation(self):
        """Test CriticalPathSegment can be created."""
        segment = CriticalPathSegment(
            operation="matmul",
            duration_us=1000.0,
            category="compute",
            device="GPU",
            start_time_us=0.0,
            end_time_us=1000.0,
        )
        assert segment.category == "compute"
    
    def test_idle_time_breakdown_creation(self):
        """Test IdleTimeBreakdown can be created."""
        idle = IdleTimeBreakdown(
            total_idle_us=500.0,
            kernel_wait_us=200.0,
            memory_wait_us=150.0,
            sync_wait_us=100.0,
            other_idle_us=50.0,
            idle_percentage=10.0,
        )
        assert idle.total_idle_us == 500.0
    
    def test_comm_comp_overlap_creation(self):
        """Test CommCompOverlap can be created."""
        overlap = CommCompOverlap(
            total_compute_us=8000.0,
            total_comm_us=2000.0,
            overlapped_us=500.0,
            overlap_percentage=25.0,
            compute_exposed_us=7500.0,
            comm_exposed_us=1500.0,
        )
        assert overlap.overlap_percentage == 25.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



