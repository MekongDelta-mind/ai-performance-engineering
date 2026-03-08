#!/usr/bin/env python3
"""
Unit tests for the Differential Profile Analyzer.

Tests cover:
- Loading and parsing deep profile JSONs
- Kernel-level differential analysis
- Binding shift detection
- Improvement attribution
- Markdown report generation
"""

import json
import tempfile
from pathlib import Path

import pytest

from core.analysis.differential_profile_analyzer import (
    analyze_differential,
    generate_markdown_report,
    KernelDiff,
    ImprovementAttribution,
    DifferentialReport,
    _determine_binding,
    _compute_improvement_attribution,
    _generate_key_improvements,
    _generate_remaining_bottlenecks,
    _generate_next_steps,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def baseline_profile_data():
    """Sample baseline deep profile data."""
    return {
        "advisories": [
            {
                "kernel": "matmul_kernel",
                "precision": "fp16",
                "duration_ms": 10.5,
                "flops": 1e12,
                "bytes_transferred": 1e9,
                "roofline": {
                    "achieved_tflops": 50.0,
                    "achieved_bandwidth_gbs": 500.0,
                    "arithmetic_intensity": 10.0,
                    "compute_utilization_pct": 30.0,
                    "memory_utilization_pct": 80.0,
                    "binding": "memory",
                    "is_memory_bound": True,
                    "is_compute_bound": False,
                    "is_tmem_bound": False,
                    "ridge_point": 100.0,
                    "memory_bound_limit_tflops": 50.0,
                    "peak_tflops": 200.0,
                    "peak_bandwidth_gbs": 2000.0,
                },
                "sm_util_pct": 30.0,
                "dram_util_pct": 80.0,
                "tmem_util_pct": None,
                "occupancy_pct": 40.0,
                "tensor_util_pct": 20.0,
                "warp_execution_pct": 75.0,
                "l2_hit_pct": 50.0,
                "recommendations": [
                    "Kernel is memory-bound; focus on increasing arithmetic intensity.",
                ],
            },
            {
                "kernel": "softmax_kernel",
                "precision": "fp16",
                "duration_ms": 2.0,
                "flops": 1e10,
                "bytes_transferred": 1e8,
                "roofline": {
                    "achieved_tflops": 5.0,
                    "achieved_bandwidth_gbs": 400.0,
                    "arithmetic_intensity": 5.0,
                    "compute_utilization_pct": 10.0,
                    "memory_utilization_pct": 60.0,
                    "binding": "memory",
                    "is_memory_bound": True,
                    "is_compute_bound": False,
                    "is_tmem_bound": False,
                    "ridge_point": 100.0,
                    "memory_bound_limit_tflops": 10.0,
                    "peak_tflops": 200.0,
                    "peak_bandwidth_gbs": 2000.0,
                },
                "sm_util_pct": 10.0,
                "dram_util_pct": 60.0,
                "tmem_util_pct": None,
                "occupancy_pct": 30.0,
                "tensor_util_pct": None,
                "warp_execution_pct": 80.0,
                "l2_hit_pct": 40.0,
                "recommendations": ["Low occupancy; increase block size."],
            },
        ],
        "stats": {
            "kernel_count": 2,
            "memory_bound_kernels": ["matmul_kernel", "softmax_kernel"],
            "compute_bound_kernels": [],
            "tmem_bound_kernels": [],
        },
    }


@pytest.fixture
def optimized_profile_data():
    """Sample optimized deep profile data showing improvements."""
    return {
        "advisories": [
            {
                "kernel": "matmul_kernel",
                "precision": "fp16",
                "duration_ms": 5.0,  # 2x faster
                "flops": 1e12,
                "bytes_transferred": 5e8,  # 50% less memory traffic
                "roofline": {
                    "achieved_tflops": 100.0,
                    "achieved_bandwidth_gbs": 400.0,
                    "arithmetic_intensity": 20.0,  # 2x higher AI
                    "compute_utilization_pct": 60.0,  # Better utilization
                    "memory_utilization_pct": 50.0,
                    "binding": "compute",  # Shifted to compute-bound
                    "is_memory_bound": False,
                    "is_compute_bound": True,
                    "is_tmem_bound": False,
                    "ridge_point": 100.0,
                    "memory_bound_limit_tflops": 100.0,
                    "peak_tflops": 200.0,
                    "peak_bandwidth_gbs": 2000.0,
                },
                "sm_util_pct": 60.0,
                "dram_util_pct": 50.0,
                "tmem_util_pct": None,
                "occupancy_pct": 70.0,
                "tensor_util_pct": 50.0,
                "warp_execution_pct": 90.0,
                "l2_hit_pct": 80.0,
                "recommendations": [
                    "Kernel is compute-bound; pursue higher SM utilisation.",
                ],
            },
            {
                "kernel": "softmax_kernel",
                "precision": "fp16",
                "duration_ms": 1.0,  # 2x faster
                "flops": 1e10,
                "bytes_transferred": 5e7,
                "roofline": {
                    "achieved_tflops": 10.0,
                    "achieved_bandwidth_gbs": 350.0,
                    "arithmetic_intensity": 8.0,
                    "compute_utilization_pct": 20.0,
                    "memory_utilization_pct": 45.0,
                    "binding": "memory",
                    "is_memory_bound": True,
                    "is_compute_bound": False,
                    "is_tmem_bound": False,
                    "ridge_point": 100.0,
                    "memory_bound_limit_tflops": 16.0,
                    "peak_tflops": 200.0,
                    "peak_bandwidth_gbs": 2000.0,
                },
                "sm_util_pct": 20.0,
                "dram_util_pct": 45.0,
                "tmem_util_pct": None,
                "occupancy_pct": 50.0,
                "tensor_util_pct": None,
                "warp_execution_pct": 85.0,
                "l2_hit_pct": 60.0,
                "recommendations": ["Continue improving cache utilization."],
            },
        ],
        "stats": {
            "kernel_count": 2,
            "memory_bound_kernels": ["softmax_kernel"],
            "compute_bound_kernels": ["matmul_kernel"],
            "tmem_bound_kernels": [],
        },
    }


@pytest.fixture
def baseline_json_file(baseline_profile_data, tmp_path):
    """Create a temporary baseline JSON file."""
    path = tmp_path / "baseline_deep_profile.json"
    path.write_text(json.dumps(baseline_profile_data, indent=2))
    return path


@pytest.fixture
def optimized_json_file(optimized_profile_data, tmp_path):
    """Create a temporary optimized JSON file."""
    path = tmp_path / "optimized_deep_profile.json"
    path.write_text(json.dumps(optimized_profile_data, indent=2))
    return path


# ============================================================================
# Unit Tests: Helper Functions
# ============================================================================

class TestDetermineBinding:
    """Tests for _determine_binding helper."""
    
    def test_memory_bound(self):
        advisory = {
            "roofline": {
                "is_tmem_bound": False,
                "is_memory_bound": True,
                "is_compute_bound": False,
            }
        }
        assert _determine_binding(advisory) == "memory-bound"
    
    def test_compute_bound(self):
        advisory = {
            "roofline": {
                "is_tmem_bound": False,
                "is_memory_bound": False,
                "is_compute_bound": True,
            }
        }
        assert _determine_binding(advisory) == "compute-bound"
    
    def test_tmem_bound(self):
        advisory = {
            "roofline": {
                "is_tmem_bound": True,
                "is_memory_bound": False,
                "is_compute_bound": False,
            }
        }
        assert _determine_binding(advisory) == "tmem-bound"
    
    def test_binding_from_string(self):
        advisory = {
            "roofline": {
                "binding": "l2",
            }
        }
        assert _determine_binding(advisory) == "l2-bound"
    
    def test_empty_advisory(self):
        assert _determine_binding({}) == "unknown"
    
    def test_no_roofline(self):
        advisory = {"kernel": "test"}
        assert _determine_binding(advisory) == "unknown"


class TestComputeImprovementAttribution:
    """Tests for _compute_improvement_attribution."""
    
    def test_no_improvements(self):
        kernel_diffs = [
            KernelDiff(
                name="kernel1",
                baseline_time_ms=1.0,
                optimized_time_ms=1.0,
                speedup=1.0,
                baseline_binding="memory-bound",
                optimized_binding="memory-bound",
                binding_changed=False,
                sm_util_delta=0.0,
                dram_util_delta=0.0,
                occupancy_delta=0.0,
                l2_hit_delta=0.0,
                arithmetic_intensity_delta=0.0,
                primary_improvement="no change",
            )
        ]
        attr = _compute_improvement_attribution([], [], kernel_diffs)
        assert attr.other == 1.0  # All attributed to "other"
    
    def test_compute_improvement(self):
        kernel_diffs = [
            KernelDiff(
                name="kernel1",
                baseline_time_ms=2.0,
                optimized_time_ms=1.0,
                speedup=2.0,
                baseline_binding="memory-bound",
                optimized_binding="compute-bound",
                binding_changed=True,
                sm_util_delta=30.0,  # Big SM improvement
                dram_util_delta=-20.0,
                occupancy_delta=20.0,
                l2_hit_delta=10.0,
                arithmetic_intensity_delta=5.0,
                primary_improvement="improved compute utilization",
            )
        ]
        attr = _compute_improvement_attribution([], [], kernel_diffs)
        # Should attribute some to compute
        assert attr.improved_compute_utilization > 0


class TestGenerateKeyImprovements:
    """Tests for _generate_key_improvements."""
    
    def test_with_binding_shift(self):
        kernel_diffs = []
        improvements = _generate_key_improvements(kernel_diffs, "memory-bound → compute-bound")
        assert any("memory-bound → compute-bound" in imp for imp in improvements)
    
    def test_with_improved_kernels(self):
        kernel_diffs = [
            KernelDiff(
                name="fast_kernel",
                baseline_time_ms=10.0,
                optimized_time_ms=5.0,
                speedup=2.0,
                baseline_binding="memory-bound",
                optimized_binding="compute-bound",
                binding_changed=True,
                sm_util_delta=None,
                dram_util_delta=None,
                occupancy_delta=None,
                l2_hit_delta=None,
                arithmetic_intensity_delta=None,
                primary_improvement="binding shift",
            )
        ]
        improvements = _generate_key_improvements(kernel_diffs, None)
        assert any("fast_kernel" in imp for imp in improvements)
        assert any("2.00x" in imp for imp in improvements)
    
    def test_minimal_improvements(self):
        improvements = _generate_key_improvements([], None)
        assert "Minimal improvements detected" in improvements


class TestGenerateRemainingBottlenecks:
    """Tests for _generate_remaining_bottlenecks."""
    
    def test_low_sm_utilization(self):
        advisories = [{"kernel": "test", "sm_util_pct": 30.0}]
        bottlenecks = _generate_remaining_bottlenecks(advisories)
        assert any("SM utilization" in b for b in bottlenecks)
    
    def test_high_dram_utilization(self):
        advisories = [{"kernel": "test", "dram_util_pct": 90.0}]
        bottlenecks = _generate_remaining_bottlenecks(advisories)
        assert any("HBM saturated" in b for b in bottlenecks)
    
    def test_low_occupancy(self):
        advisories = [{"kernel": "test", "occupancy_pct": 25.0}]
        bottlenecks = _generate_remaining_bottlenecks(advisories)
        assert any("occupancy" in b.lower() for b in bottlenecks)
    
    def test_no_bottlenecks(self):
        advisories = [
            {
                "kernel": "optimal",
                "sm_util_pct": 80.0,
                "dram_util_pct": 50.0,
                "occupancy_pct": 70.0,
                "l2_hit_pct": 80.0,
            }
        ]
        bottlenecks = _generate_remaining_bottlenecks(advisories)
        assert "No major bottlenecks" in bottlenecks[0]


class TestGenerateNextSteps:
    """Tests for _generate_next_steps."""
    
    def test_memory_bound_next_steps(self):
        kernel_diffs = [
            KernelDiff(
                name="mem_kernel",
                baseline_time_ms=1.0,
                optimized_time_ms=1.0,
                speedup=1.0,
                baseline_binding="memory-bound",
                optimized_binding="memory-bound",
                binding_changed=False,
                sm_util_delta=None,
                dram_util_delta=None,
                occupancy_delta=None,
                l2_hit_delta=None,
                arithmetic_intensity_delta=None,
                primary_improvement="no change",
            )
        ]
        steps = _generate_next_steps([], kernel_diffs, 1.2)
        assert any("arithmetic intensity" in s.lower() for s in steps)
    
    def test_low_speedup_suggestion(self):
        steps = _generate_next_steps([], [], 1.2)
        assert any("algorithmic" in s.lower() for s in steps)
    
    def test_high_speedup_encouragement(self):
        steps = _generate_next_steps([], [], 2.5)
        assert any("Good progress" in s for s in steps)


# ============================================================================
# Integration Tests: Full Analysis
# ============================================================================

class TestAnalyzeDifferential:
    """Integration tests for analyze_differential."""
    
    def test_full_analysis(self, baseline_json_file, optimized_json_file):
        """Test complete differential analysis."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        
        # Check overall metrics
        assert report is not None
        assert report.overall_speedup > 1.0  # Should show improvement
        assert report.total_baseline_time_ms > report.total_optimized_time_ms
        
        # Check binding shift detection
        assert report.baseline_dominant_binding == "memory-bound"
        # Optimized has 1 memory-bound, 1 compute-bound, so could be either
        
        # Check kernel diffs
        assert len(report.kernel_diffs) == 2
        matmul_diff = next(kd for kd in report.kernel_diffs if kd.name == "matmul_kernel")
        assert matmul_diff.speedup == pytest.approx(2.1, rel=0.1)
        assert matmul_diff.binding_changed
        
    def test_speedup_calculation(self, baseline_json_file, optimized_json_file):
        """Test that speedup is calculated correctly."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        
        # Baseline: 10.5 + 2.0 = 12.5 ms
        # Optimized: 5.0 + 1.0 = 6.0 ms
        # Speedup: 12.5 / 6.0 ≈ 2.08x
        expected_speedup = 12.5 / 6.0
        assert report.overall_speedup == pytest.approx(expected_speedup, rel=0.01)
    
    def test_kernel_diff_metrics(self, baseline_json_file, optimized_json_file):
        """Test kernel-level metric deltas."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        
        matmul_diff = next(kd for kd in report.kernel_diffs if kd.name == "matmul_kernel")
        
        # SM util: 60 - 30 = 30
        assert matmul_diff.sm_util_delta == pytest.approx(30.0, rel=0.01)
        
        # DRAM util: 50 - 80 = -30
        assert matmul_diff.dram_util_delta == pytest.approx(-30.0, rel=0.01)
        
        # L2 hit: 80 - 50 = 30
        assert matmul_diff.l2_hit_delta == pytest.approx(30.0, rel=0.01)
        
        # AI: 20 - 10 = 10
        assert matmul_diff.arithmetic_intensity_delta == pytest.approx(10.0, rel=0.01)
    
    def test_missing_baseline_file(self, optimized_json_file, tmp_path):
        """Test handling of missing baseline file."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            analyze_differential(missing, optimized_json_file)
    
    def test_missing_optimized_file(self, baseline_json_file, tmp_path):
        """Test handling of missing optimized file."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            analyze_differential(baseline_json_file, missing)


class TestGenerateMarkdownReport:
    """Tests for markdown report generation."""
    
    def test_report_structure(self, baseline_json_file, optimized_json_file):
        """Test that markdown report has expected sections."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        markdown = generate_markdown_report(report)
        
        # Check for key sections
        assert "# Differential Profile Analysis" in markdown
        assert "## Summary" in markdown
        assert "## Improvement Attribution" in markdown
        assert "## Key Improvements" in markdown
        assert "## Kernel-Level Analysis" in markdown
        assert "## Remaining Bottlenecks" in markdown
        assert "## Recommended Next Steps" in markdown
    
    def test_report_contains_speedup(self, baseline_json_file, optimized_json_file):
        """Test that speedup is shown in report."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        markdown = generate_markdown_report(report)
        
        # Should show ~2.08x speedup
        assert "2.0" in markdown or "2.1" in markdown
    
    def test_report_contains_kernel_names(self, baseline_json_file, optimized_json_file):
        """Test that kernel names appear in report."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        markdown = generate_markdown_report(report)
        
        assert "matmul_kernel" in markdown
        assert "softmax_kernel" in markdown
    
    def test_report_shows_binding_changes(self, baseline_json_file, optimized_json_file):
        """Test that binding changes are shown."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        markdown = generate_markdown_report(report)
        
        # matmul_kernel should show binding change
        assert "→" in markdown or "->" in markdown


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_advisories(self, tmp_path):
        """Test with empty advisories list."""
        baseline = tmp_path / "baseline.json"
        optimized = tmp_path / "optimized.json"
        
        baseline.write_text(json.dumps({"advisories": [], "stats": {}}))
        optimized.write_text(json.dumps({"advisories": [], "stats": {}}))
        
        report = analyze_differential(baseline, optimized)
        assert report.overall_speedup == 1.0
        assert len(report.kernel_diffs) == 0
    
    def test_mismatched_kernels(self, tmp_path):
        """Test when baseline and optimized have different kernels."""
        baseline = tmp_path / "baseline.json"
        optimized = tmp_path / "optimized.json"
        
        baseline.write_text(json.dumps({
            "advisories": [
                {"kernel": "kernel_a", "duration_ms": 5.0, "roofline": None}
            ],
            "stats": {},
        }))
        optimized.write_text(json.dumps({
            "advisories": [
                {"kernel": "kernel_b", "duration_ms": 3.0, "roofline": None}
            ],
            "stats": {},
        }))
        
        report = analyze_differential(baseline, optimized)
        # Should have both kernels in diffs
        kernel_names = [kd.name for kd in report.kernel_diffs]
        assert "kernel_a" in kernel_names
        assert "kernel_b" in kernel_names
    
    def test_kernel_eliminated(self, tmp_path):
        """Test when a kernel is eliminated in optimized version."""
        baseline = tmp_path / "baseline.json"
        optimized = tmp_path / "optimized.json"
        
        baseline.write_text(json.dumps({
            "advisories": [
                {"kernel": "eliminated_kernel", "duration_ms": 10.0, "roofline": None}
            ],
            "stats": {},
        }))
        optimized.write_text(json.dumps({
            "advisories": [],
            "stats": {},
        }))
        
        report = analyze_differential(baseline, optimized)
        eliminated_diff = next(kd for kd in report.kernel_diffs if kd.name == "eliminated_kernel")
        assert eliminated_diff.speedup == float('inf')
    
    def test_new_kernel_added(self, tmp_path):
        """Test when a new kernel appears in optimized version."""
        baseline = tmp_path / "baseline.json"
        optimized = tmp_path / "optimized.json"
        
        baseline.write_text(json.dumps({
            "advisories": [],
            "stats": {},
        }))
        optimized.write_text(json.dumps({
            "advisories": [
                {"kernel": "new_kernel", "duration_ms": 5.0, "roofline": None}
            ],
            "stats": {},
        }))
        
        report = analyze_differential(baseline, optimized)
        new_diff = next(kd for kd in report.kernel_diffs if kd.name == "new_kernel")
        assert new_diff.speedup == 0.0  # New kernel = regression (no baseline)
    
    def test_null_duration(self, tmp_path):
        """Test with null duration values."""
        baseline = tmp_path / "baseline.json"
        optimized = tmp_path / "optimized.json"
        
        baseline.write_text(json.dumps({
            "advisories": [
                {"kernel": "test", "duration_ms": None, "roofline": None}
            ],
            "stats": {},
        }))
        optimized.write_text(json.dumps({
            "advisories": [
                {"kernel": "test", "duration_ms": None, "roofline": None}
            ],
            "stats": {},
        }))
        
        report = analyze_differential(baseline, optimized)
        assert report is not None


# ============================================================================
# Test Report Serialization
# ============================================================================

class TestReportSerialization:
    """Tests for report serialization."""
    
    def test_to_dict(self, baseline_json_file, optimized_json_file):
        """Test that report can be serialized to dict."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        report_dict = report.to_dict()
        
        assert "overall_speedup" in report_dict
        assert "kernel_diffs" in report_dict
        assert "improvement_attribution" in report_dict
        assert "key_improvements" in report_dict
    
    def test_to_json(self, baseline_json_file, optimized_json_file):
        """Test that report can be serialized to JSON."""
        report = analyze_differential(baseline_json_file, optimized_json_file)
        report_dict = report.to_dict()
        
        # Should not raise
        json_str = json.dumps(report_dict, indent=2)
        assert len(json_str) > 0
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["overall_speedup"] == pytest.approx(report.overall_speedup, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



