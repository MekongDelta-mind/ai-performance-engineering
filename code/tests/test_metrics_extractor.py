"""Unit tests for metrics_extractor module."""

import subprocess
import sys
from pathlib import Path
import pytest

from core.profiling.metrics_extractor import (
    NsysMetrics,
    NcuMetrics,
    extract_nsys_metrics,
    extract_ncu_metrics,
    get_ncu_metric_description,
    _parse_nsys_csv,
    _parse_ncu_csv,
)


class TestNsysMetrics:
    """Tests for NsysMetrics dataclass."""
    
    def test_nsys_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NsysMetrics(
            total_gpu_time_ms=123.45,
            raw_metrics={"kernel_time": 100.0, "memory_throughput": 50.0}
        )
        result = metrics.to_dict()
        
        assert result["nsys_total_gpu_time_ms"] == 123.45
        assert result["nsys_kernel_time"] == 100.0
        assert result["nsys_memory_throughput"] == 50.0
    
    def test_nsys_metrics_empty(self):
        """Test empty metrics."""
        metrics = NsysMetrics()
        result = metrics.to_dict()
        
        assert len(result) == 0


class TestNcuMetrics:
    """Tests for NcuMetrics dataclass."""
    
    def test_ncu_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NcuMetrics(
            kernel_time_ms=10.5,
            sm_throughput_pct=85.0,
            dram_throughput_pct=60.0,
            l2_throughput_pct=70.0,
            occupancy_pct=90.0,
            raw_metrics={"tensor_cores": 100.0}
        )
        result = metrics.to_dict()
        
        assert result["ncu_kernel_time_ms"] == 10.5
        assert result["ncu_sm_throughput_pct"] == 85.0
        assert result["ncu_dram_throughput_pct"] == 60.0
        assert result["ncu_l2_throughput_pct"] == 70.0
        assert result["ncu_occupancy_pct"] == 90.0
        assert result["ncu_tensor_cores"] == 100.0
    
    def test_ncu_metrics_empty(self):
        """Test empty metrics."""
        metrics = NcuMetrics()
        result = metrics.to_dict()
        
        assert len(result) == 0


class TestParseNsysCsv:
    """Tests for nsys CSV parsing."""
    
    def test_parse_nsys_csv_with_total_gpu_time(self):
        """Test parsing nsys CSV with total GPU time."""
        csv_text = "Metric,Value\nTotal GPU Time,123.45"
        result = _parse_nsys_csv(csv_text)
        
        assert "nsys_total_gpu_time_ms" in result
        assert result["nsys_total_gpu_time_ms"] == 123.45
    
    def test_parse_nsys_csv_empty(self):
        """Test parsing empty CSV."""
        result = _parse_nsys_csv("")
        assert len(result) == 0
    
    def test_parse_nsys_csv_no_match(self):
        """Test parsing CSV without matching pattern."""
        csv_text = "Metric,Value\nSome Other Metric,100"
        result = _parse_nsys_csv(csv_text)
        assert len(result) == 0


class TestParseNcuCsv:
    """Tests for ncu CSV parsing."""
    
    def test_parse_ncu_csv_simple(self):
        """Test parsing simple ncu CSV."""
        csv_text = '"gpu__time_duration.avg","100.5"\n"sm__throughput.avg.pct_of_peak_sustained_elapsed","85.0"'
        result = _parse_ncu_csv(csv_text)
        
        assert "gpu__time_duration.avg" in result
        assert result["gpu__time_duration.avg"] == 100.5
        assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in result
        assert result["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 85.0
    
    def test_parse_ncu_csv_empty(self):
        """Test parsing empty CSV."""
        result = _parse_ncu_csv("")
        assert len(result) == 0
    
    def test_parse_ncu_csv_malformed(self):
        """Test parsing malformed CSV."""
        csv_text = "not,a,valid,csv"
        result = _parse_ncu_csv(csv_text)
        # Should handle gracefully without crashing
        assert isinstance(result, dict)


class TestGetNcuMetricDescription:
    """Tests for metric description lookup."""
    
    def test_get_known_metric_description(self):
        """Test getting description for known metric."""
        desc = get_ncu_metric_description("gpu__time_duration.avg")
        assert desc == "Kernel Execution Time"
    
    def test_get_unknown_metric_description(self):
        """Test getting description for unknown metric."""
        desc = get_ncu_metric_description("unknown_metric_id")
        # Should return cleaned version
        assert isinstance(desc, str)
        assert len(desc) > 0
    
    def test_get_clean_metric_name(self):
        """Test getting description for clean metric name."""
        desc = get_ncu_metric_description("ncu_sm_throughput_pct")
        # Should match to known metric
        assert "throughput" in desc.lower() or "SM" in desc


class TestExtractNsysMetrics:
    """Tests for nsys metrics extraction."""

    def test_extract_nsys_metrics_success(self, tmp_path):
        """Test successful nsys metrics extraction from a real nsys report."""
        script_path = tmp_path / "workload.py"
        script_path.write_text(
            "import torch\n"
            "x = torch.randn(512, 512, device='cuda')\n"
            "y = torch.randn(512, 512, device='cuda')\n"
            "z = x @ y\n"
            "torch.cuda.synchronize()\n"
            "print(float(z[0, 0]))\n"
        )

        out_base = tmp_path / "nsys_out"
        subprocess.run(
            [
                "nsys",
                "profile",
                "-t",
                "cuda,nvtx",
                "--force-overwrite",
                "true",
                "-o",
                str(out_base),
                sys.executable,
                str(script_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        nsys_path = tmp_path / "nsys_out.nsys-rep"
        assert nsys_path.exists()

        metrics = extract_nsys_metrics(nsys_path)
        assert metrics.total_gpu_time_ms is not None
        assert metrics.total_gpu_time_ms > 0.0

    def test_extract_nsys_metrics_file_not_found(self, tmp_path):
        """Test extraction when file doesn't exist."""
        nsys_path = tmp_path / "nonexistent.nsys-rep"
        metrics = extract_nsys_metrics(nsys_path)
        assert metrics.total_gpu_time_ms is None


class TestExtractNcuMetrics:
    """Tests for ncu metrics extraction."""

    def test_extract_ncu_metrics_success(self, tmp_path):
        """Test successful ncu metrics extraction from a real ncu report."""
        script_path = tmp_path / "workload.py"
        script_path.write_text(
            "import torch\n"
            "x = torch.randn(512, 512, device='cuda')\n"
            "y = torch.randn(512, 512, device='cuda')\n"
            "z = x @ y\n"
            "torch.cuda.synchronize()\n"
            "print(float(z[0, 0]))\n"
        )

        metrics = ",".join(
            [
                "gpu__time_duration.avg",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            ]
        )
        out_base = tmp_path / "ncu_out"
        subprocess.run(
            ["ncu", "--metrics", metrics, "--force-overwrite", "-o", str(out_base), sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        ncu_path = tmp_path / "ncu_out.ncu-rep"
        assert ncu_path.exists()

        metrics_obj = extract_ncu_metrics(ncu_path)
        assert metrics_obj.kernel_time_ms is not None
        assert metrics_obj.kernel_time_ms > 0.0
        assert metrics_obj.sm_throughput_pct is not None

    def test_extract_ncu_metrics_file_not_found(self, tmp_path):
        """Test extraction when file doesn't exist."""
        ncu_path = tmp_path / "nonexistent.ncu-rep"
        metrics_obj = extract_ncu_metrics(ncu_path)
        assert metrics_obj.kernel_time_ms is None

    def test_extract_ncu_metrics_companion_csv(self, tmp_path):
        """Test extraction from companion CSV file."""
        ncu_path = tmp_path / "test.ncu-rep"
        ncu_path.touch()

        csv_path = tmp_path / "test.csv"
        csv_path.write_text('"gpu__time_duration.avg","20.0"')

        metrics_obj = extract_ncu_metrics(ncu_path)
        assert metrics_obj.kernel_time_ms == 20.0


class TestGoldenFileMetrics:
    """Golden file tests for metric extraction with real sample data."""
    
    def test_nsys_csv_golden(self):
        """Golden test for nsys CSV parsing with realistic data from golden file."""
        # Read from actual golden file
        golden_file = Path(__file__).parent / "golden" / "nsys_stats_sample.csv"
        assert golden_file.exists(), f"Golden file not found: {golden_file}"
        
        nsys_csv = golden_file.read_text()
        result = _parse_nsys_csv(nsys_csv)
        
        # Verify expected metrics are extracted
        assert "nsys_total_gpu_time_ms" in result
        assert result["nsys_total_gpu_time_ms"] == 1234.56
        
        # Verify raw metrics contain other values
        assert len(result) > 0
    
    def test_ncu_csv_golden(self):
        """Golden test for ncu CSV parsing with realistic data from golden file."""
        # Read from actual golden file
        golden_file = Path(__file__).parent / "golden" / "ncu_stats_sample.csv"
        assert golden_file.exists(), f"Golden file not found: {golden_file}"
        
        ncu_csv = golden_file.read_text()
        result = _parse_ncu_csv(ncu_csv)
        
        # Verify expected metrics are extracted
        assert "gpu__time_duration.avg" in result
        assert result["gpu__time_duration.avg"] == 10.5
        assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in result
        assert result["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 90.0  # Last value wins
    
    def test_nsys_csv_edge_cases(self):
        """Test nsys CSV parsing with edge cases."""
        # Test with extra whitespace
        csv1 = "Metric,Value\n  Total GPU Time  ,  123.45  "
        result1 = _parse_nsys_csv(csv1)
        assert result1.get("nsys_total_gpu_time_ms") == 123.45
        
        # Test with different line endings
        csv2 = "Metric,Value\r\nTotal GPU Time,456.78"
        result2 = _parse_nsys_csv(csv2)
        assert result2.get("nsys_total_gpu_time_ms") == 456.78
        
        # Test with missing header
        csv3 = "Total GPU Time,789.01"
        result3 = _parse_nsys_csv(csv3)
        # Should handle gracefully
        assert isinstance(result3, dict)
    
    def test_ncu_csv_edge_cases(self):
        """Test ncu CSV parsing with edge cases."""
        # Test with escaped quotes
        csv1 = '"gpu__time_duration.avg","10.5"'
        result1 = _parse_ncu_csv(csv1)
        assert result1.get("gpu__time_duration.avg") == 10.5
        
        # Test with scientific notation
        csv2 = '"gpu__time_duration.avg","1.5e-3"'
        result2 = _parse_ncu_csv(csv2)
        assert result2.get("gpu__time_duration.avg") == 0.0015
        
        # Test with empty values
        csv3 = '"gpu__time_duration.avg",""'
        result3 = _parse_ncu_csv(csv3)
        # Should handle gracefully
        assert isinstance(result3, dict)
