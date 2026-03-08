"""Integration test for bench_commands manifest persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from core.env import apply_env_defaults
apply_env_defaults()

from core.benchmark.bench_commands import _execute_benchmarks


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)


def _write_benchmark(path: Path) -> None:
    code = """\
import torch
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class SimpleMatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    allow_cpu = False

    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.weight = None
        self.output = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Keep the synthetic workload large enough that timing cross-validation
        # remains stable under the real harness.
        self.input = torch.randn(1024, 1024, device=self.device, dtype=torch.float16)
        self.weight = torch.randn(1024, 1024, device=self.device, dtype=torch.float16)

    def benchmark_fn(self) -> None:
        if self.input is None or self.weight is None:
            raise RuntimeError("Benchmark not initialized")
        self.output = self.input @ self.weight

    def capture_verification_payload(self) -> None:
        if self.input is None or self.weight is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.input, "weight": self.weight},
            output=self.output,
            batch_size=self.input.shape[0],
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(1e-3, 1e-3),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            full_device_sync=True,
            timing_method="wall_clock",
        )


def get_benchmark() -> BaseBenchmark:
    return SimpleMatmulBenchmark()
"""
    path.write_text(code, encoding="utf-8")


def test_bench_commands_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_root = tmp_path / "bench_root"
    chapter_dir = bench_root / "ch01"
    chapter_dir.mkdir(parents=True)

    _write_benchmark(chapter_dir / "baseline_manifest_demo.py")
    _write_benchmark(chapter_dir / "optimized_manifest_demo.py")

    artifacts_dir = tmp_path / "artifacts"
    # This test validates manifest persistence, not host isolation enforcement.
    # Allow foreign GPU workloads so an unrelated local vLLM service does not make
    # the integration test nondeterministic on shared development hosts.
    monkeypatch.setenv("AISP_ALLOW_FOREIGN_GPU_PROCESSES", "1")
    _execute_benchmarks(
        targets=["ch01:manifest_demo"],
        bench_root=bench_root,
        output_format="json",
        profile_type="none",
        validity_profile="portable",
        iterations=5,
        warmup=5,
        single_gpu=True,
        artifacts_dir=str(artifacts_dir),
    )

    manifest_files = list(artifacts_dir.rglob("manifest.json"))
    assert manifest_files, "manifest.json not written to artifacts directory"

    data = json.loads(manifest_files[0].read_text(encoding="utf-8"))
    manifests = data.get("manifests", [])
    assert manifests, "manifest.json missing per-run entries"

    variants = {entry.get("variant") for entry in manifests}
    assert "baseline" in variants
    assert "optimized" in variants

    hardware = manifests[0]["manifest"].get("hardware", {})
    assert hardware.get("gpu_app_clock_mhz") is not None
    assert hardware.get("memory_app_clock_mhz") is not None
