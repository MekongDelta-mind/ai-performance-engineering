"""Benchmark wrapper for the occupancy_tuning CUDA binary."""

from __future__ import annotations
from typing import Optional

import os
from pathlib import Path

from core.benchmark.cuda_binary_benchmark import (  # noqa: E402
    ARCH_SUFFIX,
    CudaBinaryBenchmark,
    detect_supported_arch,
)


class OccupancyBinaryBenchmark(CudaBinaryBenchmark):
    """Wraps occupancy_tuning.cu so it runs under the standard harness."""

    def __init__(
        self,
        *,
        build_env: dict[str, str] | None = None,
        friendly_name: str,
        run_args: list[str] | None = None,
    ):
        chapter_dir = Path(__file__).parent
        args = run_args or [
            "--block-size",
            "32",
            "--smem-bytes",
            "45000",
            "--unroll",
            "1",
            "--inner-iters",
            "1",
            "--reps",
            "60",
        ]
        params = self._parse_run_args(args)
        signature_params = {k: params[k] for k in ("inner_iters", "reps") if k in params}
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="occupancy_tuning",
            friendly_name=friendly_name,
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            # Baseline: small block, heavy shared memory to depress occupancy.
            run_args=args,
            workload_params={"batch_size": 1, **signature_params},
            time_regex=r"avg_kernel_ms=([0-9]+\.?[0-9]*)",  # Parse kernel time from binary output.
        )
        self.build_env = build_env or {}
        self.register_workload_metadata(requests_per_iteration=1.0)

    @staticmethod
    def _parse_run_args(args: list[str]) -> dict[str, int]:
        """Convert CLI-style run args into explicit workload parameters."""
        if len(args) % 2 != 0:
            raise ValueError("run_args must contain flag/value pairs")
        params: dict[str, int] = {}
        for flag, value in zip(args[0::2], args[1::2]):
            if not flag.startswith("--"):
                raise ValueError(f"Expected flag starting with '--', got {flag}")
            key = flag.lstrip("-").replace("-", "_")
            try:
                params[key] = int(value)
            except ValueError:
                raise ValueError(f"run_args value for {flag} must be an integer, got {value}")
        return params

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)

    def _build_binary(self, verify_mode: bool = False) -> None:
        """Compile the executable with optional env overrides (e.g., MAXRREGCOUNT)."""
        self.arch = detect_supported_arch()
        suffix = ARCH_SUFFIX[self.arch]
        target = f"{self.binary_name}_verify{suffix}" if verify_mode else f"{self.binary_name}{suffix}"
        env = os.environ.copy()
        env.update(self.build_env)
        build_cmd = ["make", f"ARCH={self.arch}", target]
        if verify_mode:
            build_cmd.append("VERIFY=1")

        completed = self._run_make(build_cmd, env)
        path = self.chapter_dir / target
        if not path.exists():
            raise FileNotFoundError(f"Built binary not found at {path}")
        if verify_mode:
            self._verify_exec_path = path
        else:
            self.exec_path = path

    def _run_make(self, build_cmd, env):
        import subprocess

        try:
            completed = subprocess.run(
                build_cmd,
                cwd=self.chapter_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Build timeout: {' '.join(build_cmd)} exceeded 60 seconds")

        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to build {build_cmd[-1]} (arch={self.arch}).\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed


class BaselineOccupancyTuningBenchmark(OccupancyBinaryBenchmark):
    def __init__(self) -> None:
        super().__init__(friendly_name="Occupancy Tuning (baseline)")

    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for occupancy_tuning."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="occupancy_tuning",
        )


def get_benchmark() -> BaselineOccupancyTuningBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineOccupancyTuningBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
