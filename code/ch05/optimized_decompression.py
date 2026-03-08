"""optimized_decompression.py - GPU-assisted decompression stand-in.

Decodes the same toy RLE format as `baseline_decompression.py`, but performs the
repeat expansion on the GPU to model offloading decompression work.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin  # noqa: E402
from core.harness.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class GPUDecompressionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.counts: Optional[torch.Tensor] = None
        self.counts_i64: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(bytes_per_iteration=0.0)

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for GPU decompression demo")
        torch.manual_seed(42)
        total_len = 1024 * 1024
        run_len = 256
        if total_len % run_len != 0:
            raise RuntimeError("total_len must be divisible by run_len for this benchmark")
        num_runs = total_len // run_len
        counts = torch.full((num_runs,), run_len, dtype=torch.int32)
        values = torch.randn((num_runs,), dtype=torch.float32)
        self.counts = counts.to(self.device)
        self.counts_i64 = self.counts.to(torch.int64)
        self.values = values.to(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.counts_i64 is None or self.values is None:
            raise RuntimeError("SKIPPED: missing encoded RLE buffers")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("gpu_decompress_rle", enable=enable_nvtx):
            out = torch.repeat_interleave(self.values, self.counts_i64)
        latency_ms = self._record_stop(start)
        self.output = out.detach().clone()
        self._payload_counts = self.counts
        self._payload_values = self.values
        return {"latency_ms": latency_ms, "decompressed_len": int(out.numel())}

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        counts = self._payload_counts
        values = self._payload_values
        if counts is None or values is None:
            raise RuntimeError("benchmark_fn() must stash inputs for verification")
        self._set_verification_payload(
            inputs={"counts": counts.detach().clone(), "values": values.detach().clone()},
            output=self.output[:4096].detach().clone(),
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_storage_io_metrics
        return compute_storage_io_metrics(
            bytes_read=getattr(self, '_bytes_read', 0.0),
            bytes_written=getattr(self, '_bytes_written', 0.0),
            read_time_ms=getattr(self, '_read_time_ms', 1.0),
            write_time_ms=getattr(self, '_write_time_ms', 1.0),
        )


def get_benchmark() -> BaseBenchmark:
    return GPUDecompressionBenchmark()
