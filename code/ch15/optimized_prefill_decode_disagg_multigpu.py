#!/usr/bin/env python3
"""optimized_prefill_decode_disagg_multigpu.py - Peer-copy + pipelined disaggregation.

Same semantic workload as the baseline:
- Prefill on even-index GPUs
- Decode on the next odd-index GPU

Optimizations:
- KV handoff uses direct GPU peer copy
- Decode overlaps prefill across GPU pairs
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from core.benchmark.gpu_requirements import require_peer_access
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch15.verification_payload_mixin import VerificationPayloadMixin


class OptimizedPrefillDecodeDisaggBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized prefill/decode disaggregation with peer KV handoff + pipelining."""

    multi_gpu_required = True

    def __init__(
        self,
        *,
        batch_size: int = 8,
        prefill_length: int = 1024,
        decode_length: int = 64,
        hidden_size: int = 2048,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.prefill_length = int(prefill_length)
        self.decode_length = int(decode_length)
        self.hidden_size = int(hidden_size)

        tokens = self.batch_size * (self.prefill_length + self.decode_length)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

        self.pairs: List[Tuple[int, int]] = []
        self.prefill_models: List[nn.Module] = []
        self.decode_models: List[nn.Module] = []
        self.prefill_inputs: List[torch.Tensor] = []
        self._verify_probe: Optional[torch.Tensor] = None
        self._output_shards: Optional[List[torch.Tensor]] = None
        self.output: Optional[torch.Tensor] = None

    def _resolve_pairs(self) -> List[Tuple[int, int]]:
        device_count = torch.cuda.device_count()
        if device_count < 2:
            raise RuntimeError("SKIPPED: prefill/decode disaggregation requires >=2 GPUs")
        if device_count % 2 != 0:
            raise RuntimeError(
                "SKIPPED: requires even GPU count for prefill/decode pairing; set CUDA_VISIBLE_DEVICES accordingly"
            )
        return [(idx, idx + 1) for idx in range(0, device_count, 2)]

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for prefill/decode disaggregation")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.pairs = self._resolve_pairs()
        for prefill_id, decode_id in self.pairs:
            require_peer_access(prefill_id, decode_id)

        num_pairs = len(self.pairs)
        if self.batch_size < num_pairs:
            self.batch_size = num_pairs
            tokens = self.batch_size * (self.prefill_length + self.decode_length)
            self._workload = WorkloadMetadata(
                requests_per_iteration=float(self.batch_size),
                tokens_per_iteration=float(tokens),
            )
            self.register_workload_metadata(
                requests_per_iteration=float(self.batch_size),
                tokens_per_iteration=float(tokens),
            )
        base = self.batch_size // num_pairs
        remainder = self.batch_size % num_pairs
        split_sizes = [base + (1 if idx < remainder else 0) for idx in range(num_pairs)]
        if base == 0 and remainder == 0:
            raise RuntimeError("batch_size must be >= number of GPU pairs")

        data_gen = torch.Generator().manual_seed(1234)
        cpu_inputs = torch.randn(
            self.batch_size,
            self.prefill_length,
            self.hidden_size,
            generator=data_gen,
            dtype=torch.bfloat16,
        )

        self.prefill_models = []
        self.decode_models = []
        self.prefill_inputs = []
        offset = 0
        for (prefill_id, decode_id), split_size in zip(self.pairs, split_sizes):
            prefill_device = torch.device(f"cuda:{prefill_id}")
            decode_device = torch.device(f"cuda:{decode_id}")
            prefill_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
                prefill_device, dtype=torch.bfloat16
            ).eval()
            decode_model = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(
                decode_device, dtype=torch.bfloat16
            ).eval()
            self.prefill_models.append(prefill_model)
            self.decode_models.append(decode_model)

            slice_end = offset + split_size
            batch_slice = cpu_inputs[offset:slice_end].to(prefill_device)
            self.prefill_inputs.append(batch_slice)
            offset = slice_end

        self._verify_probe = self.prefill_inputs[0][:1, :1, :256].detach()
        for prefill_id, decode_id in self.pairs:
            torch.cuda.synchronize(prefill_id)
            torch.cuda.synchronize(decode_id)

    def benchmark_fn(self) -> None:
        if not self.prefill_models or not self.decode_models or not self.prefill_inputs:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: List[torch.Tensor] = []
        with self._nvtx_range("optimized_prefill_decode_disagg_multigpu"):
            with torch.no_grad():
                for (prefill_id, decode_id), prefill_model, decode_model, batch in zip(
                    self.pairs,
                    self.prefill_models,
                    self.decode_models,
                    self.prefill_inputs,
                ):
                    decode_device = torch.device(f"cuda:{decode_id}")
                    for idx in range(batch.shape[0]):
                        prefill_out = prefill_model(batch[idx : idx + 1])
                        kv_decode = prefill_out.to(decode_device, non_blocking=True)
                        token_state = kv_decode[:, -1:, :]
                        for _ in range(self.decode_length):
                            token_state = decode_model(token_state)
                        outputs.append(token_state.squeeze(0).squeeze(0))

        for prefill_id, decode_id in self.pairs:
            torch.cuda.synchronize(prefill_id)
            torch.cuda.synchronize(decode_id)
        self._output_shards = outputs
        self.output = None

    def capture_verification_payload(self) -> None:
        if self._output_shards is None or self._verify_probe is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        param_count = sum(p.numel() for m in self.prefill_models for p in m.parameters()) + sum(
            p.numel() for m in self.decode_models for p in m.parameters()
        )
        selected = self._output_shards[:2]
        output_cpu = torch.stack([t.detach().cpu() for t in selected], dim=0)
        output_slice = output_cpu[:, :256].float().clone()
        self._set_verification_payload(
            inputs={"probe": self._verify_probe.detach().cpu()},
            output=output_slice,
            batch_size=int(self.batch_size),
            parameter_count=int(param_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        self.prefill_models = []
        self.decode_models = []
        self.prefill_inputs = []
        self._verify_probe = None
        self._output_shards = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5, multi_gpu_required=True)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    return OptimizedPrefillDecodeDisaggBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
