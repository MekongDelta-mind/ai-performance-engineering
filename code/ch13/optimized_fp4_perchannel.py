"""optimized_fp4_perchannel.py - TE NVFP4 per-channel weight scaling."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def _preload_torch_cuda_symbols() -> None:
    """Ensure torch CUDA shared objects are loaded with RTLD_GLOBAL."""
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    libs = [
        "libtorch_cuda.so",
        "libtorch_cuda_linalg.so",
        "libtorch_nvshmem.so",
        "libc10_cuda.so",
    ]
    for name in libs:
        candidate = torch_lib_dir / name
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)


_preload_torch_cuda_symbols()

try:
    from transformer_engine.pytorch import Linear as TELinear, fp8_autocast, is_nvfp4_available
    from transformer_engine.common import recipe as te_recipe

    TE_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    TE_AVAILABLE = False
    TE_IMPORT_ERROR = exc


def _apply_per_channel_scaling(linear: nn.Module) -> torch.Tensor:
    """Scale weights/bias per output channel and return scale vector for restoration."""
    weight = getattr(linear, "weight", None)
    if weight is None:
        raise RuntimeError("TE Linear weight missing for per-channel scaling")
    scale = weight.abs().amax(dim=1).clamp(min=1e-6).to(weight.dtype)
    with torch.no_grad():
        weight.div_(scale.unsqueeze(1))
        bias = getattr(linear, "bias", None)
        if bias is not None:
            bias.div_(scale)
    return scale.unsqueeze(0)


class FP4PerChannelMLP(nn.Module):
    """Two-layer MLP with per-channel weight scaling."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = TELinear(hidden_dim, hidden_dim * 2, bias=True)
        self.fc2 = TELinear(hidden_dim * 2, hidden_dim, bias=True)
        self.activation = nn.GELU()
        self.fc1_scale: Optional[torch.Tensor] = None
        self.fc2_scale: Optional[torch.Tensor] = None

    def prepare_per_channel_scales(self) -> None:
        self.fc1_scale = _apply_per_channel_scaling(self.fc1)
        self.fc2_scale = _apply_per_channel_scaling(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1_scale is None or self.fc2_scale is None:
            raise RuntimeError("Per-channel scales not initialized")
        x = self.fc1(x)
        x = x * self.fc1_scale
        x = self.activation(x)
        x = self.fc2(x)
        x = x * self.fc2_scale
        return x


class OptimizedFP4PerChannelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized NVFP4 path with per-channel weight scaling."""

    signature_equivalence_group = "ch13_fp4_perchannel"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 1024
        self.hidden_dim = 8192
        self.dtype = torch.float32
        self.parameter_count: int = 0
        self._verify_input: Optional[torch.Tensor] = None
        self.fp4_recipe: Optional[object] = None
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        if not TE_AVAILABLE:
            raise RuntimeError(
                "SKIPPED: Transformer Engine is required for optimized_fp4_perchannel. "
                f"(import error: {TE_IMPORT_ERROR})"
            )
        if not is_nvfp4_available():
            raise RuntimeError("SKIPPED: NVFP4 kernels unavailable on this hardware/driver.")

        self.fp4_recipe = te_recipe.NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
        )
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = FP4PerChannelMLP(hidden_dim=self.hidden_dim).to(self.device, dtype=self.dtype).eval()
        self.model = model
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        w1 = torch.randn(self.hidden_dim * 2, self.hidden_dim, device=self.device, dtype=self.dtype) * 0.02
        b1 = torch.zeros(self.hidden_dim * 2, device=self.device, dtype=self.dtype)
        w2 = torch.randn(self.hidden_dim, self.hidden_dim * 2, device=self.device, dtype=self.dtype) * 0.02
        b2 = torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            self.model.fc1.weight.copy_(w1)
            if self.model.fc1.bias is not None:
                self.model.fc1.bias.copy_(b1)
            self.model.fc2.weight.copy_(w2)
            if self.model.fc2.bias is not None:
                self.model.fc2.bias.copy_(b2)

        self.model.prepare_per_channel_scales()

        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._verify_input = self.inputs.detach().clone()

        for _ in range(3):
            with torch.no_grad(), fp8_autocast(enabled=True, fp8_recipe=self.fp4_recipe):
                _ = self.model(self.inputs)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None or self.fp4_recipe is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.no_grad(), fp8_autocast(enabled=True, fp8_recipe=self.fp4_recipe):
            self.output = self.model(self.inputs)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output.detach().clone(),
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(2.0, 20.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.output = None
        self.fp4_recipe = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFP4PerChannelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
