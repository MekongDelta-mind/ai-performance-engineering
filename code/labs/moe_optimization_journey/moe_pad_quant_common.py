"""Shared MoE pad+quant/finalize+slice helpers for benchmark pairs."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from labs.moe_optimization_journey.moe_model import ConfigurableMoEModel, MoEOptimizations, create_model


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def fake_quant_int8(x: torch.Tensor) -> torch.Tensor:
    """Simple per-token symmetric int8 fake quantization."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    # Compute quant/dequant in fp32 for stability, then cast back to preserve
    # model-wide dtype (the benchmark runs the full model in bf16).
    scale = (127.0 / amax).to(torch.float32)
    x_fp32 = x.to(torch.float32)
    x_int8 = torch.clamp((x_fp32 * scale).round(), -128, 127).to(torch.int8)
    dequant = x_int8.to(torch.float32) / scale
    return dequant.to(dtype=x.dtype)


class MoEPadQuantFinalize(nn.Module):
    """MoE wrapper that inserts pad+quant and finalize+slice stages."""

    def __init__(
        self,
        model: ConfigurableMoEModel,
        hidden_size: int,
        pad_multiple: int = 128,
    ) -> None:
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.pad_multiple = pad_multiple
        padded = _next_multiple(hidden_size, pad_multiple)
        self.padded_size = padded
        self.finalize = nn.Linear(padded, padded, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Run MoE blocks (same as ConfigurableMoEModel up to ln_f).
        x = self.model.embed(input_ids)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.ln_f(x)

        # Pad + quantize (MoE finalize path).
        pad_amount = self.padded_size - self.hidden_size
        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount))
        x = fake_quant_int8(x)
        x = self.finalize(x)
        # Finalize + slice back to original hidden size.
        x = x[..., : self.hidden_size]
        return self.model.lm_head(x)


def build_moe_pad_quant_model(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    vocab_size: int,
    num_layers: int = 1,
    num_heads: int = 8,
    level: int = 4,
) -> Tuple[MoEPadQuantFinalize, MoEOptimizations]:
    model, opts = create_model(
        level=level,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    return MoEPadQuantFinalize(model, hidden_size), opts
