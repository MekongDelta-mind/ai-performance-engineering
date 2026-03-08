#!/usr/bin/env python3
"""Optimized MoE: Level 2 (Triton Fused)."""

from labs.moe_optimization_journey.level2_fused import Level2Fused


def get_benchmark() -> Level2Fused:
    return Level2Fused()


__all__ = ["Level2Fused", "get_benchmark"]


