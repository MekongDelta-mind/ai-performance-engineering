#!/usr/bin/env python3
"""Optimized MoE: Level 2 (FP8)."""

from labs.moe_optimization_journey.level2_fp8 import Level2FP8


def get_benchmark() -> Level2FP8:
    return Level2FP8()


__all__ = ["Level2FP8", "get_benchmark"]
