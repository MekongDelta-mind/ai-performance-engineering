#!/usr/bin/env python3
"""Optimized MoE: Level 4 (Grouped GEMM)."""

from labs.moe_optimization_journey.level4_grouped import Level4Grouped


def get_benchmark() -> Level4Grouped:
    return Level4Grouped()


__all__ = ["Level4Grouped", "get_benchmark"]
