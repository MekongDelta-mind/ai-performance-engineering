#!/usr/bin/env python3
"""Optimized MoE: Level 6 (torch.compile - Full Optimization)."""

from labs.moe_optimization_journey.level6_compiled import Level6Compiled


def get_benchmark() -> Level6Compiled:
    return Level6Compiled()


__all__ = ["Level6Compiled", "get_benchmark"]
