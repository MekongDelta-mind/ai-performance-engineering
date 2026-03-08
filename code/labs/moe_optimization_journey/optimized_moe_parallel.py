#!/usr/bin/env python3
"""Optimized MoE: Level 4 (Expert Parallelism)."""

from labs.moe_optimization_journey.level4_parallel import Level4Parallel


def get_benchmark() -> Level4Parallel:
    return Level4Parallel()


__all__ = ["Level4Parallel", "get_benchmark"]
