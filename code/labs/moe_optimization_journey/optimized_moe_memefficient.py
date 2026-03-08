#!/usr/bin/env python3
"""Optimized MoE: Level 3 (Memory Efficient)."""

from labs.moe_optimization_journey.level3_memefficient import Level3MemEfficient


def get_benchmark() -> Level3MemEfficient:
    return Level3MemEfficient()


__all__ = ["Level3MemEfficient", "get_benchmark"]


