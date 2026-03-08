#!/usr/bin/env python3
"""Optimized MoE: Level 2 (Sorted)."""

from labs.moe_optimization_journey.level2_sorted import Level2Sorted


def get_benchmark() -> Level2Sorted:
    return Level2Sorted()


__all__ = ["Level2Sorted", "get_benchmark"]
