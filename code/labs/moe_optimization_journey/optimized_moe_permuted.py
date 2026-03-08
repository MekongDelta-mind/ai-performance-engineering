#!/usr/bin/env python3
"""Optimized MoE: Level 2 (Token Permutation)."""

from labs.moe_optimization_journey.level2_permuted import Level2Permuted


def get_benchmark() -> Level2Permuted:
    return Level2Permuted()


__all__ = ["Level2Permuted", "get_benchmark"]


