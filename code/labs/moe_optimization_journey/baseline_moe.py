#!/usr/bin/env python3
"""Baseline MoE: Level 0 (Naive)."""

from labs.moe_optimization_journey.level0_naive import Level0Naive


def get_benchmark() -> Level0Naive:
    return Level0Naive()


__all__ = ["Level0Naive", "get_benchmark"]
