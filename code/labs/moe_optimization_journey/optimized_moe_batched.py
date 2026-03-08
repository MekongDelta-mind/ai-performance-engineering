#!/usr/bin/env python3
"""Optimized MoE: Level 1 (Batched)."""

from labs.moe_optimization_journey.level1_batched import Level1Batched


def get_benchmark() -> Level1Batched:
    return Level1Batched()


__all__ = ["Level1Batched", "get_benchmark"]
