#!/usr/bin/env python3
"""Optimized MoE: Level 4 (CUDA Graphs)."""

from labs.moe_optimization_journey.level4_graphs import Level4Graphs


def get_benchmark() -> Level4Graphs:
    return Level4Graphs()


__all__ = ["Level4Graphs", "get_benchmark"]
