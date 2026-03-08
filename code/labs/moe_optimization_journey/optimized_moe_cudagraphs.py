#!/usr/bin/env python3
"""Optimized MoE: Level 5 (CUDA Graphs)."""

from labs.moe_optimization_journey.level5_cudagraphs import Level5CUDAGraphs


def get_benchmark() -> Level5CUDAGraphs:
    return Level5CUDAGraphs()


__all__ = ["Level5CUDAGraphs", "get_benchmark"]


