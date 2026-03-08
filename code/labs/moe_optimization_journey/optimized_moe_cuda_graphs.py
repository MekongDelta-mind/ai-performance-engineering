#!/usr/bin/env python3
"""Optimized MoE: CUDA graphs.

Pairs with: baseline_moe.py

This wrapper must stay workload-equivalent with the baseline benchmark. Use the
MoEJourneyBenchmark implementation (Level 5) to keep parameter_count, inputs,
and verification semantics consistent across levels.
"""
from labs.moe_optimization_journey.level5_cudagraphs import Level5CUDAGraphs


def get_benchmark() -> Level5CUDAGraphs:
    return Level5CUDAGraphs()


__all__ = ["Level5CUDAGraphs", "get_benchmark"]

