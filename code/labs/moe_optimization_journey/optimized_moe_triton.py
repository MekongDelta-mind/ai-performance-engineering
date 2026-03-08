#!/usr/bin/env python3
"""Optimized MoE: Triton fused SiLU*up (Level 2).

Pairs with: baseline_moe.py

This wrapper must stay workload-equivalent with the baseline benchmark. Use the
MoEJourneyBenchmark implementation (Level 2) to keep parameter_count, inputs,
and verification semantics consistent.
"""
from labs.moe_optimization_journey.level2_fused import Level2Fused


def get_benchmark() -> Level2Fused:
    return Level2Fused()


__all__ = ["Level2Fused", "get_benchmark"]

