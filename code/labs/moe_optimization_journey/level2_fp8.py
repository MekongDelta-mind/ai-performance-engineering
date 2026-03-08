#!/usr/bin/env python3
"""Level 2: FP8 Reduced Precision.

ADDS: Lower precision computation.
- Reduces memory bandwidth requirements
- Faster tensor operations
- Uses bfloat16 as FP8 approximation

Cumulative: batched + FP8
"""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2FP8(MoEJourneyBenchmark):
    """Level 2: + FP8 simulation."""
    LEVEL = 2

def get_benchmark() -> Level2FP8:
    return Level2FP8()


if __name__ == "__main__":
    run_level(2)
