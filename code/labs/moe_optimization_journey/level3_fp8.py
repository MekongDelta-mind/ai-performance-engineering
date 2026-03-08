#!/usr/bin/env python3
"""Level 3: FP8 Quantization.

ADDS: 8-bit floating point precision.
- Reduces memory bandwidth requirements
- Faster computation on supported hardware
- Uses Transformer Engine when available

Cumulative: batched + sorting + FP8
"""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level3FP8(MoEJourneyBenchmark):
    """Level 3: + FP8 quantization."""
    LEVEL = 3

def get_benchmark() -> Level3FP8:
    return Level3FP8()


if __name__ == "__main__":
    run_level(3)
