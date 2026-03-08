#!/usr/bin/env python3
"""Level 4: CUDA Graphs.

ADDS: Graph capture for reduced kernel launch overhead.
- Captures sequence of CUDA operations
- Replays with single launch
- Reduces CPU-GPU synchronization

Cumulative: batched + sorting + FP8 + CUDA graphs
"""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level4Graphs(MoEJourneyBenchmark):
    """Level 4: + CUDA graphs."""
    LEVEL = 4

def get_benchmark() -> Level4Graphs:
    return Level4Graphs()


if __name__ == "__main__":
    run_level(4)
