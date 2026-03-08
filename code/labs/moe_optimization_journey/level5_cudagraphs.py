#!/usr/bin/env python3
"""Level 5: CUDA Graphs.

ADDS: Capture kernel sequence for replay with minimal overhead.

CUDA Graphs eliminate:
- Kernel launch latency
- CPU-GPU synchronization overhead
- Python interpreter overhead

Note: Requires static shapes for graph capture.

Cumulative: batched + fused + mem_efficient + grouped + cuda_graphs
"""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level5CUDAGraphs(MoEJourneyBenchmark):
    """Level 5: + CUDA Graphs."""
    LEVEL = 5

def get_benchmark() -> Level5CUDAGraphs:
    return Level5CUDAGraphs()


if __name__ == "__main__":
    run_level(5)


