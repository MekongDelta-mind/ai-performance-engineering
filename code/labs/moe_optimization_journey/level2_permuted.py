#!/usr/bin/env python3
"""Level 2: Token Permutation.

ADDS: Sort tokens by expert for memory coalescing.
Based on ch19/mxfp8_moe_common.py bucket_by_expert() pattern.

- Groups tokens going to same expert together
- Better cache utilization
- Enables grouped GEMM in later levels

Cumulative: batched + permuted
"""
import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level2Permuted(MoEJourneyBenchmark):
    """Level 2: + Token permutation."""
    LEVEL = 2

def get_benchmark() -> Level2Permuted:
    return Level2Permuted()


if __name__ == "__main__":
    run_level(2)


