#!/usr/bin/env python3
"""Level 7: torch.compile on the baseline MoE workload.

This reuses the shared MoEJourneyBenchmark stack but enables the Level 7
optimization flag so the model is created with torch.compile (mode="max-autotune")
on the same batch/seq/hidden dimensions as the baseline.
"""

import torch

from labs.moe_optimization_journey.moe_benchmark import MoEJourneyBenchmark, run_level


class Level7Compiled(MoEJourneyBenchmark):
    """torch.compile applied to the baseline MoE workload."""

    LEVEL = 7

def get_benchmark() -> MoEJourneyBenchmark:
    return Level7Compiled()


if __name__ == "__main__":
    run_level(7)
