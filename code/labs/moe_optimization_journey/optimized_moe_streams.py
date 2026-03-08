#!/usr/bin/env python3
"""Optimized MoE: Level 2 (Multi-Stream)."""

from labs.moe_optimization_journey.level2_streams import Level2Streams


def get_benchmark() -> Level2Streams:
    return Level2Streams()


__all__ = ["Level2Streams", "get_benchmark"]


