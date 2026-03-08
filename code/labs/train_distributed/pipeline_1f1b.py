"""Dispatcher for baseline vs optimized 1F1B demos (single-GPU simulation)."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_pipeline_1f1b as baseline_run
import labs.train_distributed.optimized_pipeline_1f1b as optimized_run


def main():
    parser = argparse.ArgumentParser(description="1F1B toy pipeline (single-GPU simulation).")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
