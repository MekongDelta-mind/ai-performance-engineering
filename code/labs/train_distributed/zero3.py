"""Entry point for ZeRO-3 demos (baseline vs optimized)."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_zero3_multigpu as baseline_run
import labs.train_distributed.optimized_zero3_multigpu as optimized_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="optimized",
        help="Select which variant to run.",
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
