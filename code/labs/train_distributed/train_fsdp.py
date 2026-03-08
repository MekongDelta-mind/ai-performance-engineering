"""Entry point for FSDP training demos (baseline vs optimized)."""

from __future__ import annotations

import argparse
import sys

import labs.train_distributed.baseline_fsdp as baseline_single
import labs.train_distributed.baseline_fsdp_multigpu as baseline_multi
import labs.train_distributed.optimized_fsdp as optimized_single
import labs.train_distributed.optimized_fsdp_multigpu as optimized_multi


def main():
    parser = argparse.ArgumentParser(description="FSDP training examples.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="optimized",
        help="Which variant to execute.",
    )
    parser.add_argument(
        "--variant",
        choices=["single", "multigpu"],
        default="multigpu",
        help="Which workload size to execute.",
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.variant == "single":
        run = baseline_single if args.mode == "baseline" else optimized_single
    else:
        run = baseline_multi if args.mode == "baseline" else optimized_multi
    run.main()


if __name__ == "__main__":
    main()
