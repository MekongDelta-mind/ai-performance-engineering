"""Dispatcher for GPipe vs DualPipe schedule comparison (multi-GPU)."""

from __future__ import annotations

import argparse
import sys

from core.benchmark.gpu_requirements import require_min_gpus

import labs.train_distributed.baseline_pipeline_gpipe_to_dualpipe_multigpu as baseline_run
import labs.train_distributed.optimized_pipeline_gpipe_to_dualpipe_multigpu as optimized_run


def main():
    require_min_gpus(2, script_name="pipeline_gpipe_to_dualpipe_multigpu.py")
    parser = argparse.ArgumentParser(description="GPipe vs DualPipe schedule comparison.")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    if args.mode == "baseline":
        baseline_run.main()
    else:
        optimized_run.main()


if __name__ == "__main__":
    main()
