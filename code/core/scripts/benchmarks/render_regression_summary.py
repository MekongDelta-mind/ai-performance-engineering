#!/usr/bin/env python3
"""Render a tier-1 regression summary from two summary.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.analysis.regressions import compare_suite_summaries, render_regression_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render tier-1 regression summary markdown.")
    parser.add_argument("--current-summary", type=Path, required=True, help="Current tier-1 summary.json")
    parser.add_argument("--baseline-summary", type=Path, default=None, help="Previous tier-1 summary.json")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path")
    args = parser.parse_args()

    current = json.loads(args.current_summary.read_text(encoding="utf-8"))
    baseline = (
        json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        if args.baseline_summary
        else None
    )
    comparison = compare_suite_summaries(current, baseline)
    markdown = render_regression_summary(current, baseline, comparison)

    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(args.output)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
