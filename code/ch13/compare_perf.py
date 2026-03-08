#!/usr/bin/env python3

"""Compare two performance JSON files and print delta summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast
import sys


def load(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return cast(Dict[str, Any], json.load(f))


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: compare_perf.py baseline.json candidate.json")
        return 1

    baseline = load(Path(sys.argv[1]))
    candidate = load(Path(sys.argv[2]))

    metric = "tokens_per_sec"
    b = float(baseline.get(metric, 0.0))
    c = float(candidate.get(metric, 0.0))
    delta = c - b
    pct = (delta / b * 100.0) if b else 0.0

    print(f"Baseline: {b:.2f} {metric}")
    print(f"Candidate: {c:.2f} {metric}")
    print(f"Diff: {delta:+.2f} ({pct:+.1f}%)")
    return 0 if c >= b * 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
