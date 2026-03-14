#!/usr/bin/env python3
"""Render BenchmarkRun YAML from the shared repo template."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, cast

from core.benchmark.contracts_surface import render_benchmark_run_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overrides-json",
        default="{}",
        help="JSON object with BenchmarkRun generator overrides.",
    )
    parser.add_argument(
        "--yaml-only",
        action="store_true",
        help="Print only the rendered YAML instead of the full JSON result object.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        overrides = json.loads(args.overrides_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--overrides-json must be valid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise SystemExit("--overrides-json must decode to a JSON object")

    result = render_benchmark_run_yaml(cast(Dict[str, Any], overrides))
    if args.yaml_only:
        print(result["rendered_yaml"], end="")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
