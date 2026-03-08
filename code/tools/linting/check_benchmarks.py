#!/usr/bin/env python3
"""Thin wrapper to run the core benchmark linter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    pythonpath = [str(repo_root)]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath.extend(part for part in existing.split(os.pathsep) if part)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
    result = subprocess.run(
        [sys.executable, "-m", "core.scripts.linting.check_benchmarks", *sys.argv[1:]],
        env=env,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
