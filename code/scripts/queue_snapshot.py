#!/usr/bin/env python3
"""Capture active bench/profiling queue state for restore after reboot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Dict, List


QUEUE_MARKERS = (
    "wait_for_idle",
    "cli.aisp bench run",
    "cli/aisp.py bench run",
    "python -m cli.aisp bench run",
)

ACTIVE_MARKERS = (
    "cli.aisp bench run",
    "cli/aisp.py bench run",
    "python -m cli.aisp bench run",
    "torchrun_wrapper.py",
    "core.harness.torchrun_wrapper",
    "nsys profile",
    "ncu --force-overwrite",
)

RESTORE_MARKERS = (
    "cli.aisp bench run",
    "cli/aisp.py bench run",
    "python -m cli.aisp bench run",
)


def _parse_ps() -> List[Dict[str, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,command"],
        check=True,
        text=True,
        capture_output=True,
    )
    lines = result.stdout.splitlines()
    records: List[Dict[str, str]] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, cmd = parts
        records.append({"pid": pid, "ppid": ppid, "cmd": cmd})
    return records


def _is_queue_script(cmd: str) -> bool:
    if "bash" not in cmd and "sh " not in cmd:
        return False
    return any(marker in cmd for marker in QUEUE_MARKERS)


def _is_active(cmd: str) -> bool:
    return any(marker in cmd for marker in ACTIVE_MARKERS)


def _is_restore_candidate(cmd: str) -> bool:
    return any(marker in cmd for marker in RESTORE_MARKERS)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _default_output_path(cwd: str) -> str:
    timestamp = _utc_now().strftime("%Y%m%d_%H%M%S")
    filename = f"artifacts_tmp_queue_snapshot_{timestamp}.json"
    return os.path.join(cwd, filename)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write snapshot JSON (default: artifacts_tmp_queue_snapshot_<UTC>.json).",
    )
    args = parser.parse_args()

    cwd = os.getcwd()
    output_path = args.output or _default_output_path(cwd)

    records = _parse_ps()
    queue_scripts: List[Dict[str, str]] = []
    active_runs: List[Dict[str, str]] = []

    for record in records:
        cmd = record["cmd"]
        if _is_queue_script(cmd):
            queue_scripts.append(record)
            continue
        if _is_active(cmd):
            active_runs.append(record)

    restore_commands: List[str] = []
    seen = set()
    for record in queue_scripts:
        cmd = record["cmd"]
        if cmd not in seen:
            restore_commands.append(cmd)
            seen.add(cmd)

    for record in active_runs:
        cmd = record["cmd"]
        if _is_restore_candidate(cmd) and cmd not in seen:
            restore_commands.append(cmd)
            seen.add(cmd)

    snapshot = {
        "created_at_utc": _utc_now().isoformat(timespec="seconds"),
        "cwd": cwd,
        "queue_scripts": queue_scripts,
        "active_processes": active_runs,
        "restore_commands": restore_commands,
        "notes": [
            "Queue scripts are captured verbatim and can be restored via scripts/queue_restore.py.",
            "Active NCU/NSYS subprocesses often reference ephemeral /tmp scripts and are informational only.",
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
        f.write("\n")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
