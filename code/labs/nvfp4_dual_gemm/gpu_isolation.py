"""GPU isolation helpers for strict benchmark runs.

The helpers rely on `nvidia-smi --query-compute-apps` and process-tree checks
to identify foreign GPU compute jobs and optionally terminate them.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComputeProc:
    pid: int
    process_name: str
    used_memory_mib: int
    cmdline: str


def _run_cmd(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def _children_of(pid: int) -> set[int]:
    children: set[int] = set()
    frontier = [int(pid)]
    while frontier:
        cur = frontier.pop()
        try:
            out = _run_cmd(["ps", "-o", "pid=", "--ppid", str(cur)])
        except Exception:
            continue
        for tok in out.split():
            try:
                child = int(tok)
            except ValueError:
                continue
            if child not in children:
                children.add(child)
                frontier.append(child)
    return children


def _allowed_pids(owner_pid: int | None) -> set[int]:
    if owner_pid is None:
        owner_pid = os.getpid()
    allowed = {int(owner_pid)}
    allowed.update(_children_of(int(owner_pid)))
    return allowed


def query_compute_processes() -> list[ComputeProc]:
    try:
        out = _run_cmd(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi query failed: {exc.output}") from exc

    if not out:
        return []

    rows: list[ComputeProc] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            used = int(parts[2])
        except ValueError:
            continue
        cmdline = ""
        try:
            cmdline = _run_cmd(["ps", "-p", str(pid), "-o", "args="]).strip()
        except Exception:
            cmdline = ""
        rows.append(ComputeProc(pid=pid, process_name=parts[1], used_memory_mib=used, cmdline=cmdline))
    return rows


def _matches_allowlist(cmdline: str, allow_cmd_substrings: list[str] | None) -> bool:
    if not allow_cmd_substrings:
        return False
    return any(s for s in allow_cmd_substrings if s and s in cmdline)


def _foreign_processes(owner_pid: int | None, allow_cmd_substrings: list[str] | None) -> list[ComputeProc]:
    allowed = _allowed_pids(owner_pid)
    procs = query_compute_processes()
    return [p for p in procs if p.pid not in allowed and not _matches_allowlist(p.cmdline, allow_cmd_substrings)]


def _kill_pids(pids: list[int]) -> list[int]:
    killed: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    return killed


def _format_processes(procs: list[ComputeProc]) -> str:
    if not procs:
        return "<none>"
    lines = ["pid process_name used_memory_mib cmdline"]
    for p in procs:
        lines.append(f"{p.pid} {p.process_name} {p.used_memory_mib} {p.cmdline}")
    return "\n".join(lines)


def ensure_gpu_isolation(
    *,
    owner_pid: int | None = None,
    kill_foreign: bool = False,
    require_idle: bool = False,
    settle_seconds: float = 1.0,
    context: str = "",
    allow_cmd_substrings: list[str] | None = None,
) -> dict[str, Any]:
    """Check/optionally enforce no foreign GPU compute jobs are active."""
    before = _foreign_processes(owner_pid, allow_cmd_substrings)
    killed: list[int] = []

    if kill_foreign and before:
        killed = _kill_pids([p.pid for p in before])
        time.sleep(max(0.0, float(settle_seconds)))

    after = _foreign_processes(owner_pid, allow_cmd_substrings)
    ok = len(after) == 0

    if require_idle and not ok:
        ctx = f" ({context})" if context else ""
        raise RuntimeError(
            "GPU isolation preflight failed"
            f"{ctx}: foreign compute processes detected.\n"
            f"before:\n{_format_processes(before)}\n"
            f"after:\n{_format_processes(after)}"
        )

    return {
        "context": context,
        "owner_pid": owner_pid,
        "kill_foreign": bool(kill_foreign),
        "require_idle": bool(require_idle),
        "allow_cmd_substrings": allow_cmd_substrings or [],
        "before": [p.__dict__ for p in before],
        "after": [p.__dict__ for p in after],
        "killed_pids": killed,
        "ok": ok,
    }
