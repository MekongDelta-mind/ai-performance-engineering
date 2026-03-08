#!/usr/bin/env python3
"""Lock GPU clocks (strict) and run a command.

This is intended for cluster evaluation scripts where we require stable clocks.
It uses the repo harness `lock_gpu_clocks()` (which uses `sudo -n nvidia-smi`
when available) and fails fast if clock locking cannot be acquired.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[2]
    _env = os.environ.copy()
    _pythonpath = _env.get("PYTHONPATH")
    _env["PYTHONPATH"] = str(_repo_root) if not _pythonpath else os.pathsep.join([str(_repo_root), _pythonpath])
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "cluster.scripts.lock_and_run", *sys.argv[1:]],
        _env,
    )

from core.harness.benchmark_harness import lock_gpu_clocks, _resolve_physical_device_index  # type: ignore


def _comma_ints(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _app_clock_snapshot(physical_device_index: int) -> Dict[str, Any]:
    try:
        import pynvml
    except ImportError as exc:
        return {"error": f"pynvml import failed: {exc}"}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_index)
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
        cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        return {
            "applications_sm_mhz": app_sm,
            "applications_mem_mhz": app_mem,
            "current_sm_mhz": cur_sm,
            "current_mem_mhz": cur_mem,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _print_lock_failure_help() -> None:
    msg = (
        "ERROR: GPU clock lock is required but was not acquired.\n"
        "\n"
        "Fix:\n"
        "  1) Ensure `sudo -n true` works for this user (passwordless sudo).\n"
        "  2) Ensure `nvidia-smi` supports clock locking on this system.\n"
        "  3) Re-run the command.\n"
    )
    print(msg, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lock GPU clocks and run a command (strict by default).")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", ""), help="Run id for structured metadata")
    parser.add_argument("--label", default=os.environ.get("LABEL", ""), help="Label for structured metadata")
    parser.add_argument(
        "--devices",
        default="",
        help="Comma-separated logical CUDA device indices to lock (default: all visible devices).",
    )
    parser.add_argument(
        "--lock-meta-out",
        default="",
        help="Optional path to write lock metadata JSON (includes per-device app clocks).",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run (prefix with --).")
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided. Use: lock_and_run.py -- <command> [args...]")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; cannot lock GPU clocks.")

    require_lock = True

    if args.devices:
        devices = _comma_ints(args.devices)
    else:
        devices = list(range(torch.cuda.device_count()))

    if not devices:
        raise SystemExit("No CUDA devices selected.")

    cmd_str = shlex.join(cmd)
    run_meta: Dict[str, Any] = {
        "run_id": args.run_id,
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "require_clock_lock": True,
        "devices": devices,
        "cmd": cmd_str,
        "locks": [],
    }

    out_path: Optional[Path] = None
    if args.lock_meta_out:
        out_path = Path(args.lock_meta_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_meta() -> None:
        if out_path is None:
            return
        out_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    with ExitStack() as stack:
        for device in devices:
            theoretical_tflops, theoretical_mem_gbps = stack.enter_context(lock_gpu_clocks(device=device))
            locked = bool(theoretical_tflops) or bool(theoretical_mem_gbps)
            physical = _resolve_physical_device_index(device)
            clocks = _app_clock_snapshot(physical)
            payload = {
                "device": device,
                "physical_gpu": physical,
                "lock": {
                    "locked": locked,
                    "theoretical_tflops_fp16": theoretical_tflops,
                    "theoretical_mem_gbps": theoretical_mem_gbps,
                },
                "clocks": clocks,
            }
            run_meta["locks"].append(payload)
            print(f"APP_CLOCKS {json.dumps(payload, sort_keys=True)}", flush=True)
            _write_meta()
            if require_lock and not locked:
                run_meta["error"] = "clock_lock_required_but_unavailable"
                _write_meta()
                _print_lock_failure_help()
                return 3

        # Execute the command while clocks are locked.
        run_meta["command_start_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_meta()
        print(f"== running: {cmd_str}", flush=True)
        env = os.environ.copy()
        # Allow downstream benchmark scripts to enforce that they were launched
        # under a successful clock-locking context.
        all_locked = all(bool(x.get("lock", {}).get("locked")) for x in run_meta.get("locks", []))
        env["AISP_CLOCK_LOCKED"] = "1" if all_locked else "0"
        proc = subprocess.run(cmd, env=env)
        rc = int(proc.returncode)

    run_meta["returncode"] = rc
    run_meta["command_end_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_meta()

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
