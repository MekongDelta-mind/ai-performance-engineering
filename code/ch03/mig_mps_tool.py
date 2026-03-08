#!/usr/bin/env python3
"""Report MIG and MPS configuration for a given GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    import pynvml  # type: ignore
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError("mig_mps_tool requires pynvml (nvidia-ml-py)") from exc

def _get_mps_processes(handle) -> List[object]:
    for name in (
        "nvmlDeviceGetMPSComputeRunningProcesses_v3",
        "nvmlDeviceGetMPSComputeRunningProcesses_v2",
        "nvmlDeviceGetMPSComputeRunningProcesses",
    ):
        if hasattr(pynvml, name):
            return list(getattr(pynvml, name)(handle))
    raise RuntimeError("pynvml does not expose MPS process queries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report MIG and MPS status.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    args = parser.parse_args()

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
        name = pynvml.nvmlDeviceGetName(handle)
        mig_mode, mig_mode_pending = pynvml.nvmlDeviceGetMigMode(handle)

        print("MIG/MPS report")
        print(f"  Device: {args.device} ({name})")
        print(f"  MIG mode: {mig_mode} (pending: {mig_mode_pending})")

        if mig_mode:
            try:
                count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
            except Exception:
                count = 0
            mig_devices = []
            for idx in range(count):
                try:
                    mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, idx)
                except Exception:
                    continue
                try:
                    mig_uuid = pynvml.nvmlDeviceGetUUID(mig_handle)
                except Exception:
                    mig_uuid = "unknown"
                mig_devices.append(mig_uuid)
            print(f"  MIG devices: {len(mig_devices)}")
            for uuid in mig_devices:
                print(f"    - {uuid}")
        else:
            print("  MIG devices: 0")

        mps_procs = _get_mps_processes(handle)
        print(f"  MPS processes: {len(mps_procs)}")
        for proc in mps_procs:
            try:
                pid = proc.pid
                mem = getattr(proc, "usedGpuMemory", None)
                mem_str = f" ({mem / (1024 ** 2):.1f} MiB)" if mem is not None else ""
                print(f"    - pid {pid}{mem_str}")
            except Exception:
                continue
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
