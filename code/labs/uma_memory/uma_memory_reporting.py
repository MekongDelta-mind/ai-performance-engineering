"""UMA-aware memory reporting (tool; not a comparable benchmark pair).

This script prints a snapshot of:
- `cudaMemGetInfo()` free/total bytes (HBM or device memory).
- Host MemAvailable + SwapFree from `/proc/meminfo`.
- A simple UMA "allocatable" estimate: MemAvailable + reclaim_fraction * SwapFree.

It is intended as a diagnostics utility for unified CPU↔GPU memory systems
(e.g., Grace-Blackwell). It is NOT a baseline/optimized benchmark pair and
should be run via `aisp tools uma-memory -- ...`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ch02.uma_memory_utils import format_bytes, is_integrated_gpu, read_meminfo  # noqa: E402


def _collect_per_process_usage_bytes(device_index: int) -> Optional[int]:
    """Sum NVML-reported per-process GPU memory usage for the device."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            memory_used = sum(
                int(p.usedGpuMemory)
                for p in procs
                if getattr(p, "usedGpuMemory", None) is not None
            )
        finally:
            pynvml.nvmlShutdown()
        return int(memory_used)
    except Exception:
        return None


def collect_snapshot(*, reclaim_fraction: float, device_index: int) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UMA memory reporting.")

    device_index = int(device_index)
    reclaim_fraction = float(reclaim_fraction)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    meminfo = read_meminfo()

    if meminfo is None:
        memavailable_bytes = None
        swapfree_bytes = None
        allocatable_bytes = int(free_bytes)
    else:
        memavailable_bytes = int(meminfo.effective_available_kb() * 1024)
        swapfree_bytes = int(meminfo.swap_free_kb * 1024)
        allocatable_bytes = int(meminfo.allocatable_bytes(reclaim_fraction=reclaim_fraction))

    per_process_bytes = _collect_per_process_usage_bytes(device_index)
    props = torch.cuda.get_device_properties(device_index)

    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "timestamp_utc": timestamp_utc,
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda if torch.version else None,
        "device_index": device_index,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_mem_bytes": int(props.total_memory),
        "integrated_gpu": is_integrated_gpu(),
        "reclaim_fraction": reclaim_fraction,
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
        "memavailable_bytes": memavailable_bytes,
        "swapfree_bytes": swapfree_bytes,
        "uma_allocatable_bytes": int(allocatable_bytes),
        "per_process_bytes": int(per_process_bytes) if per_process_bytes is not None else None,
    }


def _write_snapshot(snapshot: Dict[str, Any], snapshot_dir: Path, file_name: Optional[str]) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if file_name is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_name = f"uma_snapshot_{timestamp}.json"
    out_path = snapshot_dir / file_name
    out_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return out_path


def _print_human(snapshot: Dict[str, Any]) -> None:
    print("\n=== UMA-aware CUDA memory report ===")
    print(f"Host: {snapshot.get('hostname')}")
    print(f"Device[{snapshot.get('device_index')}]: {snapshot.get('device_name')}")
    print(f"Integrated GPU detected: {snapshot.get('integrated_gpu')}")
    print(f"cudaMemGetInfo free:        {format_bytes(int(snapshot['cuda_free_bytes']))}")
    print(f"cudaMemGetInfo total:       {format_bytes(int(snapshot['cuda_total_bytes']))}")

    memavailable = snapshot.get("memavailable_bytes")
    if memavailable is not None:
        print(f"Host MemAvailable:          {format_bytes(int(memavailable))}")

    swapfree = snapshot.get("swapfree_bytes")
    if swapfree is not None:
        reclaim_fraction = float(snapshot.get("reclaim_fraction", 0.0))
        print(f"SwapFree (reclaimable {reclaim_fraction:.0%}): {format_bytes(int(swapfree))}")

    print(f"UMA allocatable estimate:   {format_bytes(int(snapshot['uma_allocatable_bytes']))}")

    per_process = snapshot.get("per_process_bytes")
    if per_process is not None:
        print(f"Per-process NVML usage sum: {format_bytes(int(per_process))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMA-aware memory reporting (tool).")
    parser.add_argument(
        "--reclaim-fraction",
        "-r",
        type=float,
        default=0.9,
        help="Fraction of SwapFree to treat as reclaimable for UMA estimation.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to report.",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Write a JSON snapshot to the snapshot directory.",
    )
    parser.add_argument(
        "--snapshot-dir",
        "-o",
        type=Path,
        default=Path("artifacts/uma_memory_snapshots"),
        help="Directory for UMA snapshot output.",
    )
    parser.add_argument(
        "--snapshot-name",
        type=str,
        default=None,
        help="Optional snapshot file name (default: timestamped).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the snapshot JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = collect_snapshot(
        reclaim_fraction=args.reclaim_fraction,
        device_index=args.device_index,
    )

    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        _print_human(snapshot)

    if args.snapshot:
        out_path = _write_snapshot(snapshot, args.snapshot_dir, args.snapshot_name)
        print(f"\nSnapshot written: {out_path}")


if __name__ == "__main__":
    main()
