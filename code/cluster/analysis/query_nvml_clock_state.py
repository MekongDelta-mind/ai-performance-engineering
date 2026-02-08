#!/usr/bin/env python3
"""
Query NVML clock state, including (when supported) locked clock ranges.

We use ctypes directly because some pynvml builds don't expose the newer
`nvmlDeviceGet*LockedClocks` APIs.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3

NVML_CLOCK_GRAPHICS = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2


def _load_nvml() -> ctypes.CDLL:
    return ctypes.CDLL("libnvidia-ml.so.1")


def _nvml_strerror(lib: ctypes.CDLL, rc: int) -> str:
    try:
        lib.nvmlErrorString.restype = ctypes.c_char_p
        s = lib.nvmlErrorString(rc)
        return (s or b"").decode("utf-8", errors="replace")
    except Exception:
        return f"nvml rc={rc}"


def _maybe_getattr(lib: ctypes.CDLL, name: str):
    try:
        return getattr(lib, name)
    except AttributeError:
        return None


@dataclass
class Nvml:
    lib: ctypes.CDLL

    def __post_init__(self) -> None:
        self._init = _maybe_getattr(self.lib, "nvmlInit_v2") or _maybe_getattr(self.lib, "nvmlInit")
        self._shutdown = _maybe_getattr(self.lib, "nvmlShutdown")
        self._get_count = _maybe_getattr(self.lib, "nvmlDeviceGetCount_v2") or _maybe_getattr(
            self.lib, "nvmlDeviceGetCount"
        )
        self._get_handle_by_index = _maybe_getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2") or _maybe_getattr(
            self.lib, "nvmlDeviceGetHandleByIndex"
        )
        self._get_uuid = _maybe_getattr(self.lib, "nvmlDeviceGetUUID")
        self._get_clock_info = _maybe_getattr(self.lib, "nvmlDeviceGetClockInfo")
        self._get_app_clock = _maybe_getattr(self.lib, "nvmlDeviceGetApplicationsClock")
        self._get_locked_gpu = _maybe_getattr(self.lib, "nvmlDeviceGetGpuLockedClocks")
        self._get_locked_mem = _maybe_getattr(self.lib, "nvmlDeviceGetMemoryLockedClocks")

        if self._init is None or self._shutdown is None:
            raise RuntimeError("NVML init/shutdown symbols not found in libnvidia-ml.so.1")
        if self._get_count is None or self._get_handle_by_index is None:
            raise RuntimeError("NVML device enumeration symbols not found")
        if self._get_clock_info is None or self._get_app_clock is None:
            raise RuntimeError("NVML clock query symbols not found")

        # Set minimal prototypes.
        self._init.restype = ctypes.c_int
        self._shutdown.restype = ctypes.c_int

        self._get_count.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self._get_count.restype = ctypes.c_int

        self._get_handle_by_index.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self._get_handle_by_index.restype = ctypes.c_int

        if self._get_uuid is not None:
            self._get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            self._get_uuid.restype = ctypes.c_int

        self._get_clock_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        self._get_clock_info.restype = ctypes.c_int

        self._get_app_clock.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        self._get_app_clock.restype = ctypes.c_int

        if self._get_locked_gpu is not None:
            self._get_locked_gpu.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_uint),
            ]
            self._get_locked_gpu.restype = ctypes.c_int

        if self._get_locked_mem is not None:
            self._get_locked_mem.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_uint),
            ]
            self._get_locked_mem.restype = ctypes.c_int

    def init(self) -> None:
        rc = int(self._init())
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"nvmlInit failed: {_nvml_strerror(self.lib, rc)}")

    def shutdown(self) -> None:
        rc = int(self._shutdown())
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"nvmlShutdown failed: {_nvml_strerror(self.lib, rc)}")

    def device_count(self) -> int:
        count = ctypes.c_uint(0)
        rc = int(self._get_count(ctypes.byref(count)))
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"nvmlDeviceGetCount failed: {_nvml_strerror(self.lib, rc)}")
        return int(count.value)

    def handle_by_index(self, idx: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        rc = int(self._get_handle_by_index(ctypes.c_uint(idx), ctypes.byref(handle)))
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex({idx}) failed: {_nvml_strerror(self.lib, rc)}")
        return handle

    def uuid(self, handle: ctypes.c_void_p) -> Optional[str]:
        if self._get_uuid is None:
            return None
        buf = ctypes.create_string_buffer(96)
        rc = int(self._get_uuid(handle, buf, ctypes.c_uint(len(buf))))
        if rc != NVML_SUCCESS:
            return None
        return buf.value.decode("utf-8", errors="replace")

    def clock_info_mhz(self, handle: ctypes.c_void_p, clock_type: int) -> int:
        out = ctypes.c_uint(0)
        rc = int(self._get_clock_info(handle, ctypes.c_int(clock_type), ctypes.byref(out)))
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"nvmlDeviceGetClockInfo failed: {_nvml_strerror(self.lib, rc)}")
        return int(out.value)

    def applications_clock_mhz(self, handle: ctypes.c_void_p, clock_type: int) -> int:
        out = ctypes.c_uint(0)
        rc = int(self._get_app_clock(handle, ctypes.c_int(clock_type), ctypes.byref(out)))
        if rc != NVML_SUCCESS:
            raise RuntimeError(f"nvmlDeviceGetApplicationsClock failed: {_nvml_strerror(self.lib, rc)}")
        return int(out.value)

    def locked_gpu_clocks_mhz(self, handle: ctypes.c_void_p) -> Optional[Dict[str, Any]]:
        if self._get_locked_gpu is None:
            return {"supported": False}
        mn = ctypes.c_uint(0)
        mx = ctypes.c_uint(0)
        rc = int(self._get_locked_gpu(handle, ctypes.byref(mn), ctypes.byref(mx)))
        if rc == NVML_ERROR_NOT_SUPPORTED:
            return {"supported": False}
        if rc != NVML_SUCCESS:
            return {"supported": True, "error": _nvml_strerror(self.lib, rc)}
        return {"supported": True, "min_mhz": int(mn.value), "max_mhz": int(mx.value)}

    def locked_mem_clocks_mhz(self, handle: ctypes.c_void_p) -> Optional[Dict[str, Any]]:
        if self._get_locked_mem is None:
            return {"supported": False}
        mn = ctypes.c_uint(0)
        mx = ctypes.c_uint(0)
        rc = int(self._get_locked_mem(handle, ctypes.byref(mn), ctypes.byref(mx)))
        if rc == NVML_ERROR_NOT_SUPPORTED:
            return {"supported": False}
        if rc != NVML_SUCCESS:
            return {"supported": True, "error": _nvml_strerror(self.lib, rc)}
        return {"supported": True, "min_mhz": int(mn.value), "max_mhz": int(mx.value)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="Optional JSON output path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lib = _load_nvml()
    nv = Nvml(lib)

    payload: Dict[str, Any] = {"gpus": []}

    nv.init()
    try:
        n = nv.device_count()
        payload["device_count"] = n
        for i in range(n):
            h = nv.handle_by_index(i)
            gpu = {
                "index": i,
                "uuid": nv.uuid(h),
                "current": {
                    "graphics_mhz": nv.clock_info_mhz(h, NVML_CLOCK_GRAPHICS),
                    "sm_mhz": nv.clock_info_mhz(h, NVML_CLOCK_SM),
                    "mem_mhz": nv.clock_info_mhz(h, NVML_CLOCK_MEM),
                },
                "applications": {
                    "graphics_mhz": nv.applications_clock_mhz(h, NVML_CLOCK_GRAPHICS),
                    "sm_mhz": nv.applications_clock_mhz(h, NVML_CLOCK_SM),
                    "mem_mhz": nv.applications_clock_mhz(h, NVML_CLOCK_MEM),
                },
                "locked": {
                    "gpu_clocks": nv.locked_gpu_clocks_mhz(h),
                    "mem_clocks": nv.locked_mem_clocks_mhz(h),
                },
            }
            payload["gpus"].append(gpu)
    finally:
        nv.shutdown()

    out_text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(out_text)
        print(out)
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

