#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _parse_size_bytes(raw: str) -> int:
    s = raw.strip()
    if not s:
        raise ValueError("empty size")
    s_l = s.lower()
    mul = 1
    for suf, m in (
        ("k", 1024),
        ("kb", 1024),
        ("m", 1024**2),
        ("mb", 1024**2),
        ("g", 1024**3),
        ("gb", 1024**3),
        ("t", 1024**4),
        ("tb", 1024**4),
    ):
        if s_l.endswith(suf):
            mul = m
            s_l = s_l[: -len(suf)].strip()
            break
    if not s_l.isdigit():
        raise ValueError(f"invalid size: {raw!r}")
    return int(s_l) * mul


def _fnum(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.3f}"


@dataclass(frozen=True)
class IOResult:
    seconds: float
    bytes_total: int
    fsync_seconds: float

    @property
    def mb_s(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return (self.bytes_total / self.seconds) / (1024.0 * 1024.0)


def _write_file(path: Path, bytes_total: int, block_bytes: int, do_fsync: bool) -> IOResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray(block_bytes)
    mv = memoryview(buf)
    fd = os.open(str(path), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o644)
    try:
        remaining = bytes_total
        start = time.perf_counter()
        while remaining > 0:
            n = block_bytes if remaining >= block_bytes else remaining
            os.write(fd, mv[:n])
            remaining -= n
        end_write = time.perf_counter()
        fsync_s = 0.0
        if do_fsync:
            fsync_start = time.perf_counter()
            os.fsync(fd)
            fsync_s = time.perf_counter() - fsync_start
        total_s = time.perf_counter() - start
        # Separate the write() loop time from fsync time for visibility.
        _ = end_write  # keep explicit for readability
        return IOResult(seconds=total_s, bytes_total=bytes_total, fsync_seconds=fsync_s)
    finally:
        os.close(fd)


def _read_file(path: Path, block_bytes: int, do_fsync: bool) -> IOResult:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        total = 0
        start = time.perf_counter()
        while True:
            chunk = os.read(fd, block_bytes)
            if not chunk:
                break
            total += len(chunk)
        fsync_s = 0.0
        # fsync on read isn't meaningful; keep the field for schema parity.
        _ = do_fsync
        total_s = time.perf_counter() - start
        return IOResult(seconds=total_s, bytes_total=total, fsync_seconds=fsync_s)
    finally:
        os.close(fd)


def main() -> int:
    p = argparse.ArgumentParser(description="Checkpoint-like I/O benchmark (write + read, fsync optional).")
    p.add_argument("--run-id", default=os.environ.get("RUN_ID") or time.strftime("%Y-%m-%d"))
    p.add_argument("--label", default=os.environ.get("LABEL") or socket.gethostname())
    p.add_argument("--test-dir", default=os.environ.get("TEST_DIR") or "/tmp")
    p.add_argument("--bytes", default=os.environ.get("BYTES") or "4G", help="Bytes per file (default: 4G).")
    p.add_argument("--block-size", default=os.environ.get("BLOCK_SIZE") or "4M", help="Write/read block size.")
    p.add_argument("--files", type=int, default=int(os.environ.get("FILES") or "1"), help="Number of files.")
    p.add_argument("--fsync", type=int, default=int(os.environ.get("FSYNC") or "1"), help="1 to fsync after write.")
    p.add_argument("--write", type=int, default=int(os.environ.get("DO_WRITE") or "1"), help="1 to run write test.")
    p.add_argument("--read", type=int, default=int(os.environ.get("DO_READ") or "1"), help="1 to run read test.")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-csv", default="")
    args = p.parse_args()

    bytes_per_file = _parse_size_bytes(args.bytes)
    block_bytes = _parse_size_bytes(args.block_size)
    if block_bytes <= 0:
        raise SystemExit("--block-size must be > 0")
    if args.files <= 0:
        raise SystemExit("--files must be >= 1")

    run_id = str(args.run_id)
    label = str(args.label)
    test_dir = Path(args.test_dir).expanduser().resolve()
    work_dir = test_dir / f"aisp_checkpoint_io_{run_id}_{label}"
    work_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.output_json) if args.output_json else Path("results/structured") / f"{run_id}_{label}_checkpoint_io.json"
    out_csv = Path(args.output_csv) if args.output_csv else Path("results/structured") / f"{run_id}_{label}_checkpoint_io.csv"

    print("========================================")
    print("Checkpoint I/O Benchmark")
    print("========================================")
    print(f"Date: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"Host: {socket.gethostname()}")
    print(f"RUN_ID: {run_id}")
    print(f"Label: {label}")
    print(f"Work dir: {work_dir}")
    print(f"Bytes/file: {bytes_per_file}")
    print(f"Block size: {block_bytes}")
    print(f"Files: {args.files}")
    print(f"Write: {bool(args.write)} (fsync={bool(args.fsync)})")
    print(f"Read:  {bool(args.read)}")
    try:
        st = os.statvfs(str(work_dir))
        free = st.f_bavail * st.f_frsize
        print(f"Work dir free bytes: {free}")
    except Exception:
        pass
    print("")

    file_paths = [work_dir / f"ckpt_{i:03d}.bin" for i in range(args.files)]

    write_results = []
    read_results = []
    for fp in file_paths:
        if args.write:
            print(f"== write {fp.name} ==")
            r = _write_file(fp, bytes_per_file, block_bytes, bool(args.fsync))
            write_results.append(
                {
                    "path": str(fp),
                    "bytes": r.bytes_total,
                    "seconds": r.seconds,
                    "mb_s": r.mb_s,
                    "fsync_seconds": r.fsync_seconds,
                }
            )
            print(f"  write_mb_s={r.mb_s:.2f} total_s={r.seconds:.3f} fsync_s={r.fsync_seconds:.3f}")
        if args.read:
            print(f"== read {fp.name} ==")
            r = _read_file(fp, block_bytes, bool(args.fsync))
            read_results.append(
                {
                    "path": str(fp),
                    "bytes": r.bytes_total,
                    "seconds": r.seconds,
                    "mb_s": r.mb_s,
                }
            )
            print(f"  read_mb_s={r.mb_s:.2f} total_s={r.seconds:.3f}")

    def _agg(xs, key: str) -> Optional[float]:
        vals = [float(x.get(key) or 0.0) for x in xs]
        if not vals:
            return None
        return sum(vals) / len(vals)

    payload = {
        "run_id": run_id,
        "label": label,
        "host": socket.gethostname(),
        "work_dir": str(work_dir),
        "bytes_per_file": bytes_per_file,
        "block_bytes": block_bytes,
        "files": args.files,
        "fsync": bool(args.fsync),
        "results": {
            "write": write_results,
            "read": read_results,
            "write_mb_s_mean": _agg(write_results, "mb_s"),
            "read_mb_s_mean": _agg(read_results, "mb_s"),
        },
        "notes": {
            "cleanup": "This tool does not delete checkpoint files by default. Files are under work_dir.",
            "read_cache": "Read results may be affected by page cache unless you drop caches separately.",
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_json}")

    cols = [
        "run_id",
        "label",
        "host",
        "work_dir",
        "bytes_per_file",
        "block_bytes",
        "files",
        "fsync",
        "write_mb_s_mean",
        "read_mb_s_mean",
    ]
    row = [
        run_id,
        label,
        socket.gethostname(),
        str(work_dir),
        bytes_per_file,
        block_bytes,
        args.files,
        int(bool(args.fsync)),
        _fnum(payload["results"].get("write_mb_s_mean")),
        _fnum(payload["results"].get("read_mb_s_mean")),
    ]

    write_header = True
    if out_csv.exists():
        try:
            if out_csv.read_text(encoding="utf-8").strip():
                write_header = False
        except Exception:
            pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(cols)
        w.writerow(row)
    print(f"Wrote {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

