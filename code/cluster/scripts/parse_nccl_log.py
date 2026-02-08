#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any


APP_CLOCK_PREFIX = "APP_CLOCKS "
MAX_MATCH_LINES = 20


def parse_nccl_log(path: Path) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    app_clocks: List[Dict[str, Any]] = []

    size_row = re.compile(r"^\s*\d+")
    nvls_channels_re = re.compile(r"(\d+)\s+nvls\s+channels", re.IGNORECASE)
    mnnvl_re = re.compile(r"\bMNNVL\s+(\d+)", re.IGNORECASE)
    nnodes_re = re.compile(r"\bnNodes\s+(\d+)", re.IGNORECASE)

    summary: Dict[str, Any] = {
        "nvls": {"channels": None, "comm_lines": 0},
        "mnnvl": {"mnnvl": None, "nNodes": None},
        "collnet": {"seen": False, "lines": []},
        "sharp": {"seen": False, "lines": []},
        "errors": {"nvls_init_failure": False, "lines": []},
    }

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(APP_CLOCK_PREFIX):
                payload = line[len(APP_CLOCK_PREFIX) :]
                try:
                    app_clocks.append(json.loads(payload))
                except json.JSONDecodeError:
                    app_clocks.append({"raw": payload, "error": "json_decode"})
                continue

            # Extract useful "INIT" metadata from NCCL_DEBUG logs.
            m = nvls_channels_re.search(line)
            if m:
                try:
                    summary["nvls"]["channels"] = int(m.group(1))
                except ValueError:
                    pass
            if "NVLS comm" in line:
                summary["nvls"]["comm_lines"] += 1

            m = mnnvl_re.search(line)
            if m and summary["mnnvl"]["mnnvl"] is None:
                try:
                    summary["mnnvl"]["mnnvl"] = int(m.group(1))
                except ValueError:
                    pass
            m = nnodes_re.search(line)
            if m and summary["mnnvl"]["nNodes"] is None:
                try:
                    summary["mnnvl"]["nNodes"] = int(m.group(1))
                except ValueError:
                    pass

            if re.search(r"\bcollnet", line, flags=re.IGNORECASE):
                summary["collnet"]["seen"] = True
                if len(summary["collnet"]["lines"]) < MAX_MATCH_LINES:
                    summary["collnet"]["lines"].append(line)
            if "SHARP" in line or "sharp" in line:
                summary["sharp"]["seen"] = True
                if len(summary["sharp"]["lines"]) < MAX_MATCH_LINES:
                    summary["sharp"]["lines"].append(line)

            if re.search(r"transport/nvls\.cc", line, flags=re.IGNORECASE) or "Cuda failure 801" in line:
                summary["errors"]["nvls_init_failure"] = True
                if len(summary["errors"]["lines"]) < MAX_MATCH_LINES:
                    summary["errors"]["lines"].append(line)

            if line.startswith("#"):
                continue
            if not size_row.match(line):
                continue

            # nccl-tests prints a data row with either:
            #   size count type redop time_us algbw busbw ...
            # or (most commonly):
            #   size count type redop root time_us algbw busbw ...
            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                size_bytes = int(parts[0])
                count = int(parts[1])
                dtype = parts[2]
                redop = parts[3]

                root = None
                if len(parts) >= 8:
                    try:
                        root = int(parts[4])
                        time_us = float(parts[5])
                        algbw = float(parts[6])
                        busbw = float(parts[7])
                    except ValueError:
                        root = None
                        time_us = float(parts[4])
                        algbw = float(parts[5])
                        busbw = float(parts[6])
                else:
                    time_us = float(parts[4])
                    algbw = float(parts[5])
                    busbw = float(parts[6])
            except ValueError:
                continue

            results.append(
                {
                    "size_bytes": size_bytes,
                    "count": count,
                    "dtype": dtype,
                    "redop": redop,
                    "root": root,
                    "time_us": time_us,
                    "algbw_gbps": algbw,
                    "busbw_gbps": busbw,
                }
            )

    return {"results": results, "app_clocks": app_clocks, "log_summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to nccl-tests log")
    parser.add_argument("--output", required=True, help="Path to write structured JSON")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--hosts", required=True, help="Comma-separated host list")
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--command", required=True, help="Command string executed")
    args = parser.parse_args()

    log_path = Path(args.input)
    out_path = Path(args.output)
    parsed = parse_nccl_log(log_path)

    payload = {
        "run_id": args.run_id,
        "hosts": [h.strip() for h in args.hosts.split(",") if h.strip()],
        "gpus_per_node": args.gpus_per_node,
        "total_ranks": args.gpus_per_node * len([h for h in args.hosts.split(",") if h.strip()]),
        "command": args.command,
        "raw_log": str(log_path),
        "results": parsed["results"],
        "app_clocks": parsed["app_clocks"],
        "log_summary": parsed["log_summary"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
