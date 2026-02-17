#!/usr/bin/env bash
set -euo pipefail

OUTPUT=""
LABEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "Usage: $0 --output <path> [--label <node_label>]" >&2
  exit 1
fi

python3 - <<'PY' "$OUTPUT" "$LABEL"
import json
import subprocess
import sys
import time

out_path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ""

def run(cmd: str):
    proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

cmds = [
    ("date", "date -Iseconds"),
    ("hostname", "hostname"),
    ("lsblk_json", "lsblk -J -O"),
    ("lsblk", "lsblk -o NAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL"),
    ("blkid", "blkid"),
    ("df", "df -hT"),
    ("mount", "mount"),
    ("disk_by_id", "ls -l /dev/disk/by-id"),
]

results = {
    "label": label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "commands": {},
}

for name, cmd in cmds:
    results["commands"][name] = run(cmd)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, sort_keys=True)

print(f"Wrote {out_path}")
PY
