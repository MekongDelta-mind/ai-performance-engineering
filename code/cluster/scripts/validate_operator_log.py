#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = ["timestamp", "node", "action", "command", "result"]
ALLOWED_RESULTS = {"ok", "skipped", "error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate operator actions JSONL schema.")
    parser.add_argument("--input", required=True, help="Path to operator_actions.jsonl")
    parser.add_argument("--strict", action="store_true", help="Fail on unknown fields")
    return parser.parse_args()


def is_iso8601(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"ERROR line {idx}: invalid JSON ({exc})")
                errors += 1
                continue

            missing = [k for k in REQUIRED_FIELDS if k not in record]
            if missing:
                print(f"ERROR line {idx}: missing fields {missing}")
                errors += 1
                continue

            if not isinstance(record["timestamp"], str) or not is_iso8601(record["timestamp"]):
                print(f"ERROR line {idx}: invalid timestamp '{record['timestamp']}'")
                errors += 1

            if record["result"] not in ALLOWED_RESULTS:
                print(f"ERROR line {idx}: invalid result '{record['result']}'")
                errors += 1

            if args.strict:
                unknown = [k for k in record.keys() if k not in REQUIRED_FIELDS + ["notes"]]
                if unknown:
                    print(f"ERROR line {idx}: unknown fields {unknown}")
                    errors += 1

    if errors:
        print(f"Validation failed with {errors} error(s)")
        return 1

    print("Validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
