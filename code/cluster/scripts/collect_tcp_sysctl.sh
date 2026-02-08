#!/usr/bin/env bash
set -euo pipefail

OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "Usage: $0 --output <path>" >&2
  exit 1
fi

sudo sysctl -a 2>/dev/null | grep -E '^net\.ipv4\.tcp_' > "$OUTPUT"

