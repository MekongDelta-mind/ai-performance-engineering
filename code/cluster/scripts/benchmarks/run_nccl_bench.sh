#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG_ARGS=("$@")
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" -- "$0" "${ORIG_ARGS[@]}"
fi

BIN=/tmp/p2p_test/nccl_allreduce_bench
OUT=/tmp/p2p_test/nccl_allreduce.csv

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN. Build it first." >&2
  exit 1
fi

# min_bytes max_bytes factor iters
MIN=8
MAX=$((1<<30))
FACTOR=2
ITERS=20

$BIN $MIN $MAX $FACTOR $ITERS | tee "$OUT" >/dev/null

echo "Wrote $OUT" >&2
