#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG_ARGS=("$@")
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" -- "$0" "${ORIG_ARGS[@]}"
fi

BIN=/tmp/p2p_test/p2p_bw_latency
OUT=/tmp/p2p_test/p2p_sweep.csv

if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN. Build it first." >&2
  exit 1
fi

echo "test_type,size_bytes,bw_bytes,lat_bytes,bw_iters,lat_iters,src,dst,bw_gbs,lat_us,peer_access" > "$OUT"

# Bandwidth sweep (latency fixed at 4B)
bw_sizes=(1048576 4194304 16777216 67108864 268435456 1073741824) # 1MiB..1GiB
bw_iters=10
lat_bytes=4
lat_iters=10000

for s in "${bw_sizes[@]}"; do
  echo "Running bandwidth sweep size=${s} bytes" >&2
  "$BIN" "$s" "$bw_iters" "$lat_bytes" "$lat_iters" | tail -n +2 | while IFS=, read -r src dst bw lat peer; do
    echo "bw,$s,$s,$lat_bytes,$bw_iters,$lat_iters,$src,$dst,$bw,$lat,$peer" >> "$OUT"
  done
done

# Latency sweep (bandwidth fixed at 1MiB)
lat_sizes=(4 16 64 256 1024 4096 16384 65536 262144 1048576) # 4B..1MiB
bw_bytes=1048576
bw_iters=5
lat_iters=20000

for s in "${lat_sizes[@]}"; do
  echo "Running latency sweep size=${s} bytes" >&2
  "$BIN" "$bw_bytes" "$bw_iters" "$s" "$lat_iters" | tail -n +2 | while IFS=, read -r src dst bw lat peer; do
    echo "lat,$s,$bw_bytes,$s,$bw_iters,$lat_iters,$src,$dst,$bw,$lat,$peer" >> "$OUT"
  done
done

echo "Wrote $OUT" >&2
