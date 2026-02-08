#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Run a strict DeepGEMM FP8xFP4 smoke/perf check in container and write:
  - raw log
  - structured JSON
  - clock-lock metadata

Usage:
  scripts/run_cluster_perf_fp4_smoke.sh [options]

Options:
  --run-id <id>     RUN_ID prefix (default: YYYY-MM-DD_fp4_smoke)
  --label <label>   Label used in output filenames (default: hostname)
  --image <image>   Container image (default: ghcr.io/jordannanos/cmax-compute:latest)
  --m <int>         M dimension (default: 4096)
  --n <int>         N dimension (default: 4096)
  --k <int>         K dimension (default: 4096)
  --warmup <n>      Warmup iterations (default: 10)
  --iters <n>       Measured iterations (default: 30)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)_fp4_smoke"
LABEL="$(hostname)"
IMAGE="${CONTAINER_IMAGE:-ghcr.io/jordannanos/cmax-compute:latest}"
M="4096"
N="4096"
K="4096"
WARMUP="10"
ITERS="30"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --m) M="${2:-}"; shift 2 ;;
    --n) N="${2:-}"; shift 2 ;;
    --k) K="${2:-}"; shift 2 ;;
    --warmup) WARMUP="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${ROOT_DIR}/results/raw" "${ROOT_DIR}/results/structured"

OUT_LOG="${ROOT_DIR}/results/raw/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke.log"
OUT_JSON="${ROOT_DIR}/results/structured/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke.json"
LOCK_META="${ROOT_DIR}/results/structured/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke_clock_lock.json"
OUT_JSON_IN_CONTAINER="/workspace/results/structured/${RUN_ID}_${LABEL}_cluster_perf_fp4_smoke.json"

echo "== Cluster Perf FP4 Smoke =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "IMAGE=${IMAGE}"
echo "SHAPE=${M}x${N}x${K}"
echo "WARMUP=${WARMUP}"
echo "ITERS=${ITERS}"
echo

"${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
  --lock-meta-out "$LOCK_META" \
  -- bash -lc "set -euo pipefail; docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v \"${ROOT_DIR}:/workspace\" -w /workspace \"${IMAGE}\" python -u analysis/smoke_deepgemm_fp8_fp4.py --m \"${M}\" --n \"${N}\" --k \"${K}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" --out-json \"${OUT_JSON_IN_CONTAINER}\" 2>&1 | tee \"${OUT_LOG}\""

echo
echo "Outputs:"
echo "  - $OUT_LOG"
echo "  - $OUT_JSON"
echo "  - $LOCK_META"
