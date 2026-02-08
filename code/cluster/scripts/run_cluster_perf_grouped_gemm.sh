#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Run the Cluster Perf grouped GEMM benchmark (DeepGEMM FP8xFP4 path + torch baselines)
and write a structured log + summary + plot under results/ and docs/.

This validates the DeepGEMM grouped-GEMM FP8xFP4 path on GB200/SM100.

Usage:
  scripts/run_cluster_perf_grouped_gemm.sh \
    --suite-dir /path/to/cluster_perf_suite \
    --run-id 2026-02-08_deepgemm_grouped_gemm \
    --label node1

Options:
  --suite-dir <dir>   Required. Cluster Perf suite root dir (has standalone/compute/).
                      You can also set CLUSTER_PERF_SUITE_DIR instead.
  --run-id <id>       RUN_ID prefix for outputs (default: YYYY-MM-DD_grouped_gemm).
  --label <label>     Label used in output filenames (default: hostname).
  --preset <name>     Preset passed to grouped_gemm_bench.py (default: all).
  --warmup <n>        Warmup iterations (default: 5).
  --iters <n>         Benchmark iterations (default: 30).
  --image <image>     Container image (default: ghcr.io/jordannanos/cmax-compute:latest).
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)_grouped_gemm"
LABEL="$(hostname)"
PRESET="all"
WARMUP="5"
ITERS="30"
SUITE_DIR="${CLUSTER_PERF_SUITE_DIR:-}"
IMAGE="${CONTAINER_IMAGE:-ghcr.io/jordannanos/cmax-compute:latest}"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --suite-dir) SUITE_DIR="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --warmup) WARMUP="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SUITE_DIR" ]]; then
  echo "ERROR: --suite-dir is required (or set CLUSTER_PERF_SUITE_DIR)." >&2
  exit 1
fi

COMPUTE_DIR="${SUITE_DIR}/standalone/compute"
if [[ ! -d "$COMPUTE_DIR" ]]; then
  echo "ERROR: suite compute dir not found: $COMPUTE_DIR" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/results/structured" "${ROOT_DIR}/docs/figures"

OUT_LOG="${ROOT_DIR}/results/structured/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm.txt"
OUT_JSON="${ROOT_DIR}/results/structured/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_summary.json"
OUT_PNG="${ROOT_DIR}/docs/figures/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_tflops.png"
LOCK_META="${ROOT_DIR}/results/structured/${RUN_ID}_${LABEL}_cluster_perf_grouped_gemm_clock_lock.json"

echo "== Cluster Perf Grouped GEMM =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "SUITE_DIR=${SUITE_DIR}"
echo "COMPUTE_DIR=${COMPUTE_DIR}"
echo "IMAGE=${IMAGE}"
echo "PRESET=${PRESET}"
echo "WARMUP=${WARMUP}"
echo "ITERS=${ITERS}"
echo

"${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
  --lock-meta-out "$LOCK_META" \
  -- bash -lc "set -euo pipefail; docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v \"${COMPUTE_DIR}:/workspace\" -w /workspace \"${IMAGE}\" python -u gemm-bench/grouped_gemm_bench.py --preset \"${PRESET}\" --warmup \"${WARMUP}\" --iters \"${ITERS}\" 2>&1 | tee \"${OUT_LOG}\""

"${ROOT_DIR}/env/venv/bin/python" \
  "${ROOT_DIR}/analysis/summarize_grouped_gemm_torch_fp16_vs_fp8.py" \
  --in-log "$OUT_LOG" \
  --out-json "$OUT_JSON"

"${ROOT_DIR}/env/venv/bin/python" \
  "${ROOT_DIR}/analysis/plot_grouped_gemm_torch_fp16_vs_fp8.py" \
  --summary-json "$OUT_JSON" \
  --title "Grouped GEMM: Torch FP16/FP8 vs DeepGEMM FP8xFP4" \
  --out "$OUT_PNG"

echo
echo "Outputs:"
echo "  - $OUT_LOG"
echo "  - $OUT_JSON"
echo "  - $OUT_PNG"
echo "  - $LOCK_META"
