#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a vLLM online-serving concurrency sweep in the official vLLM container.

This reproduces the "vLLM Concurrency Sweep Results" and per-concurrency
"Serving Benchmark Result" screenshots (vllm bench serve).

Usage:
  scripts/repro/run_vllm_serve_sweep_container.sh [options]

Options:
  --run-id <id>              RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>            Label for output paths (default: hostname)
  --model <hf_model_id>      (default: openai/gpt-oss-120b)
  --tp <n>                   Tensor parallel size (default: all visible GPUs)
  --isl <n>                  Input sequence length (default: 1024)
  --osl <n>                  Output sequence length (default: 1024)
  --concurrency-range "..."  Space-separated concurrencies (default: "32 64 128 256 512")
  --port <port>              (default: 8888)
  --image <docker_image>     (default: auto by architecture)
  --detach                   Start the sweep container in detached mode.
                             (Default: run attached and stream logs.)

Env:
  HF_TOKEN can be set to enable gated model downloads.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="$(hostname)"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
PORT="${PORT:-8888}"
CONCURRENCY_RANGE="${CONCURRENCY_RANGE:-32 64 128 256 512}"
DETACH=0

ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --isl) ISL="$2"; shift 2 ;;
    --osl) OSL="$2"; shift 2 ;;
    --concurrency-range) CONCURRENCY_RANGE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --image) CONTAINER_IMAGE="$2"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found" >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ -z "$CONTAINER_IMAGE" ]]; then
  if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    CONTAINER_IMAGE="vllm/vllm-openai:cu130-nightly-aarch64"
  else
    CONTAINER_IMAGE="vllm/vllm-openai:cu130-nightly"
  fi
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ -z "$TP" ]]; then
  TP="$GPU_COUNT"
fi
if [[ "$TP" -gt "$GPU_COUNT" ]]; then
  echo "WARNING: Requested TP=$TP but only $GPU_COUNT GPUs available. Using TP=$GPU_COUNT" >&2
  TP="$GPU_COUNT"
fi

OUT_DIR="${ROOT_DIR}/results/raw/${RUN_ID}_${LABEL}_vllm_serve_sweep"
mkdir -p "$OUT_DIR"

MAX_MODEL_LEN=$((ISL + OSL + 256))

STRUCT_DIR="${ROOT_DIR}/results/structured"
mkdir -p "$STRUCT_DIR"
LOCK_META_OUT="${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep_clock_lock.json"

# vLLM is executed via Docker + NVIDIA runtime. Ensure nvidia-persistenced is
# running so its socket exists for container mount hooks.
if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  if command -v systemctl >/dev/null 2>&1; then
    if sudo -n true >/dev/null 2>&1; then
      sudo systemctl start nvidia-persistenced >/dev/null 2>&1 || true
    fi
  fi
fi
if [[ ! -S "/run/nvidia-persistenced/socket" ]]; then
  echo "ERROR: /run/nvidia-persistenced/socket is missing." >&2
  echo "This is required for running the vLLM container with GPU access." >&2
  echo "Fix: sudo systemctl start nvidia-persistenced" >&2
  exit 1
fi

# Enforce strict GPU clock locking for the entire sweep (server + benchmarks).
if [[ "${AISP_CLOCK_LOCKED:-}" != "1" ]]; then
  export RUN_ID LABEL
  exec "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META_OUT" \
    -- "$0" "${ORIG_ARGS[@]}"
fi

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
TIKTOKEN_CACHE="${HOME}/.cache/tiktoken_rs"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$HOME/.cache/flashinfer}"
mkdir -p "$HF_CACHE_DIR" "$TIKTOKEN_CACHE" "$VLLM_CACHE_DIR" "$FLASHINFER_CACHE_DIR"

# Harmony vocab (for gpt-oss / o200k_base) used by tiktoken-rs.
TIKTOKEN_VOCAB_FILE="${TIKTOKEN_CACHE}/fb374d419588a4632f3f557e76b4b70aebbca790"
if [[ ! -f "$TIKTOKEN_VOCAB_FILE" ]]; then
  echo "Downloading harmony tiktoken vocab file to ${TIKTOKEN_VOCAB_FILE}..."
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$TIKTOKEN_VOCAB_FILE" https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken || true
  else
    curl -fsSL https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken -o "$TIKTOKEN_VOCAB_FILE" || true
  fi
fi

HF_MOUNT=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_MOUNT+=(-e "HF_TOKEN=${HF_TOKEN}" -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi

read -r -a CONC_ARR <<<"$CONCURRENCY_RANGE"

echo "========================================"
echo "vLLM Concurrency Sweep (Containerized)"
echo "========================================"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Architecture: $(uname -m)"
echo "Container: $CONTAINER_IMAGE"
echo ""
echo "Model: $MODEL"
echo "TP: $TP"
echo "ISL: $ISL"
echo "OSL: $OSL"
echo "Max model len: $MAX_MODEL_LEN"
echo "Concurrency range: ${CONC_ARR[*]}"
echo "Output dir: $OUT_DIR"
echo ""

echo "Pulling container (best-effort)..."
docker pull "$CONTAINER_IMAGE" 2>/dev/null || echo "Using cached container"

INNER="${ROOT_DIR}/scripts/repro/vllm_serve_sweep_inner.sh"
if [[ ! -f "$INNER" ]]; then
  echo "Missing inner script at ${INNER}" >&2
  exit 1
fi

LOG_PATH="${OUT_DIR}/sweep_log.txt"

DOCKER_ARGS=(
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --network host
  -e TIKTOKEN_RS_CACHE_DIR=/root/.cache/tiktoken_rs
  "${HF_MOUNT[@]}"
  -v "$INNER":/sweep.sh:ro
  -v "$OUT_DIR":/results
  -v "$HF_CACHE_DIR":/root/.cache/huggingface
  -v "$TIKTOKEN_CACHE":/root/.cache/tiktoken_rs
  -v "$VLLM_CACHE_DIR":/root/.cache/vllm
  -v "$FLASHINFER_CACHE_DIR":/root/.cache/flashinfer
  --entrypoint bash
  "$CONTAINER_IMAGE"
)

if [[ "$DETACH" -eq 1 ]]; then
  safe_name="$(echo "vllm_sweep_${RUN_ID}_${LABEL}" | tr -c '[:alnum:]_.' '_' )_$(date +%s)"
  echo "Starting detached container (will still wait to keep GPU clocks locked): ${safe_name}"
  docker run -d --name "$safe_name" \
    "${DOCKER_ARGS[@]}" \
    -lc "/sweep.sh \"$MODEL\" \"$TP\" \"$ISL\" \"$OSL\" \"$MAX_MODEL_LEN\" \"$PORT\" \"/results\" ${CONC_ARR[*]} > /results/sweep_log.txt 2>&1"
  echo "Detached sweep started."
  echo "  container: ${safe_name}"
  echo "  log file:  ${LOG_PATH}"
  echo "  summary:   ${OUT_DIR}/summary.txt (written on completion)"

  # The container redirects stdout/stderr to a file under /results, so docker logs
  # is not useful here. Tail the on-disk log instead.
  tail -n +1 -F "$LOG_PATH" &
  TAIL_PID=$!
  rc="$(docker wait "$safe_name")"
  kill "$TAIL_PID" 2>/dev/null || true
  wait "$TAIL_PID" 2>/dev/null || true
  docker rm "$safe_name" >/dev/null 2>&1 || true
  if [[ "$rc" -ne 0 ]]; then
    echo "ERROR: vLLM sweep container exited with code ${rc}" >&2
    exit "$rc"
  fi
fi

if [[ "$DETACH" -ne 1 ]]; then
  docker run --rm "${DOCKER_ARGS[@]}" \
    /sweep.sh \
      "$MODEL" "$TP" "$ISL" "$OSL" "$MAX_MODEL_LEN" "$PORT" \
      "/results" "${CONC_ARR[@]}" 2>&1 | tee "$LOG_PATH"
fi

if [[ -f "$OUT_DIR/summary.txt" ]]; then
  cp -f "$OUT_DIR/summary.txt" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_concurrency_sweep_summary.txt"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_concurrency_sweep_summary.txt"
fi
if [[ -f "$OUT_DIR/sweep_summary.csv" ]]; then
  cp -f "$OUT_DIR/sweep_summary.csv" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.csv"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.csv"
fi
if [[ -f "$OUT_DIR/sweep_summary.jsonl" ]]; then
  cp -f "$OUT_DIR/sweep_summary.jsonl" "${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.jsonl"
  echo "Wrote ${STRUCT_DIR}/${RUN_ID}_${LABEL}_vllm_serve_sweep.jsonl"
fi

echo "Wrote ${LOG_PATH}"
