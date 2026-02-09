#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  vllm_multinode_inner.sh \
    <model> <tp> <isl> <osl> <max_model_len> <port> <ray_port> <ray_cluster_size> \
    <server_ready_timeout_s> <ray_ready_timeout_s> <concurrency> <num_prompts> \
    <result_dir> <result_filename> <server_log> <bench_log>
USAGE
}

if [[ "$#" -ne 16 ]]; then
  usage
  exit 2
fi

MODEL="$1"
TP="$2"
ISL="$3"
OSL="$4"
MAX_MODEL_LEN="$5"
PORT="$6"
RAY_PORT="$7"
RAY_CLUSTER_SIZE="$8"
SERVER_READY_TIMEOUT="$9"
RAY_READY_TIMEOUT="${10}"
CONCURRENCY="${11}"
NUM_PROMPTS="${12}"
RESULT_DIR="${13}"
RESULT_FILENAME="${14}"
SERVER_LOG="${15}"
BENCH_LOG="${16}"

mkdir -p "$RESULT_DIR"

export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
if [[ "$MODEL" == *"gpt-oss"* ]]; then
  export VLLM_MXFP4_USE_MARLIN=1
fi

SERVER_PID=""

cleanup() {
  set +e
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  ray stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

active_ray_nodes() {
  python3 - <<'PY'
import ray

count = 0
try:
    ray.init(address="auto", logging_level=50)
    count = sum(1 for n in ray.nodes() if n.get("Alive"))
except Exception:
    count = 0
finally:
    try:
        ray.shutdown()
    except Exception:
        pass
print(count)
PY
}

echo "=== Starting Ray head (port ${RAY_PORT}) ==="
ray start --head --port "${RAY_PORT}" --include-dashboard=false >/dev/null

echo "Waiting for ${RAY_CLUSTER_SIZE} Ray node(s)..."
waited=0
while true; do
  active="$(active_ray_nodes)"
  if [[ "$active" =~ ^[0-9]+$ ]] && (( active >= RAY_CLUSTER_SIZE )); then
    echo "Ray cluster ready: ${active} node(s)"
    break
  fi
  if (( waited >= RAY_READY_TIMEOUT )); then
    echo "ERROR: Timeout waiting for Ray cluster (${waited}s)." >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
  echo "  waiting... ${waited}s (${active}/${RAY_CLUSTER_SIZE})"
done

echo "=== Starting vLLM server (TP=${TP}) ==="
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1024 \
  --disable-log-requests >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM server health endpoint..."
waited=0
while ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "ERROR: vLLM server exited before becoming healthy." >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( waited >= SERVER_READY_TIMEOUT )); then
    echo "ERROR: vLLM server did not become healthy within ${SERVER_READY_TIMEOUT}s." >&2
    tail -n 120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
  echo "  waiting... ${waited}s"
done

echo "vLLM server is healthy. Running bench serve..."
vllm bench serve \
  --model "$MODEL" \
  --backend vllm \
  --base-url "http://localhost:${PORT}" \
  --dataset-name random \
  --input-len "$ISL" \
  --output-len "$OSL" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --save-result \
  --result-dir "$RESULT_DIR" \
  --result-filename "$RESULT_FILENAME" > >(tee "$BENCH_LOG") 2>&1

echo "Benchmark completed: ${RESULT_DIR}/${RESULT_FILENAME}"
