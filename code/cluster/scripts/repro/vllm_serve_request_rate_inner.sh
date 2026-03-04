#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
TP="$2"
ISL="$3"
OSL="$4"
MAX_MODEL_LEN="$5"
PORT="$6"
SWEEP_DIR="$7"
MAX_CONCURRENCY="$8"
NUM_PROMPTS="$9"
shift 9
REQUEST_RATE_RANGE="$@"

export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
VLLM_SERVE_ENFORCE_EAGER="${VLLM_SERVE_ENFORCE_EAGER:-1}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"

if [[ "$MODEL" == *"gpt-oss"* ]]; then
  export VLLM_MXFP4_USE_MARLIN=1
fi

if ! vllm bench serve --help=all 2>/dev/null | grep -q -- "--request-rate"; then
  echo "ERROR: installed vLLM does not support '--request-rate' for bench serve." >&2
  exit 2
fi

mkdir -p "$SWEEP_DIR"

SERVER_LOG="${SWEEP_DIR}/rate_server.log"
SUMMARY_FILE="${SWEEP_DIR}/rate_summary.txt"
SUMMARY_CSV="${SWEEP_DIR}/rate_sweep_summary.csv"
SUMMARY_JSONL="${SWEEP_DIR}/rate_sweep_summary.jsonl"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== Starting vLLM Server (request-rate sweep) ==="
SERVE_ARGS=(
  "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --gpu-memory-utilization 0.9
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs 1024
  --disable-log-requests
)

if [[ -z "$VLLM_KV_CACHE_MEMORY_BYTES" ]]; then
  total_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ "$total_mem_mib" =~ ^[0-9]+$ ]]; then
    kv_cache_mib=$(( total_mem_mib * 70 / 100 ))
    if [[ "$kv_cache_mib" -lt 512 ]]; then
      kv_cache_mib=512
    fi
    VLLM_KV_CACHE_MEMORY_BYTES=$(( kv_cache_mib * 1024 * 1024 ))
  fi
fi
if [[ -n "$VLLM_KV_CACHE_MEMORY_BYTES" ]]; then
  echo "Using --kv-cache-memory-bytes=${VLLM_KV_CACHE_MEMORY_BYTES}"
  SERVE_ARGS+=(--kv-cache-memory-bytes "$VLLM_KV_CACHE_MEMORY_BYTES")
fi

if [[ "$VLLM_SERVE_ENFORCE_EAGER" == "1" ]]; then
  echo "Enabling --enforce-eager for startup robustness."
  SERVE_ARGS+=(--enforce-eager)
fi

vllm serve "${SERVE_ARGS[@]}" >"$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"

echo "Waiting for server to be ready..."
MAX_WAIT=1200
WAITED=0
while ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  if [[ -f "$SERVER_LOG" ]] && grep -qE "Engine core initialization failed|AssertionError: Error in memory profiling|RuntimeError: Engine core initialization failed" "$SERVER_LOG"; then
    echo "ERROR: Server reported a fatal initialization error before becoming healthy"
    tail -120 "$SERVER_LOG" || true
    exit 1
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server died before becoming healthy"
    tail -100 "$SERVER_LOG" || true
    exit 1
  fi
  if [[ "$WAITED" -ge "$MAX_WAIT" ]]; then
    echo "ERROR: Server failed to start within ${MAX_WAIT}s"
    tail -100 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  echo "  Waiting... (${WAITED}s)"
done

echo "Server is ready!"
echo

echo "========================================" >"$SUMMARY_FILE"
echo "vLLM Request-Rate Sweep Results" >>"$SUMMARY_FILE"
echo "========================================" >>"$SUMMARY_FILE"
echo "Date: $(date)" >>"$SUMMARY_FILE"
echo "Model: $MODEL" >>"$SUMMARY_FILE"
echo "TP: $TP" >>"$SUMMARY_FILE"
echo "ISL: $ISL, OSL: $OSL" >>"$SUMMARY_FILE"
echo "Max concurrency cap: $MAX_CONCURRENCY" >>"$SUMMARY_FILE"
echo "Num prompts: $NUM_PROMPTS" >>"$SUMMARY_FILE"
echo >>"$SUMMARY_FILE"
echo "Req/s | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | P99 TTFT | P99 TPOT" >>"$SUMMARY_FILE"
echo "------|--------------|-------------|-----------|-----------|----------|----------" >>"$SUMMARY_FILE"

echo "model,tp,isl,osl,request_rate,max_concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,completed,failed" >"$SUMMARY_CSV"
: >"$SUMMARY_JSONL"

for RATE in $REQUEST_RATE_RANGE; do
  echo
  echo "========================================"
  echo "=== Running Benchmark: Request rate ${RATE} req/s ==="
  echo "========================================"

  RESULT_JSON="rate${RATE}_isl${ISL}_osl${OSL}_tp${TP}.json"
  RESULT_TXT="rate${RATE}_bench.txt"

  set +e
  vllm bench serve \
    --model "$MODEL" \
    --backend vllm \
    --base-url "http://localhost:$PORT" \
    --dataset-name random \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --request-rate "$RATE" \
    --save-result \
    --result-dir "$SWEEP_DIR" \
    --result-filename "$RESULT_JSON" > >(tee "${SWEEP_DIR}/${RESULT_TXT}") 2>&1
  BENCH_RC=$?
  set -e

  if [[ "$BENCH_RC" -ne 0 ]]; then
    echo "ERROR: vllm bench serve failed for request-rate ${RATE} (rc=${BENCH_RC})" >&2
    exit "$BENCH_RC"
  fi

  python3 - <<'PY' \
    "$MODEL" "$TP" "$ISL" "$OSL" "$RATE" "$MAX_CONCURRENCY" "$NUM_PROMPTS" \
    "${SWEEP_DIR}/${RESULT_JSON}" \
    "$SUMMARY_CSV" "$SUMMARY_JSONL" "$SUMMARY_FILE"
import json
import sys
from pathlib import Path

model, tp, isl, osl, request_rate, max_conc, num_prompts = sys.argv[1:8]
result_path = Path(sys.argv[8])
csv_out = Path(sys.argv[9])
jsonl_out = Path(sys.argv[10])
summary_txt = Path(sys.argv[11])

data = json.loads(result_path.read_text())

row = {
    "model": model,
    "tp": int(tp),
    "isl": int(isl),
    "osl": int(osl),
    "request_rate": float(request_rate),
    "max_concurrency": int(max_conc),
    "num_prompts": int(num_prompts),
    "request_throughput": float(data.get("request_throughput", 0.0) or 0.0),
    "output_throughput": float(data.get("output_throughput", 0.0) or 0.0),
    "total_token_throughput": float(data.get("total_token_throughput", 0.0) or 0.0),
    "mean_ttft_ms": float(data.get("mean_ttft_ms", 0.0) or 0.0),
    "median_ttft_ms": float(data.get("median_ttft_ms", 0.0) or 0.0),
    "p99_ttft_ms": float(data.get("p99_ttft_ms", 0.0) or 0.0),
    "mean_tpot_ms": float(data.get("mean_tpot_ms", 0.0) or 0.0),
    "median_tpot_ms": float(data.get("median_tpot_ms", 0.0) or 0.0),
    "p99_tpot_ms": float(data.get("p99_tpot_ms", 0.0) or 0.0),
    "completed": int(data.get("completed", 0) or 0),
    "failed": int(data.get("failed", 0) or 0),
    "result_json": str(result_path),
}

with jsonl_out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")

def fnum(v):
    return f"{float(v):.6f}"

with csv_out.open("a", encoding="utf-8") as f:
    f.write(",".join(
        [
            model,
            str(int(tp)),
            str(int(isl)),
            str(int(osl)),
            fnum(row["request_rate"]),
            str(int(max_conc)),
            str(int(num_prompts)),
            fnum(row["request_throughput"]),
            fnum(row["output_throughput"]),
            fnum(row["total_token_throughput"]),
            fnum(row["mean_ttft_ms"]),
            fnum(row["median_ttft_ms"]),
            fnum(row["p99_ttft_ms"]),
            fnum(row["mean_tpot_ms"]),
            fnum(row["median_tpot_ms"]),
            fnum(row["p99_tpot_ms"]),
            str(row["completed"]),
            str(row["failed"]),
        ]
    ) + "\n")

with summary_txt.open("a", encoding="utf-8") as f:
    f.write(
        f"{float(request_rate):<6.2f} | {row['output_throughput']:<12.2f} | {row['total_token_throughput']:<11.2f} | "
        f"{row['mean_ttft_ms']:<9.2f} | {row['mean_tpot_ms']:<9.3f} | {row['p99_ttft_ms']:<8.2f} | {row['p99_tpot_ms']:<8.3f}\n"
    )
PY

  echo
  echo "Results saved to:"
  echo "  - ${SWEEP_DIR}/${RESULT_JSON}"
  echo "  - ${SWEEP_DIR}/${RESULT_TXT}"
done

echo
echo "========================================"
echo "=== Request-rate Sweep Complete ==="
echo "========================================"
cat "$SUMMARY_FILE"
