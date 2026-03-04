#!/usr/bin/env bash
set -euo pipefail

MODEL="$1"
TP="$2"
ISL="$3"
OSL="$4"
MAX_MODEL_LEN="$5"
PORT="$6"
SWEEP_DIR="$7"
shift 7
CONCURRENCY_RANGE="$@"

export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
VLLM_SERVE_ENFORCE_EAGER="${VLLM_SERVE_ENFORCE_EAGER:-1}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"

if [[ "$MODEL" == *"gpt-oss"* ]]; then
  export VLLM_MXFP4_USE_MARLIN=1
fi

mkdir -p "$SWEEP_DIR"

SERVER_LOG="${SWEEP_DIR}/server.log"
SUMMARY_FILE="${SWEEP_DIR}/summary.txt"
SUMMARY_CSV="${SWEEP_DIR}/sweep_summary.csv"
SUMMARY_JSONL="${SWEEP_DIR}/sweep_summary.jsonl"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== Starting vLLM Server ==="
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
echo "vLLM Concurrency Sweep Results" >>"$SUMMARY_FILE"
echo "========================================" >>"$SUMMARY_FILE"
echo "Date: $(date)" >>"$SUMMARY_FILE"
echo "Model: $MODEL" >>"$SUMMARY_FILE"
echo "TP: $TP" >>"$SUMMARY_FILE"
echo "ISL: $ISL, OSL: $OSL" >>"$SUMMARY_FILE"
echo >>"$SUMMARY_FILE"
echo "Concurrency | Output tok/s | Total tok/s | Mean TTFT | Mean TPOT | P99 TPOT" >>"$SUMMARY_FILE"
echo "------------|--------------|-------------|-----------|-----------|----------" >>"$SUMMARY_FILE"

echo "model,tp,isl,osl,concurrency,num_prompts,request_throughput,output_throughput,total_token_throughput,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,gpu_util_mean_pct,gpu_util_p95_pct,mem_used_mean_mb,mem_used_max_mb,completed,failed" >"$SUMMARY_CSV"
: >"$SUMMARY_JSONL"

for CONC in $CONCURRENCY_RANGE; do
  echo
  echo "========================================"
  echo "=== Running Benchmark: Concurrency $CONC ==="
  echo "========================================"

  NUM_PROMPTS=$((CONC * 10))
  RESULT_JSON="conc${CONC}_isl${ISL}_osl${OSL}_tp${TP}.json"
  RESULT_TXT="conc${CONC}_bench.txt"
  TELEMETRY_CSV="conc${CONC}_telemetry.csv"

  # Capture GPU util/memory telemetry while the benchmark is running.
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,nounits >"${SWEEP_DIR}/${TELEMETRY_CSV}" || true

  # Run bench in background so we can sample telemetry concurrently.
  set +e
  vllm bench serve \
    --model "$MODEL" \
    --backend vllm \
    --base-url "http://localhost:$PORT" \
    --dataset-name random \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --save-result \
    --result-dir "$SWEEP_DIR" \
    --result-filename "$RESULT_JSON" > >(tee "${SWEEP_DIR}/${RESULT_TXT}") 2>&1 &
  BENCH_PID=$!

  while kill -0 "$BENCH_PID" >/dev/null 2>&1; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >>"${SWEEP_DIR}/${TELEMETRY_CSV}" || true
    sleep 1
  done

  wait "$BENCH_PID"
  BENCH_RC=$?
  set -e

  if [[ "$BENCH_RC" -ne 0 ]]; then
    echo "ERROR: vllm bench serve failed for concurrency ${CONC} (rc=${BENCH_RC})" >&2
    exit "$BENCH_RC"
  fi

  python3 - <<'PY' \
    "$MODEL" "$TP" "$ISL" "$OSL" "$CONC" "$NUM_PROMPTS" \
    "${SWEEP_DIR}/${RESULT_JSON}" "${SWEEP_DIR}/${TELEMETRY_CSV}" \
    "$SUMMARY_CSV" "$SUMMARY_JSONL" "$SUMMARY_FILE"
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

model, tp, isl, osl, conc, num_prompts = sys.argv[1:7]
result_path = Path(sys.argv[7])
telemetry_path = Path(sys.argv[8])
csv_out = Path(sys.argv[9])
jsonl_out = Path(sys.argv[10])
summary_txt = Path(sys.argv[11])

def pctl(vals, q):
    if not vals:
        return None
    xs = sorted(vals)
    # Nearest-rank percentile.
    k = max(0, min(len(xs) - 1, int(math.ceil((q / 100.0) * len(xs))) - 1))
    return xs[k]

data = json.loads(result_path.read_text())

# Parse telemetry (one row per GPU per sample).
tele = {}
if telemetry_path.exists():
    with telemetry_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            idx = row.get("index", "")
            if idx == "":
                continue
            try:
                gpu = int(idx)
            except ValueError:
                continue
            def to_f(key):
                v = row.get(key, "")
                try:
                    return float(v)
                except ValueError:
                    return None
            util = to_f("utilization.gpu [%]")
            mem_used = to_f("memory.used [MiB]")
            if util is None and mem_used is None:
                continue
            tele.setdefault(gpu, {"util_gpu": [], "mem_used": []})
            if util is not None:
                tele[gpu]["util_gpu"].append(util)
            if mem_used is not None:
                tele[gpu]["mem_used"].append(mem_used)

per_gpu = []
util_means = []
util_p95s = []
mem_means = []
mem_maxs = []
for gpu, series in sorted(tele.items()):
    util = series.get("util_gpu", [])
    mem = series.get("mem_used", [])
    u_mean = mean(util) if util else None
    u_p95 = pctl(util, 95) if util else None
    m_mean = mean(mem) if mem else None
    m_max = max(mem) if mem else None
    per_gpu.append(
        {
            "gpu": gpu,
            "util_gpu_mean_pct": u_mean,
            "util_gpu_p95_pct": u_p95,
            "mem_used_mean_mib": m_mean,
            "mem_used_max_mib": m_max,
        }
    )
    if u_mean is not None:
        util_means.append(u_mean)
    if u_p95 is not None:
        util_p95s.append(u_p95)
    if m_mean is not None:
        mem_means.append(m_mean)
    if m_max is not None:
        mem_maxs.append(m_max)

util_mean = mean(util_means) if util_means else None
util_p95 = mean(util_p95s) if util_p95s else None
mem_mean = mean(mem_means) if mem_means else None
mem_max = max(mem_maxs) if mem_maxs else None

row = {
    "model": model,
    "tp": int(tp),
    "isl": int(isl),
    "osl": int(osl),
    "concurrency": int(conc),
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
    "gpu_telemetry": {
        "per_gpu": per_gpu,
        "util_gpu_mean_pct": util_mean,
        "util_gpu_p95_pct": util_p95,
        "mem_used_mean_mib": mem_mean,
        "mem_used_max_mib": mem_max,
    },
    "result_json": str(result_path),
    "telemetry_csv": str(telemetry_path),
}

with jsonl_out.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")

def fnum(v):
    return "" if v is None else f"{v:.3f}"

with csv_out.open("a", encoding="utf-8", newline="") as f:
    f.write(",".join(
        [
            model,
            str(int(tp)),
            str(int(isl)),
            str(int(osl)),
            str(int(conc)),
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
            fnum(util_mean),
            fnum(util_p95),
            fnum(mem_mean),
            fnum(mem_max),
            str(row["completed"]),
            str(row["failed"]),
        ]
    ) + "\n")

# Human-readable table in summary.txt
out_toks = row["output_throughput"]
total_toks = row["total_token_throughput"]
mean_ttft = row["mean_ttft_ms"]
mean_tpot = row["mean_tpot_ms"]
p99_tpot = row["p99_tpot_ms"]
with summary_txt.open("a", encoding="utf-8") as f:
    f.write(f"{int(conc):<11d} | {out_toks:<12.2f} | {total_toks:<11.2f} | {mean_ttft:<9.2f} | {mean_tpot:<9.3f} | {p99_tpot:<9.3f}\n")
PY

  echo
  echo "Results saved to:"
  echo "  - ${SWEEP_DIR}/${RESULT_JSON}"
  echo "  - ${SWEEP_DIR}/${RESULT_TXT}"
  echo "  - ${SWEEP_DIR}/${TELEMETRY_CSV}"
done

echo
echo "========================================"
echo "=== Sweep Complete ==="
echo "========================================"
cat "$SUMMARY_FILE"
