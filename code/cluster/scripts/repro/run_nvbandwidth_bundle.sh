#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a dedicated nvbandwidth benchmark bundle with strict GPU clock locking.

Usage:
  scripts/repro/run_nvbandwidth_bundle.sh [options]

Options:
  --run-id <id>        RUN_ID prefix (default: YYYY-MM-DD)
  --label <label>      Label for artifact names (default: hostname -s)
  --runtime <mode>     host|container (default: host)
  --image <image>      Container image for runtime=container
                       (default: cfregly/cluster_perf_orig_parity:latest or $CONTAINER_IMAGE)
  --nvbw-bin <path>    nvbandwidth executable path (default: nvbandwidth)
  --quick              Run reduced testcase subset with lower samples for faster turnaround

Artifacts:
  - results/raw/<run_id>_<label>_nvbandwidth/nvbandwidth.log
  - results/structured/<run_id>_<label>_nvbandwidth.json
  - results/structured/<run_id>_<label>_nvbandwidth_sums.csv
  - results/structured/<run_id>_<label>_nvbandwidth_clock_lock.json
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)"
LABEL="$(hostname -s)"
RUNTIME="host"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-cfregly/cluster_perf_orig_parity:latest}"
NVBW_BIN="${NVBW_BIN:-nvbandwidth}"
QUICK=0

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --label) LABEL="${2:-}"; shift 2 ;;
    --runtime) RUNTIME="${2:-}"; shift 2 ;;
    --image) CONTAINER_IMAGE="${2:-}"; shift 2 ;;
    --nvbw-bin) NVBW_BIN="${2:-}"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${1}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$RUNTIME" != "host" && "$RUNTIME" != "container" ]]; then
  echo "ERROR: --runtime must be host or container (got: ${RUNTIME})" >&2
  exit 1
fi
if [[ "$RUNTIME" == "container" && -z "$CONTAINER_IMAGE" ]]; then
  echo "ERROR: --image is required for --runtime container." >&2
  exit 1
fi

RAW_DIR="${ROOT_DIR}/results/raw/${RUN_ID}_${LABEL}_nvbandwidth"
STRUCT_DIR="${ROOT_DIR}/results/structured"
mkdir -p "$RAW_DIR" "$STRUCT_DIR"

RAW_LOG="${RAW_DIR}/nvbandwidth.log"
SUMMARY_JSON="${STRUCT_DIR}/${RUN_ID}_${LABEL}_nvbandwidth.json"
SUMS_CSV="${STRUCT_DIR}/${RUN_ID}_${LABEL}_nvbandwidth_sums.csv"
LOCK_META="${STRUCT_DIR}/${RUN_ID}_${LABEL}_nvbandwidth_clock_lock.json"
NVBW_ARGS=()
if [[ "$QUICK" -eq 1 ]]; then
  NVBW_ARGS+=(
    -i 1
    -b 128
    -t host_to_device_memcpy_ce
    -t device_to_host_memcpy_ce
    -t device_to_device_memcpy_read_ce
    -t device_to_device_memcpy_write_ce
    -t device_to_device_bidirectional_memcpy_read_ce
    -t device_to_device_bidirectional_memcpy_write_ce
    -t all_to_host_memcpy_ce
    -t host_to_all_memcpy_ce
  )
fi

echo "========================================"
echo "nvbandwidth Bundle"
echo "========================================"
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "RUNTIME=${RUNTIME}"
if [[ "$RUNTIME" == "container" ]]; then
  echo "IMAGE=${CONTAINER_IMAGE}"
else
  echo "NVBW_BIN=${NVBW_BIN}"
fi
echo "QUICK=${QUICK}"
echo "RAW_LOG=${RAW_LOG}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "SUMS_CSV=${SUMS_CSV}"
echo "LOCK_META=${LOCK_META}"
echo ""

contains_ptx_toolchain_mismatch() {
  local log_path="$1"
  if [[ ! -f "$log_path" ]]; then
    return 1
  fi
  grep -qiE "cudaErrorUnsupportedPtxVersion|unsupported toolchain" "$log_path"
}

run_host_nvbandwidth() {
  RUN_ID="$RUN_ID" LABEL="$LABEL" \
    "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META" \
    -- "$NVBW_BIN" "${NVBW_ARGS[@]}" 2>&1 | tee "$RAW_LOG"
  return "${PIPESTATUS[0]}"
}

run_container_nvbandwidth() {
  local -a docker_args=(
    docker run --rm
    --gpus all
    --ipc=host
    --network host
    --ulimit memlock=-1
    --ulimit stack=67108864
  )
  if [[ -d /dev/infiniband ]]; then
    docker_args+=( -v /dev/infiniband:/dev/infiniband )
  fi
  if [[ -e /dev/nvidia_imex ]]; then
    docker_args+=( -v /dev/nvidia_imex:/dev/nvidia_imex )
  fi
  NVBW_ARGS_STR="$(printf ' %q' "${NVBW_ARGS[@]}")"
  docker_args+=( "${CONTAINER_IMAGE}" bash -lc "set -euo pipefail; ${NVBW_BIN}${NVBW_ARGS_STR}" )

  RUN_ID="$RUN_ID" LABEL="$LABEL" \
    "${ROOT_DIR}/scripts/run_with_gpu_clocks.sh" \
    --lock-meta-out "$LOCK_META" \
    -- "${docker_args[@]}" 2>&1 | tee "$RAW_LOG"
  return "${PIPESTATUS[0]}"
}

REQUESTED_RUNTIME="$RUNTIME"
EFFECTIVE_RUNTIME="$RUNTIME"
FALLBACK_USED=0
run_rc=0

if [[ "$REQUESTED_RUNTIME" == "host" ]]; then
  if ! command -v "$NVBW_BIN" >/dev/null 2>&1; then
    echo "ERROR: nvbandwidth binary not found on PATH: ${NVBW_BIN}" >&2
    exit 1
  fi
  set +e
  run_host_nvbandwidth
  run_rc=$?
  set -e

  if [[ "$run_rc" -ne 0 ]] && contains_ptx_toolchain_mismatch "$RAW_LOG"; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "ERROR: nvbandwidth host run failed with unsupported PTX toolchain and docker is unavailable for fallback." >&2
      exit "$run_rc"
    fi
    echo "Host nvbandwidth failed with unsupported PTX/toolchain. Retrying in container runtime." | tee -a "$RAW_LOG"
    FALLBACK_USED=1
    EFFECTIVE_RUNTIME="container_fallback"
    set +e
    run_container_nvbandwidth
    run_rc=$?
    set -e
  fi
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found; required for runtime=container." >&2
    exit 1
  fi
  set +e
  run_container_nvbandwidth
  run_rc=$?
  set -e
fi

if [[ "$run_rc" -ne 0 ]]; then
  echo "ERROR: nvbandwidth execution failed (requested_runtime=${REQUESTED_RUNTIME}, effective_runtime=${EFFECTIVE_RUNTIME}, rc=${run_rc})." >&2
  exit "$run_rc"
fi

python3 - <<'PY' "$RUN_ID" "$LABEL" "$RAW_LOG" "$LOCK_META" "$SUMMARY_JSON" "$SUMS_CSV" "$REQUESTED_RUNTIME" "$EFFECTIVE_RUNTIME" "$FALLBACK_USED" "$NVBW_BIN" "$CONTAINER_IMAGE" "$QUICK"
import csv
import json
import re
import sys
from pathlib import Path

(
    run_id,
    label,
    raw_log_path,
    lock_meta_path,
    summary_json_path,
    sums_csv_path,
    requested_runtime,
    effective_runtime,
    fallback_used,
    nvbw_bin,
    container_image,
    quick_flag,
) = sys.argv[1:]

raw_log = Path(raw_log_path)
lock_meta = Path(lock_meta_path)
summary_json = Path(summary_json_path)
sums_csv = Path(sums_csv_path)

sum_re = re.compile(r"^SUM\s+(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
gpu_re = re.compile(r"^Device\s+(\d+):\s+(.+)$")

sums = []
gpus = {}

for line in raw_log.read_text(encoding="utf-8", errors="replace").splitlines():
    m = sum_re.match(line.strip())
    if m:
        sums.append({"test": m.group(1), "sum_gbps": float(m.group(2))})
        continue
    gm = gpu_re.match(line.strip())
    if gm:
        gpus[gm.group(1)] = gm.group(2).strip()

def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

lock_payload = load_json(lock_meta) or {}
locks = lock_payload.get("locks") or []
clock_summary = {
    "returncode": lock_payload.get("returncode"),
    "device_count": len(locks),
    "all_devices_locked": bool(locks) and all(bool((x or {}).get("lock", {}).get("locked")) for x in locks),
    "application_sm_mhz": sorted(
        {
            int((x or {}).get("clocks", {}).get("applications_sm_mhz"))
            for x in locks
            if (x or {}).get("clocks", {}).get("applications_sm_mhz") is not None
        }
    ),
    "application_mem_mhz": sorted(
        {
            int((x or {}).get("clocks", {}).get("applications_mem_mhz"))
            for x in locks
            if (x or {}).get("clocks", {}).get("applications_mem_mhz") is not None
        }
    ),
}

sums_by_test = {entry["test"]: entry["sum_gbps"] for entry in sums}
peak_sum = max((entry["sum_gbps"] for entry in sums), default=None)

key_tests = [
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
    "device_to_device_memcpy_read_ce",
    "device_to_device_memcpy_write_ce",
    "device_to_device_bidirectional_memcpy_read_ce_total",
    "device_to_device_bidirectional_memcpy_write_ce_total",
    "all_to_host_memcpy_ce",
    "host_to_all_memcpy_ce",
    "all_to_all_memcpy_read_ce",
    "all_to_all_memcpy_write_ce",
]
key_sums = {name: sums_by_test.get(name) for name in key_tests if name in sums_by_test}

payload = {
    "run_id": run_id,
    "label": label,
    "status": "ok" if sums and clock_summary["all_devices_locked"] else "failed",
    "requested_runtime": requested_runtime,
    "effective_runtime": effective_runtime,
    "fallback_used": fallback_used == "1",
    "runtime": effective_runtime,
    "image": container_image if "container" in effective_runtime else None,
    "nvbandwidth_bin": nvbw_bin if effective_runtime == "host" else None,
    "quick": quick_flag == "1",
    "artifacts": {
        "raw_log": str(raw_log),
        "clock_lock": str(lock_meta),
        "sums_csv": str(sums_csv),
    },
    "clock_lock": clock_summary,
    "gpu_inventory": gpus,
    "sum_entries": sums,
    "sum_count": len(sums),
    "key_sum_gbps": key_sums,
    "peak_sum_gbps": peak_sum,
}

summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

with sums_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["test", "sum_gbps"])
    writer.writeheader()
    for row in sums:
        writer.writerow(row)
PY

echo "Wrote ${RAW_LOG}"
echo "Wrote ${SUMMARY_JSON}"
echo "Wrote ${SUMS_CSV}"
echo "Wrote ${LOCK_META}"
