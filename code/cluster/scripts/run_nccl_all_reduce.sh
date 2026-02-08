#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_nccl_all_reduce.sh [options]

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --gpus-per-node <n>    GPUs per node (default: all visible GPUs)
  --min-bytes <size>     Minimum message size (default: 1M)
  --max-bytes <size>     Maximum message size (default: 64M)
  --warmup <n>           Warmup iterations (default: 5)
  --iters <n>            Measurement iterations (default: 20)
  --label <label>        Label for raw log (default: allnodes)
  --socket-ifname <if>   Export NCCL_SOCKET_IFNAME to MPI ranks (default: $NCCL_SOCKET_IFNAME)
  --nccl-ib-hca <list>   Export NCCL_IB_HCA allowlist to MPI ranks (default: $NCCL_IB_HCA)
  --ssh-key <path>       SSH key for mpirun (default: $SSH_KEY)
  --oob-if <iface>       Interface for OpenMPI OOB/TCP (default: $OOB_IF)
  -h, --help             Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)"
HOSTS=""
GPUS_PER_NODE=""
MIN_BYTES="1M"
MAX_BYTES="64M"
WARMUP=5
ITERS=20
LABEL="allnodes"
SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
IB_HCA="${NCCL_IB_HCA:-}"
SSH_KEY="${SSH_KEY:-}"
OOB_IF="${OOB_IF:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --hosts)
      HOSTS="$2"
      shift 2
      ;;
    --gpus-per-node)
      GPUS_PER_NODE="$2"
      shift 2
      ;;
    --min-bytes)
      MIN_BYTES="$2"
      shift 2
      ;;
    --max-bytes)
      MAX_BYTES="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --socket-ifname)
      SOCKET_IFNAME="$2"
      shift 2
      ;;
    --nccl-ib-hca)
      IB_HCA="$2"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    --oob-if)
      OOB_IF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

OUT_RAW_DIR="${ROOT_DIR}/results/raw"
OUT_STRUCT_DIR="${ROOT_DIR}/results/structured"
HOSTFILE="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_hosts.txt"
RAW_LOG="${OUT_RAW_DIR}/${RUN_ID}_${LABEL}_nccl_all_reduce.log"
STRUCT_OUT="${OUT_STRUCT_DIR}/${RUN_ID}_nccl.json"

mkdir -p "$OUT_RAW_DIR" "$OUT_STRUCT_DIR"

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
if [[ "${#HOST_ARR[@]}" -eq 0 ]]; then
  echo "No hosts specified" >&2
  exit 1
fi

TOTAL_RANKS=$((GPUS_PER_NODE * ${#HOST_ARR[@]}))

printf "%s\n" "${HOST_ARR[@]}" > "$HOSTFILE"

MPI_PY="${ROOT_DIR}/env/venv/bin/python"
WRAPPER="${ROOT_DIR}/scripts/nccl_lock_wrapper.py"
NCCL_BIN="${ROOT_DIR}/tools/nccl-tests/build/all_reduce_perf"

if [[ ! -x "$MPI_PY" ]]; then
  echo "Missing python venv at $MPI_PY" >&2
  exit 1
fi
if [[ ! -x "$NCCL_BIN" ]]; then
  echo "Missing nccl-tests binary at $NCCL_BIN" >&2
  exit 1
fi

MPIRUN_CMD=(
  mpirun
  --hostfile "$HOSTFILE"
  --map-by "ppr:${GPUS_PER_NODE}:node"
  --bind-to none
  --mca routed direct
)

if [[ -n "$SSH_KEY" ]]; then
  MPIRUN_CMD+=(
    --mca plm_rsh_args
    "-i ${SSH_KEY} -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ConnectTimeout=8 -o ConnectionAttempts=3"
  )
fi
if [[ -n "$OOB_IF" ]]; then
  MPIRUN_CMD+=(--mca oob_tcp_if_include "$OOB_IF" --mca btl_tcp_if_include "$OOB_IF")
fi

MPIRUN_CMD+=(
  -np "$TOTAL_RANKS"
  -x "NCCL_DEBUG=INFO"
  -x "NCCL_DEBUG_SUBSYS=INIT"
  -x "PATH=$PATH"
  -x "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
)

MPIRUN_CMD+=(-x "REQUIRE_CLOCK_LOCK=1")
if [[ -n "$SOCKET_IFNAME" ]]; then
  MPIRUN_CMD+=(-x "NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}")
fi
if [[ -n "$IB_HCA" ]]; then
  MPIRUN_CMD+=(-x "NCCL_IB_HCA=${IB_HCA}")
fi

MPIRUN_CMD+=(
  "$MPI_PY"
  "$WRAPPER"
  --
  "$NCCL_BIN"
  -b "$MIN_BYTES"
  -e "$MAX_BYTES"
  -f 2
  -g 1
  -w "$WARMUP"
  -n "$ITERS"
)

echo "== NCCL all-reduce sanity =="
echo "RUN_ID=$RUN_ID"
echo "HOSTS=$HOSTS"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "TOTAL_RANKS=$TOTAL_RANKS"
echo "RAW_LOG=$RAW_LOG"
if [[ -n "$SOCKET_IFNAME" ]]; then
  echo "NCCL_SOCKET_IFNAME=$SOCKET_IFNAME"
fi
if [[ -n "$IB_HCA" ]]; then
  echo "NCCL_IB_HCA=$IB_HCA"
fi

set +e
"${MPIRUN_CMD[@]}" 2>&1 | tee "$RAW_LOG"
MPIRUN_RC=${PIPESTATUS[0]}
set -e

if [[ "$MPIRUN_RC" -ne 0 ]]; then
  echo "mpirun failed with code ${MPIRUN_RC}" >&2
  exit "$MPIRUN_RC"
fi

CMD_STRING=$(printf "%q " "${MPIRUN_CMD[@]}")

"${ROOT_DIR}/scripts/parse_nccl_log.py" \
  --input "$RAW_LOG" \
  --output "$STRUCT_OUT" \
  --run-id "$RUN_ID" \
  --hosts "$HOSTS" \
  --gpus-per-node "$GPUS_PER_NODE" \
  --command "$CMD_STRING"

echo "Wrote $STRUCT_OUT"
