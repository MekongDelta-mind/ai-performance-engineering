#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/run_fp4_checks_all_nodes.sh --hosts <h1,h2,...> --suite-dir <dir> [options]

Runs FP4 checks per host:
  1) DeepGEMM FP8xFP4 smoke/perf probe
  2) Cluster Perf grouped GEMM benchmark (DeepGEMM path)

Outputs are written under:
  results/raw/
  results/structured/
  docs/figures/

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional labels (must match host count)
  --suite-dir <dir>      Cluster Perf suite path on each host (required)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
  --remote-root <path>   Repo root on remote hosts (default: this repo root)
  --image <image>        Container image (default: ghcr.io/jordannanos/cmax-compute:latest)
  --preset <name>        Grouped-GEMM preset (default: auto; GB-family hosts use all)
  --warmup <n>           Grouped-GEMM warmup (default: 5)
  --iters <n>            Grouped-GEMM measured iterations (default: 30)
  --skip-smoke           Skip the FP4 smoke/perf probe step
  --smoke-m <int>        Smoke shape M (default: 4096)
  --smoke-n <int>        Smoke shape N (default: 4096)
  --smoke-k <int>        Smoke shape K (default: 4096)
  --smoke-warmup <n>     Smoke warmup (default: 10)
  --smoke-iters <n>      Smoke measured iterations (default: 30)

Bootstrap (recommended for reproducibility; default: enabled):
  --bootstrap-nodes                Run per-node bootstrap before FP4 checks
  --skip-bootstrap-nodes           Skip bootstrap
  --bootstrap-install-system-packages   Install missing system deps (default: on)
  --bootstrap-skip-system-packages      Skip system package installation
  --bootstrap-sync-code            Sync scripts/analysis/env requirements to remotes (default: on)
  --bootstrap-skip-sync-code       Skip code sync
  --bootstrap-install-python-deps  Ensure env/venv + python deps (default: on)
  --bootstrap-skip-python-deps     Skip python dependency install
  --bootstrap-torch-index-url <url>  Torch wheel index for bootstrap (default: cu130 index)
  --bootstrap-torch-version <ver>    Torch version for bootstrap fallback install (default: 2.9.1+cu130)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SUITE_DIR=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"
REMOTE_ROOT="${REMOTE_ROOT:-$ROOT_DIR}"
IMAGE="${CONTAINER_IMAGE:-ghcr.io/jordannanos/cmax-compute:latest}"
PRESET="auto"
WARMUP="5"
ITERS="30"
SKIP_SMOKE=0
SMOKE_M="4096"
SMOKE_N="4096"
SMOKE_K="4096"
SMOKE_WARMUP="10"
SMOKE_ITERS="30"
BOOTSTRAP_NODES=1
BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1
BOOTSTRAP_SYNC_CODE=1
BOOTSTRAP_INSTALL_PYTHON_DEPS=1
BOOTSTRAP_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
BOOTSTRAP_TORCH_VERSION="2.9.1+cu130"

while [[ $# -gt 0 ]]; do
  case "${1:-}" in
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --hosts) HOSTS="${2:-}"; shift 2 ;;
    --labels) LABELS="${2:-}"; shift 2 ;;
    --suite-dir) SUITE_DIR="${2:-}"; shift 2 ;;
    --ssh-user) SSH_USER="${2:-}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:-}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --preset) PRESET="${2:-}"; shift 2 ;;
    --warmup) WARMUP="${2:-}"; shift 2 ;;
    --iters) ITERS="${2:-}"; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-m) SMOKE_M="${2:-}"; shift 2 ;;
    --smoke-n) SMOKE_N="${2:-}"; shift 2 ;;
    --smoke-k) SMOKE_K="${2:-}"; shift 2 ;;
    --smoke-warmup) SMOKE_WARMUP="${2:-}"; shift 2 ;;
    --smoke-iters) SMOKE_ITERS="${2:-}"; shift 2 ;;
    --bootstrap-nodes) BOOTSTRAP_NODES=1; shift ;;
    --skip-bootstrap-nodes) BOOTSTRAP_NODES=0; shift ;;
    --bootstrap-install-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=1; shift ;;
    --bootstrap-skip-system-packages) BOOTSTRAP_INSTALL_SYSTEM_PACKAGES=0; shift ;;
    --bootstrap-sync-code) BOOTSTRAP_SYNC_CODE=1; shift ;;
    --bootstrap-skip-sync-code) BOOTSTRAP_SYNC_CODE=0; shift ;;
    --bootstrap-install-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=1; shift ;;
    --bootstrap-skip-python-deps) BOOTSTRAP_INSTALL_PYTHON_DEPS=0; shift ;;
    --bootstrap-torch-index-url) BOOTSTRAP_TORCH_INDEX_URL="${2:-}"; shift 2 ;;
    --bootstrap-torch-version) BOOTSTRAP_TORCH_VERSION="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "$SUITE_DIR" ]]; then
  echo "ERROR: --suite-dir is required" >&2
  usage >&2
  exit 2
fi

IFS=',' read -r -a HOST_ARR <<<"$HOSTS"
IFS=',' read -r -a LABEL_ARR <<<"$LABELS"
if [[ -n "$LABELS" && "${#LABEL_ARR[@]}" -ne "${#HOST_ARR[@]}" ]]; then
  echo "ERROR: --labels count must match --hosts count" >&2
  exit 2
fi

if [[ "$BOOTSTRAP_NODES" -eq 1 ]]; then
  bootstrap_args=(
    --run-id "${RUN_ID}"
    --hosts "${HOSTS}"
    --ssh-user "${SSH_USER}"
    --remote-root "${REMOTE_ROOT}"
    --sync-suite-dir "${SUITE_DIR}"
    --torch-index-url "${BOOTSTRAP_TORCH_INDEX_URL}"
    --torch-version "${BOOTSTRAP_TORCH_VERSION}"
  )
  if [[ -n "$LABELS" ]]; then
    bootstrap_args+=(--labels "${LABELS}")
  fi
  if [[ -n "$SSH_KEY" ]]; then
    bootstrap_args+=(--ssh-key "${SSH_KEY}")
  fi
  if [[ "$BOOTSTRAP_INSTALL_SYSTEM_PACKAGES" -eq 0 ]]; then
    bootstrap_args+=(--skip-system-packages)
  fi
  if [[ "$BOOTSTRAP_SYNC_CODE" -eq 0 ]]; then
    bootstrap_args+=(--skip-sync-code)
  fi
  if [[ "$BOOTSTRAP_INSTALL_PYTHON_DEPS" -eq 0 ]]; then
    bootstrap_args+=(--skip-python-deps)
  fi

  echo "Running bootstrap across hosts..."
  "${ROOT_DIR}/scripts/bootstrap_cluster_nodes.sh" "${bootstrap_args[@]}"
fi

sanitize_label() {
  local raw="$1"
  raw="${raw//./_}"
  raw="${raw//:/_}"
  echo "$raw"
}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=5
  -o ConnectionAttempts=3
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=3
  -o IdentitiesOnly=yes
  -o IdentityAgent=none
)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

run_remote() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${host}" "$@"
}

run_host_cmd() {
  local host="$1"
  local cmd="$2"
  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    bash -lc "$cmd"
  else
    run_remote "$host" "bash -lc $(printf '%q' "$cmd")"
  fi
}

write_platform_meta() {
  local out_path="$1"
  local host="$2"
  local label="$3"
  local requested_preset="$4"
  local selected_preset="$5"
  local gb_sku="$6"
  local suite_dir="$7"
  local image="$8"
  local gpu_names_b64="$9"

  python3 - "$out_path" "$host" "$label" "$requested_preset" "$selected_preset" "$gb_sku" "$suite_dir" "$image" "$gpu_names_b64" <<'PY'
import base64
import json
import sys
import time
from pathlib import Path

out_path, host, label, requested_preset, selected_preset, gb_sku, suite_dir, image, gpu_names_b64 = sys.argv[1:]
gpu_names = [line.strip() for line in base64.b64decode(gpu_names_b64).decode("utf-8").splitlines() if line.strip()]

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "host": host,
    "label": label,
    "gpu_names": gpu_names,
    "gb_family_detected": bool(gb_sku),
    "gb_sku": gb_sku or None,
    "fp4": {
        "requested_preset": requested_preset,
        "selected_preset": selected_preset,
        "suite_dir": suite_dir,
        "image": image,
    },
}

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

fetch_remote_artifact() {
  local host="$1"
  local rel_path="$2"
  local dst_dir
  dst_dir="${ROOT_DIR}/$(dirname "$rel_path")"
  mkdir -p "$dst_dir"
  scp "${SSH_OPTS[@]}" "${SSH_USER}@${host}:${REMOTE_ROOT}/${rel_path}" "${dst_dir}/" >/dev/null 2>&1 || true
}

for idx in "${!HOST_ARR[@]}"; do
  host="$(echo "${HOST_ARR[$idx]}" | xargs)"
  [[ -n "$host" ]] || continue

  label=""
  if [[ -n "$LABELS" ]]; then
    label="$(echo "${LABEL_ARR[$idx]}" | xargs)"
  fi
  if [[ -z "$label" ]]; then
    label="$(sanitize_label "$host")"
  fi

  echo "========================================"
  echo "FP4 checks: host=${host} label=${label}"
  echo "RUN_ID=${RUN_ID}"
  echo "SUITE_DIR=${SUITE_DIR}"
  echo "IMAGE=${IMAGE}"
  echo "========================================"

  gpu_names="$(run_host_cmd "$host" "nvidia-smi --query-gpu=name --format=csv,noheader")"
  gpu_names="${gpu_names//$'\r'/}"
  if [[ -z "$gpu_names" ]]; then
    echo "ERROR: unable to detect GPU names on host ${host}" >&2
    exit 2
  fi
  gb_sku="$(printf '%s\n' "$gpu_names" | grep -Eoi 'GB[0-9]{3}' | head -n 1 | tr '[:lower:]' '[:upper:]' || true)"
  host_preset="$PRESET"
  if [[ "$PRESET" == "auto" ]]; then
    host_preset="all"
  fi

  if [[ -z "$gb_sku" ]]; then
    echo "ERROR: FP4 checks require GB-family GPUs (GB200/GB300/...). Host ${host} reported:" >&2
    printf '%s\n' "$gpu_names" | sed 's/^/  - /' >&2
    exit 2
  fi

  echo "Detected GPUs:"
  printf '%s\n' "$gpu_names" | sed 's/^/  - /'
  echo "Detected GB family SKU: ${gb_sku}"
  echo "FP4 grouped preset: ${host_preset} (requested: ${PRESET})"

  platform_meta_rel="results/structured/${RUN_ID}_${label}_cluster_perf_fp4_platform.json"
  platform_meta_abs="${ROOT_DIR}/${platform_meta_rel}"
  gpu_names_b64="$(printf '%s' "$gpu_names" | base64 | tr -d '\n')"
  write_platform_meta "$platform_meta_abs" "$host" "$label" "$PRESET" "$host_preset" "$gb_sku" "$SUITE_DIR" "$IMAGE" "$gpu_names_b64"
  echo "Platform meta: ${platform_meta_rel}"

  smoke_args=(
    scripts/run_cluster_perf_fp4_smoke.sh
    --run-id "${RUN_ID}"
    --label "${label}"
    --image "${IMAGE}"
    --m "${SMOKE_M}"
    --n "${SMOKE_N}"
    --k "${SMOKE_K}"
    --warmup "${SMOKE_WARMUP}"
    --iters "${SMOKE_ITERS}"
  )
  grouped_args=(
    scripts/run_cluster_perf_grouped_gemm.sh
    --suite-dir "${SUITE_DIR}"
    --run-id "${RUN_ID}"
    --label "${label}"
    --preset "${host_preset}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --image "${IMAGE}"
  )

  smoke_str="$(printf '%q ' "${smoke_args[@]}")"
  grouped_str="$(printf '%q ' "${grouped_args[@]}")"
  remote_cmd="cd $(printf '%q' "${REMOTE_ROOT}") && "
  if [[ "$SKIP_SMOKE" -eq 0 ]]; then
    remote_cmd+="${smoke_str} && "
  fi
  remote_cmd+="${grouped_str}"
  run_host_cmd "$host" "$remote_cmd"

  if [[ "$host" != "localhost" && "$host" != "$(hostname)" ]]; then
    if [[ "$SKIP_SMOKE" -eq 0 ]]; then
      fetch_remote_artifact "$host" "results/raw/${RUN_ID}_${label}_cluster_perf_fp4_smoke.log"
      fetch_remote_artifact "$host" "results/structured/${RUN_ID}_${label}_cluster_perf_fp4_smoke.json"
      fetch_remote_artifact "$host" "results/structured/${RUN_ID}_${label}_cluster_perf_fp4_smoke_clock_lock.json"
    fi
    fetch_remote_artifact "$host" "results/structured/${RUN_ID}_${label}_cluster_perf_grouped_gemm.txt"
    fetch_remote_artifact "$host" "results/structured/${RUN_ID}_${label}_cluster_perf_grouped_gemm_summary.json"
    fetch_remote_artifact "$host" "results/structured/${RUN_ID}_${label}_cluster_perf_grouped_gemm_clock_lock.json"
    fetch_remote_artifact "$host" "docs/figures/${RUN_ID}_${label}_cluster_perf_grouped_gemm_tflops.png"
  fi
done

echo ""
echo "FP4 checks complete."
