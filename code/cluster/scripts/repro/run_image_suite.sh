#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the full "screenshot reproduction" suite and capture results under results/.

Usage:
  scripts/repro/run_image_suite.sh [--run-id <id>] [--label <label>]
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="$(date +%Y-%m-%d)_image_suite"
LABEL="$(hostname)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

echo "== Screenshot Repro Suite =="
echo "RUN_ID=${RUN_ID}"
echo "LABEL=${LABEL}"
echo "ROOT_DIR=${ROOT_DIR}"
echo

"${ROOT_DIR}/scripts/repro/capture_image_commands.sh" \
  --run-id "$RUN_ID"

"${ROOT_DIR}/scripts/repro/run_allreduce_bench.sh" \
  --run-id "$RUN_ID" \
  --label "$LABEL"

"${ROOT_DIR}/scripts/repro/run_vllm_serve_sweep_container.sh" \
  --run-id "$RUN_ID" \
  --label "$LABEL"

echo
echo "Suite complete. Outputs:"
echo "  - results/raw/${RUN_ID}_*"
echo "  - results/structured/${RUN_ID}_*"

