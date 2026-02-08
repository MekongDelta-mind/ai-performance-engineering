#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  scripts/collect_discovery_and_tcp_sysctl.sh --hosts <h1,h2,...> [options]

Runs:
  - system discovery (`collect_system_info.sh`) on each host
  - tcp sysctl snapshot + diff
  - storage layout snapshot
  - writes a manifest JSON with file hashes for all structured artifacts for RUN_ID

Options:
  --run-id <id>          RUN_ID prefix (default: YYYY-MM-DD)
  --hosts <h1,h2,...>    Comma-separated host list (required)
  --labels <l1,l2,...>   Optional comma-separated labels (must match host count)
  --ssh-user <user>      SSH user (default: ubuntu)
  --ssh-key <path>       SSH key (default: $SSH_KEY)
EOF
}

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d)}"
HOSTS=""
LABELS=""
SSH_USER="ubuntu"
SSH_KEY="${SSH_KEY:-}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --hosts) HOSTS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --ssh-user) SSH_USER="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$HOSTS" ]]; then
  echo "ERROR: --hosts is required" >&2
  usage >&2
  exit 2
fi

common_args=(--run-id "$RUN_ID" --hosts "$HOSTS" --ssh-user "$SSH_USER")
if [[ -n "$LABELS" ]]; then
  common_args+=(--labels "$LABELS")
fi
if [[ -n "$SSH_KEY" ]]; then
  common_args+=(--ssh-key "$SSH_KEY")
fi

"${ROOT_DIR}/scripts/run_discovery_all_nodes.sh" "${common_args[@]}"

"${ROOT_DIR}/scripts/collect_software_versions_all_nodes.sh" "${common_args[@]}"

"${ROOT_DIR}/scripts/collect_container_runtime_all_nodes.sh" "${common_args[@]}"

"${ROOT_DIR}/scripts/collect_tcp_sysctl_all_nodes.sh" "${common_args[@]}"

"${ROOT_DIR}/scripts/collect_storage_layout_all_nodes.sh" "${common_args[@]}"

manifest_args=(--run-id "$RUN_ID" --hosts "$HOSTS")
if [[ -n "$LABELS" ]]; then
  manifest_args+=(--labels "$LABELS")
fi
python3 "${ROOT_DIR}/scripts/write_manifest.py" "${manifest_args[@]}"
