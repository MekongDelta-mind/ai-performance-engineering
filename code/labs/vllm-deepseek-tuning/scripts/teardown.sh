#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results"
PLOTS_DIR="${ROOT_DIR}/plots"
REPORTS_DIR="${ROOT_DIR}/reports"

echo "[teardown] Stopping vLLM processes if running..."
pkill -f "vllm serve" >/dev/null 2>&1 || true
sleep 1
pkill -9 -f "vllm serve" >/dev/null 2>&1 || true

echo "[teardown] Clearing CUDA visible setting from shell context (no-op for parent shells)"
unset CUDA_VISIBLE_DEVICES || true

if [[ "${1:-}" == "--purge-results" ]]; then
  echo "[teardown] Purging results/, plots/, and reports/"
  rm -rf "${RESULTS_DIR}"/* "${PLOTS_DIR}"/* "${REPORTS_DIR}"/* 2>/dev/null || true
fi

echo "[teardown] Done."
