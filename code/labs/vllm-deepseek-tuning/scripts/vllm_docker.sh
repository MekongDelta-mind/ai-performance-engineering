#!/usr/bin/env bash
set -euo pipefail

IMAGE="${VLLM_DOCKER_IMAGE:-vllm/vllm-openai:cu130-nightly}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "${HF_CACHE}"

env_args=()
while IFS='=' read -r key _; do
  env_args+=(--env "${key}")
done < <(env | grep -E '^(VLLM_|HF_|HUGGING_FACE_HUB_TOKEN=)')

exec docker run --rm --gpus all --network host \
  --entrypoint vllm \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${env_args[@]}" \
  "${IMAGE}" "$@"
