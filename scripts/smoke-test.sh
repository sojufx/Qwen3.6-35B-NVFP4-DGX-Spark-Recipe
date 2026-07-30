#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${SERVED_MODEL_NAME:-qwen36-35b}"

AUTH_ARGS=()
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

curl -sS "${AUTH_ARGS[@]}" "${BASE_URL}/models"
printf "\n"

curl -sS "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  "${BASE_URL}/chat/completions" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: OK\"}],\"max_tokens\":16,\"temperature\":0}"

printf "\n"
