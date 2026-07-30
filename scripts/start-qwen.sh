#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
IMAGE="${IMAGE:-ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-35b-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-35b}"
HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} is already running"
  echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Pulling image if needed: ${IMAGE}"
docker pull "${IMAGE}"

echo "Starting ${MODEL_ID} on ${HOST}:${PORT}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --user root \
  --network host \
  --ipc host \
  --shm-size=32g \
  --ulimit memlock=-1:-1 \
  --cap-add=IPC_LOCK \
  --gpus all \
  --workdir /workspace \
  --entrypoint /usr/local/bin/vllm \
  -e VLLM_TARGET_DEVICE=cuda \
  -e CUTE_DSL_ARCH=sm_121a \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "$(pwd):/workspace" \
  "${IMAGE}" \
  serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --moe-backend auto \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
    --linear-backend flashinfer_b12x \
    --attention-backend flashinfer \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --max-num-seqs "${MAX_NUM_SEQS:-24}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
    --enable-chunked-prefill \
    --async-scheduling \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt '{"image":4}' \
    --allowed-media-domains '*' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}' \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    ${VLLM_API_KEY:+--api-key "${VLLM_API_KEY}"}

echo "Waiting for readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited before readiness"
    docker logs "${CONTAINER_NAME}" || true
    exit 1
  fi
  sleep 2
done

echo "Ready."
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
