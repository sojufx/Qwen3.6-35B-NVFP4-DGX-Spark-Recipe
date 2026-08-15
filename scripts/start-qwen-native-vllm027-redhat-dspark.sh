#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
SPECULATOR_MODEL="${SPECULATOR_MODEL:-RedHatAI/Qwen3.6-35B-A3B-speculator.dspark}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/vllm-0272-nightly/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-35b}"

export VLLM_TARGET_DEVICE=cuda
export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=2
export FLASHINFER_NVCC_THREADS=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-qwen36-35b-nvfp4-dspark}"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --moe-backend marlin \
  --linear-backend auto \
  --attention-backend flashinfer \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --prefix-match-unit 16 \
  --async-scheduling \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$SPECULATOR_MODEL\",\"num_speculative_tokens\":8}" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"low"}' \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --api-key "${VLLM_API_KEY:?Set VLLM_API_KEY}"
