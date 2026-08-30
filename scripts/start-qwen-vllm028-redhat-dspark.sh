#!/usr/bin/env bash
set -euo pipefail

# Tested on a DGX Spark / GB10 with vLLM 0.28.0 and the RedHat DSpark K=8
# speculator. Set MODEL_ID and SPECULATOR_MODEL to pinned local snapshots when
# production reproducibility matters.
MODEL_ID="${MODEL_ID:-unsloth/Qwen3.6-35B-A3B-NVFP4}"
SPECULATOR_MODEL="${SPECULATOR_MODEL:-RedHatAI/Qwen3.6-35B-A3B-speculator.dspark}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-35b-nvfp4}"

export VLLM_TARGET_DEVICE=cuda
export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=2
export FLASHINFER_NVCC_THREADS=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-qwen36-35b-nvfp4-dspark-v028}"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --moe-backend marlin \
  --linear-backend auto \
  --attention-backend triton_attn \
  --gdn-prefill-backend triton \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.75 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 32 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --prefix-match-unit 16 \
  --async-scheduling \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$SPECULATOR_MODEL\",\"num_speculative_tokens\":8}" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}' \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --api-key "${VLLM_API_KEY:?Set VLLM_API_KEY}"
