#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="/home/sojufx/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4/snapshots/739af1e7aac320af1682ed1e0cce369af4c5265d"
DRAFT_MODEL="/home/sojufx/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/f181eece646affea2c38b2765f1aaa01a9734ccd"
PYTHON_BIN="/home/sojufx/vllm-026/bin/python"

export VLLM_TARGET_DEVICE=cuda
export CUTE_DSL_ARCH=sm_121a
export MAX_JOBS=2
export FLASHINFER_NVCC_THREADS=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name ornith \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --moe-backend auto \
  --kv-cache-memory-bytes 12G \
  --linear-backend auto \
  --attention-backend flashinfer \
  --max-model-len 262144 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --kv-cache-dtype fp8 \
  --limit-mm-per-prompt '{"image":4}' \
  --allowed-media-domains='*' \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":1}" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  --api-key "${VLLM_API_KEY:?Set VLLM_API_KEY}"
