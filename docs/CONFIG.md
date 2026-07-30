# Configuration notes

## Native vLLM 0.26 profile tested locally

```bash
--max-model-len 262144
--max-num-seqs 24
--max-num-batched-tokens 32768
--kv-cache-memory-bytes 12G
--linear-backend auto
--attention-backend flashinfer
--kv-cache-dtype fp8
--enable-chunked-prefill
--enable-prefix-caching
--async-scheduling
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":7,"draft_tensor_parallel_size":1}'
```

This profile measured up to 338.4 tok/s aggregate at C8 on a short code-shaped benchmark.

## Patched B12X/MTP profile

```bash
--max-model-len 262144
--max-num-seqs 24
--max-num-batched-tokens 32768
--gpu-memory-utilization 0.80
--kv-cache-dtype fp8
--enable-chunked-prefill
--async-scheduling
```

This is an aggressive profile intended for the patched GB10 image path. If your Spark gets close to OOM, first reduce:

```bash
--gpu-memory-utilization 0.70
--max-num-batched-tokens 8192
--max-num-seqs 8
```

## B12X linear backend

```bash
--linear-backend flashinfer_b12x
```

This is useful on GB10 for NVFP4 GEMM, but the Unsloth Qwen3.6 NVFP4 checkpoint is mixed FP8 + NVFP4. A patched runtime is needed so unsupported layers fall back to auto selection instead of crashing.

## MTP speculative decode

```json
{
  "method": "mtp",
  "num_speculative_tokens": 2,
  "moe_backend": "triton"
}
```

MTP is built into the checkpoint path and is the recommended speculative decode mode for this recipe.

## Tool use and reasoning

```bash
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
```

For lower latency, you can disable thinking in the chat template. For coding agents, test both.
