# Configuration notes

## Current native vLLM 0.28.0 + RedHat DSpark production profile

```bash
--max-model-len 262144
--max-num-seqs 16
--max-num-batched-tokens 8192
--max-cudagraph-capture-size 32
--gpu-memory-utilization 0.75
--moe-backend marlin
--linear-backend auto
--attention-backend triton_attn
--gdn-prefill-backend triton
--kv-cache-dtype fp8_e4m3
--enable-chunked-prefill
--enable-prefix-caching
--prefix-match-unit 16
--async-scheduling
--speculative-config '{"method":"dspark","model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","num_speculative_tokens":8}'
```

This is the retained production profile from native vLLM `0.28.0`. It keeps medium thinking while using deterministic sampling for stronger DSpark acceptance:

```bash
--default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
--override-generation-config '{"temperature":0.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
```

The startup and benchmark figures below are from the earlier vLLM `0.27.2rc1.dev91+g1f7427bc0` profile, not vLLM 0.28.

Observed startup:

```text
GPU KV cache size: 5,004,723 tokens
Maximum concurrency for 262,144 tokens per request: 19.09x
Available KV cache memory: 61.95 GiB
```

Observed fixed decode benchmark at `temperature=0.7`:

```text
Single stream: ~93-111 tok/s
C4 aggregate: ~233-274 tok/s
C8 aggregate: ~310-369 tok/s
Peak C8 aggregate: 368.9 tok/s
```

The important part is `--moe-backend marlin`. On vLLM 0.27, `auto` selected a CUTLASS FP4 MoE path on my Spark and crashed on first request.

The second important production change is letting `--gpu-memory-utilization 0.75` size the KV cache instead of pinning `--kv-cache-memory-bytes 12G`. The old 12G pin can still be safer for memory-constrained testing, but the 0.75 profile gives much more full-context headroom on this model.

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

## Safer lower-memory variant

```bash
--max-model-len 262144
--max-num-seqs 8
--max-num-batched-tokens 8192
--kv-cache-memory-bytes 10G
--kv-cache-dtype fp8
--enable-chunked-prefill
--enable-prefix-caching
--async-scheduling
```

Use this if you want the same 262K request window with less prefill pressure and lower multi-user throughput.

## Fixed-KV compatibility variant

If a newer vLLM/FlashInfer build becomes unstable with large dynamic KV allocation, fall back to:

```bash
--gpu-memory-utilization 0.65
--kv-cache-memory-bytes 12G
```

That older profile gave lower full-context concurrency but was useful while testing vLLM `0.27.1`.

## Higher-throughput variant

The tested profile used:

```bash
--max-num-seqs 24
--max-num-batched-tokens 32768
--kv-cache-memory-bytes 12G
```

This measured best in our short code-shaped concurrency run. If your Spark gets close to OOM or power-limited after crash testing, drop `--max-num-batched-tokens` first.

## DFlash speculative decode

```json
{
  "method": "dflash",
  "model": "z-lab/Qwen3.6-35B-A3B-DFlash",
  "num_speculative_tokens": 7,
  "draft_tensor_parallel_size": 1
}
```

K=7 was the stable choice in this recipe. Increasing K can look faster in short bursts, but it can waste compute if later draft positions are rarely accepted.

## RedHat DSpark speculative decode

```json
{
  "method": "dspark",
  "model": "RedHatAI/Qwen3.6-35B-A3B-speculator.dspark",
  "num_speculative_tokens": 8
}
```

K=8 is the profile tested with vLLM 0.27. During load, acceptance varied by prompt shape, but mean acceptance length was roughly 3.5-4.8 in the benchmark run.

## Tool use and reasoning

```bash
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
```

For lower latency, you can disable thinking in the chat template. For coding agents, test both.
