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

## Tool use and reasoning

```bash
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
```

For lower latency, you can disable thinking in the chat template. For coding agents, test both.
