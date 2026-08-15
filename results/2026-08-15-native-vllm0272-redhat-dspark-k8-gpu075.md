# Native vLLM 0.27.2 + RedHat DSpark K=8 + GPU 0.75 result

This result was measured on one NVIDIA DGX Spark / GB10 using native vLLM `0.27.2rc1.dev91+g1f7427bc0`.

This is the current recommended production profile for this recipe.

## Serve profile

```text
Hardware: NVIDIA DGX Spark / GB10 / 128GB unified memory
Runtime: native vLLM 0.27.2 nightly
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Speculator: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
Spec decode: DSpark, K=8
Context: 262,144 tokens
GPU memory utilization: 0.75
KV cache dtype: fp8
KV cache sizing: dynamic from gpu_memory_utilization
MoE backend: marlin
Attention backend: flashinfer
Linear backend: auto
Max num seqs: 16
Max batched tokens: 8192
Prefix caching: enabled
Prefix match unit: 16
Chunked prefill: enabled
Async scheduling: enabled
Thinking: enabled, default reasoning_effort=low
Tool parser: qwen3_coder
Reasoning parser: qwen3
```

## Startup metrics

vLLM reported:

```text
GPU KV cache size: 5,004,723 tokens
Maximum concurrency for 262,144 tokens per request: 19.09x
Available KV cache memory: 61.95 GiB
Actual model/activation/graph memory before KV: ~30.51 GiB
Graph capture memory: ~1.25 GiB
```

The major change from the older fixed-KV profile is removing the `12G` KV pin and using:

```bash
--gpu-memory-utilization 0.75
```

That gives this model a very large KV pool while still leaving enough headroom on the DGX Spark.

## Fixed decode benchmark

Benchmark command:

```bash
python3 /home/sojufx/benchmarks/vllm_fixed_decode_bench.py \
  --api http://127.0.0.1:8000/v1/chat/completions \
  --model ornith \
  --all-prompts \
  --max-tokens 256 \
  --temperature 0.7 \
  --concurrency 1,2,4,8 \
  --repeat 1
```

| Prompt | C1 | C2 aggregate | C4 aggregate | C8 aggregate |
|---|---:|---:|---:|---:|
| Agent/tool | 103.1 tok/s | 161.3 | 232.7 | 309.8 |
| Code edit | 110.9 tok/s | 171.4 | 274.4 | 361.3 |
| Generic | 92.8 tok/s | 158.0 | 233.2 | 325.9 |
| Long-context code | 109.6 tok/s | 142.8 | 243.0 | 368.9 |

## Runtime metrics observed

During the benchmark, vLLM reported:

```text
Running: 8 reqs
Waiting: 0 reqs
GPU KV cache usage: ~7.8%
Prefix cache hit rate: ~82.6%
```

Immediately before the benchmark, live traffic also showed:

```text
Avg generation throughput: ~128-129 tok/s
Running: 1 req
Waiting: 0 reqs
Prefix cache hit rate: ~82.6%
```

## Notes

- This profile prioritizes production headroom and multi-user stability over chasing the highest short-bench number.
- The older fixed `--kv-cache-memory-bytes 12G` profile produced a higher short C8 code-edit number in one run, but much lower full-context KV capacity.
- vLLM `0.27.x` should use `--moe-backend marlin` on this setup. Leaving MoE on `auto` previously selected an unstable FP4 MoE path.
- The first boot can take several minutes because FlashInfer autotune and CUDA graph capture run after weights load.
