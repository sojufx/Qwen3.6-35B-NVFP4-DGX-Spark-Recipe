# Native vLLM 0.27 + RedHat DSpark K=8 result

This result was measured on one NVIDIA DGX Spark / GB10 using native vLLM `0.27.1`.

## Serve profile

```text
Hardware: NVIDIA DGX Spark / GB10 / 128GB unified memory
Runtime: native vLLM 0.27.1
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Speculator: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
Spec decode: DSpark, K=8
Context: 262,144 tokens
KV cache dtype: fp8
KV cache memory: 12 GiB pinned
MoE backend: marlin
Attention backend: flashinfer
Max num seqs: 16
Max batched tokens: 8192
Prefix caching: enabled
Chunked prefill: enabled
Thinking: enabled
Tool parser: qwen3_coder
Reasoning parser: qwen3
```

## Why `--moe-backend marlin` matters

On this setup, leaving MoE backend on `auto` under vLLM `0.27.1` selected a CUTLASS FP4 MoE path and crashed on first request:

```text
RuntimeError: run_fp4_blockwise_scaled_group_mm_sm120
Failed to initialize GEMM: status=7
workspace_size=43008 num_experts=256 M=65536 N=1024 K=2048
```

Forcing:

```bash
--moe-backend marlin
```

made the same model and DSpark speculator boot and serve normally.

## Fixed decode benchmark

Benchmark command:

```bash
python3 /home/sojufx/benchmarks/vllm_fixed_decode_bench.py \
  --api http://127.0.0.1:8000/v1/chat/completions \
  --model ornith \
  --all-prompts \
  --max-tokens 256 \
  --temperature 0 \
  --concurrency 1,2,4,8 \
  --repeat 1
```

| Prompt | C1 | C2 aggregate | C4 aggregate | C8 aggregate |
|---|---:|---:|---:|---:|
| Agent/tool | 84.4 tok/s | 168.4 | 255.6 | 366.1 |
| Code edit | 114.6 tok/s | 197.0 | 287.7 | 417.8 |
| Generic | 99.0 tok/s | 136.8 | 239.5 | 324.4 |
| Long-context review | 134.7 tok/s | 187.4 | 272.1 | 412.8 |

## Runtime metrics observed

During the benchmark, vLLM reported:

```text
GPU KV cache usage: up to ~40.4% at C8, then back to 0.0%
Memory after benchmark: ~51 GiB used, ~70 GiB available
DSpark mean acceptance length: ~3.5-4.8
DSpark average draft acceptance: ~31-48%
```

## Notes

- This is a short fixed-output benchmark, not a full agent benchmark.
- Code-shaped prompts were fastest in this run.
- Long-context production behavior still depends on active user count, prompt length, and prefix-cache reuse.
- If tool calling behaves strangely, inspect the structured-output parser path first; one run logged non-fatal `backend_xgrammar` FSM warnings while still returning HTTP 200.
