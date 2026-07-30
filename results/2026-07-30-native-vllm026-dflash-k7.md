# Benchmark: native vLLM 0.26, DFlash K7

Measured on 2026-07-30.

```text
Hardware: 1x NVIDIA DGX Spark / GB10 / 128GB unified memory
Runtime: native vLLM 0.26.0 venv
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Draft: z-lab/Qwen3.6-35B-A3B-DFlash
Context: 262,144 tokens
KV cache dtype: fp8
KV cache memory pin: 12 GiB
Speculative decode: DFlash K=7
Linear backend: auto
Attention backend: flashinfer
Thinking: enabled / preserved
Tool parser: qwen3_coder
Reasoning parser: qwen3
Prompt shape: short code-shaped prompt, exact 500-token outputs
```

## Startup

```text
GPU KV cache size: 647,182 tokens
Maximum concurrency for 262,144 tokens/request: 2.47x
Selected NVFP4 GEMM: CutlassNvFp4LinearKernel
```

## Decode benchmark

| Concurrency | Aggregate tok/s | Mean stream tok/s | Completion tokens |
|---:|---:|---:|---:|
| 1 | 73.4 | 73.4 | 500 |
| 2 | 147.4 | 74.2 | 1,000 |
| 3 | 190.6 | 64.2 | 1,500 |
| 4 | 238.0 | 61.2 | 2,000 |
| 6 | 327.0 | 57.9 | 3,000 |
| 8 | 338.4 | 43.8 | 4,000 |

## Speculative decoding notes

During the benchmark, vLLM reported DFlash acceptance roughly in the mid-30% to low-40% range:

```text
Mean acceptance length: ~3.4-4.0
Average draft acceptance rate: ~35-42%
```

## Interpretation

This native vLLM 0.26 setup is strong for aggregate throughput while keeping the full 262K request window available. vLLM selected `CutlassNvFp4LinearKernel` for NVFP4 GEMM, FP8 KV kept the long-context memory footprint manageable, and DFlash K=7 provided useful speculative speedup.

The headline result is the multi-session shape: 73.4 tok/s at one stream and 338.4 tok/s aggregate at eight streams on this short code-shaped benchmark.
