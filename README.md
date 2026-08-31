# Qwen3.6 35B NVFP4 on 1x DGX Spark

A native vLLM recipe for running `unsloth/Qwen3.6-35B-A3B-NVFP4` on a single NVIDIA DGX Spark / GB10 with 262K context, FP8 KV cache, speculative decoding, tool/reasoning parser support, and strong multi-session throughput.

> Native vLLM 0.28. One Spark. 262K context. RedHat DSpark K=8.

![Qwen3.6 35B NVFP4 DGX Spark benchmark](assets/qwen36-spark-benchmark-card.png)

## Why this setup matters

Qwen3.6 35B-A3B NVFP4 is a strong fit for Spark because it is MoE: the model is large, but only a smaller active slice runs per token. With FP8 KV cache and speculative decoding, it can deliver both long context and serious aggregate throughput on one 128GB unified-memory box.

This recipe is focused on the native vLLM path:

- no custom container required
- existing venv/systemd production style
- cached local model snapshots supported
- 262,144-token context
- three retained speculative paths:
  - vLLM `0.26` + DFlash K=7
  - vLLM `0.27.2` + RedHatAI DSpark K=8
  - vLLM `0.28.0` + RedHatAI DSpark K=8
- up to 368.9 tok/s aggregate at 8 sessions in the measured vLLM 0.27.2 high-KV profile
- up to 417.8 tok/s aggregate in an earlier short-bench fixed-KV profile

The main value is reproducibility: this is not just a launch command, it includes the exact serve shape, autostart template, smoke test, and concurrency benchmark so other Spark owners can verify their own box instead of guessing.

## Current retained production profile: vLLM 0.28.0 + RedHat DSpark

```text
Hardware: NVIDIA DGX Spark / GB10 / 128GB unified memory
Runtime: native vLLM 0.28.0
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Draft: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
Context: 262,144 tokens
KV cache: fp8
GPU memory utilization: 0.75
KV cache memory: dynamic, sized at startup
MoE backend: marlin
Linear backend: auto
Attention backend: triton_attn
GDN prefill backend: triton
Spec decode: DSpark, K=8
Tool parser: qwen3_coder
Reasoning parser: qwen3
Thinking: enabled, medium reasoning effort
Sampling: temperature 0.2, top_p 0.95, top_k 20
```

The 0.28 launch profile is preserved from production in [`scripts/start-qwen-vllm028-redhat-dspark.sh`](scripts/start-qwen-vllm028-redhat-dspark.sh). It uses `temperature 0.2` as the balanced production default after local C1/C4 testing. Its earlier recorded settings remain in [`results/2026-08-23-vllm028-redhat-dspark-k8-production-profile.md`](results/2026-08-23-vllm028-redhat-dspark-k8-production-profile.md). The measured figures below remain tied to 0.27.2.

### Measured vLLM 0.27.2 + RedHat DSpark benchmark

Measured on one DGX Spark using native vLLM `0.27.2rc1.dev91+g1f7427bc0`, `--gpu-memory-utilization 0.75`, `--moe-backend marlin`, FP8 KV, FlashInfer attention, prefix match unit 16, and `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` at K=8.

| Prompt | C1 | C2 aggregate | C4 aggregate | C8 aggregate |
|---|---:|---:|---:|---:|
| Agent/tool | 103.1 tok/s | 161.3 | 232.7 | 309.8 |
| Code edit | 110.9 tok/s | 171.4 | 274.4 | 361.3 |
| Generic | 92.8 tok/s | 158.0 | 233.2 | 325.9 |
| Long-context code | 109.6 tok/s | 142.8 | 243.0 | 368.9 |

Runtime notes from the same run:

```text
Max model length: 262,144
GPU KV cache size: 5,004,723 tokens
Maximum concurrency for 262,144 tokens/request: 19.09x
GPU KV cache usage during C8 benchmark: ~7.8%
Prefix cache hit rate: ~82.6%
Waiting queue during benchmark: 0
```

Full notes: [`results/2026-08-15-native-vllm0272-redhat-dspark-k8-gpu075.md`](results/2026-08-15-native-vllm0272-redhat-dspark-k8-gpu075.md)

### Earlier vLLM 0.27.1 fixed-KV benchmark

The older `--kv-cache-memory-bytes 12G` profile produced a higher short code-edit aggregate in one run, but with much less full-context KV headroom.

| Prompt | C1 | C2 aggregate | C4 aggregate | C8 aggregate |
|---|---:|---:|---:|---:|
| Agent/tool | 84.4 tok/s | 168.4 | 255.6 | 366.1 |
| Code edit | 114.6 tok/s | 197.0 | 287.7 | 417.8 |
| Generic | 99.0 tok/s | 136.8 | 239.5 | 324.4 |
| Long-context review | 134.7 tok/s | 187.4 | 272.1 | 412.8 |

Full notes: [`results/2026-08-11-native-vllm027-redhat-dspark-k8.md`](results/2026-08-11-native-vllm027-redhat-dspark-k8.md)

## Earlier tested profile: vLLM 0.26 + DFlash

Measured on one DGX Spark using native vLLM `0.26.0`, `linear-backend auto`, FP8 KV, and `z-lab/Qwen3.6-35B-A3B-DFlash` at K=7.

| Load | Aggregate tok/s | Mean stream tok/s |
|---:|---:|---:|
| 1x | 73.4 | 73.4 |
| 2x | 147.4 | 74.2 |
| 3x | 190.6 | 64.2 |
| 4x | 238.0 | 61.2 |
| 6x | 327.0 | 57.9 |
| 8x | 338.4 | 43.8 |

Startup:

```text
GPU KV cache size: 647,182 tokens
Maximum concurrency for 262,144 tokens/request: 2.47x
Selected NVFP4 GEMM: CutlassNvFp4LinearKernel
```

Full notes: [`results/2026-07-30-native-vllm026-dflash-k7.md`](results/2026-07-30-native-vllm026-dflash-k7.md)

## Quickstart

```bash
git clone https://github.com/sojufx/Qwen3.6-35B-NVFP4-DGX-Spark-Recipe.git
cd Qwen3.6-35B-NVFP4-DGX-Spark-Recipe

export VLLM_API_KEY="change-me"

./scripts/start-qwen-vllm028-redhat-dspark.sh
```

OpenAI-compatible endpoint:

```text
http://127.0.0.1:8000/v1
```

## Main serve flags

Current vLLM `0.28.0` + RedHat DSpark profile:

```bash
--model unsloth/Qwen3.6-35B-A3B-NVFP4
--tensor-parallel-size 1
--trust-remote-code
--moe-backend marlin
--linear-backend auto
--attention-backend triton_attn
--gdn-prefill-backend triton
--kv-cache-dtype fp8
--gpu-memory-utilization 0.75
--max-model-len 262144
--max-num-seqs 16
--max-num-batched-tokens 8192
--max-cudagraph-capture-size 32
--enable-chunked-prefill
--enable-prefix-caching
--prefix-match-unit 16
--async-scheduling
--speculative-config '{"method":"dspark","model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","num_speculative_tokens":8}'
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
--default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
```

Earlier vLLM `0.26` + DFlash profile:

```bash
--model unsloth/Qwen3.6-35B-A3B-NVFP4
--tensor-parallel-size 1
--trust-remote-code
--moe-backend auto
--linear-backend auto
--attention-backend flashinfer
--kv-cache-dtype fp8
--kv-cache-memory-bytes 12G
--max-model-len 262144
--max-num-seqs 24
--max-num-batched-tokens 32768
--enable-chunked-prefill
--enable-prefix-caching
--async-scheduling
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":7,"draft_tensor_parallel_size":1}'
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
```

## Important notes

- Use a vLLM build with GB10/SM121 support.
- Set `CUTE_DSL_ARCH=sm_121a`.
- Use FP8 KV for 256K context.
- On vLLM `0.27`, force `--moe-backend marlin` for this model. Leaving MoE on auto selected a CUTLASS FP4 MoE path on my box and crashed at first request.
- This model can support image inputs when the runtime and client send multimodal chat messages correctly.
- The benchmark in this repo is a short fixed decode test. Run your own long-context benchmark before changing production.
- For long-context agent workloads, run your own benchmark before changing production.

## Credits

Built from our own DGX Spark experiments with:

- [Unsloth Qwen3.6 35B NVFP4 model](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)
- [z-lab Qwen3.6 35B DFlash draft](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
- [RedHatAI Qwen3.6 35B DSpark speculator](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-speculator.dspark)
- [vLLM](https://github.com/vllm-project/vllm)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

## Files

- [`scripts/start-qwen-vllm028-redhat-dspark.sh`](scripts/start-qwen-vllm028-redhat-dspark.sh) — current vLLM 0.28 + DSpark launcher
- [`scripts/start-qwen-native-vllm027-redhat-dspark.sh`](scripts/start-qwen-native-vllm027-redhat-dspark.sh) — measured vLLM 0.27.2 + DSpark launcher
- [`scripts/start-qwen-native-vllm026.sh`](scripts/start-qwen-native-vllm026.sh) — earlier native vLLM 0.26 + DFlash launcher
- [`scripts/stop-qwen.sh`](scripts/stop-qwen.sh) — stop helper
- [`scripts/benchmark-qwen.py`](scripts/benchmark-qwen.py) — OpenAI-compatible benchmark
- [`systemd/qwen-vllm028.service`](systemd/qwen-vllm028.service) — vLLM 0.28 autostart template
- [`systemd/qwen-vllm.service`](systemd/qwen-vllm.service) — vLLM 0.27 autostart template
- [`docs/CONFIG.md`](docs/CONFIG.md) — flag notes
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — common failures
- [`results/2026-08-15-native-vllm0272-redhat-dspark-k8-gpu075.md`](results/2026-08-15-native-vllm0272-redhat-dspark-k8-gpu075.md) — measured vLLM 0.27.2 + DSpark + GPU 0.75 result
- [`results/2026-08-23-vllm028-redhat-dspark-k8-production-profile.md`](results/2026-08-23-vllm028-redhat-dspark-k8-production-profile.md) — retained vLLM 0.28 production settings
- [`results/2026-08-11-native-vllm027-redhat-dspark-k8.md`](results/2026-08-11-native-vllm027-redhat-dspark-k8.md) — earlier vLLM 0.27 + fixed-KV DSpark result
- [`results/2026-07-30-native-vllm026-dflash-k7.md`](results/2026-07-30-native-vllm026-dflash-k7.md) — measured native vLLM result
