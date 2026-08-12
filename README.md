# Qwen3.6 35B NVFP4 on 1x DGX Spark

A native vLLM recipe for running `unsloth/Qwen3.6-35B-A3B-NVFP4` on a single NVIDIA DGX Spark / GB10 with 262K context, FP8 KV cache, speculative decoding, tool/reasoning parser support, and strong multi-session throughput.

> Native vLLM. One Spark. 262K context. 417 tok/s aggregate.

![Qwen3.6 35B NVFP4 DGX Spark benchmark](assets/qwen36-spark-benchmark-card.png)

## Why this setup matters

Qwen3.6 35B-A3B NVFP4 is a strong fit for Spark because it is MoE: the model is large, but only a smaller active slice runs per token. With FP8 KV cache and DFlash speculative decoding, it can deliver both long context and serious aggregate throughput on one 128GB unified-memory box.

This recipe is focused on the native vLLM path:

- no custom container required
- existing venv/systemd production style
- cached local model snapshots supported
- 262,144-token context
- two tested speculative paths:
  - vLLM `0.26` + DFlash K=7
  - vLLM `0.27` + RedHatAI DSpark K=8
- up to 417.8 tok/s aggregate at 8 sessions in my fixed decode benchmark

The main value is reproducibility: this is not just a launch command, it includes the exact serve shape, autostart template, smoke test, and concurrency benchmark so other Spark owners can verify their own box instead of guessing.

## Current tested production profile: vLLM 0.27 + RedHat DSpark

```text
Hardware: NVIDIA DGX Spark / GB10 / 128GB unified memory
Runtime: native vLLM 0.27.1 venv
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Draft: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
Context: 262,144 tokens
KV cache: fp8
KV cache memory: 12 GiB pinned
MoE backend: marlin
Linear backend: auto
Attention backend: flashinfer
Spec decode: DSpark, K=8
Tool parser: qwen3_coder
Reasoning parser: qwen3
Thinking: enabled
```

### vLLM 0.27 + RedHat DSpark benchmark

Measured on one DGX Spark using native vLLM `0.27.1`, `--moe-backend marlin`, FP8 KV, and `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark` at K=8.

| Prompt | C1 | C2 aggregate | C4 aggregate | C8 aggregate |
|---|---:|---:|---:|---:|
| Agent/tool | 84.4 tok/s | 168.4 | 255.6 | 366.1 |
| Code edit | 114.6 tok/s | 197.0 | 287.7 | 417.8 |
| Generic | 99.0 tok/s | 136.8 | 239.5 | 324.4 |
| Long-context review | 134.7 tok/s | 187.4 | 272.1 | 412.8 |

Runtime notes from the same run:

```text
Max model length: 262,144
Pinned KV cache: 12 GiB
Memory after benchmark: ~51 GiB used, ~70 GiB available
DSpark mean acceptance length during load: ~3.5-4.8
DSpark draft acceptance during load: ~31-48%
```

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

./scripts/start-qwen-native-vllm027-redhat-dspark.sh
```

OpenAI-compatible endpoint:

```text
http://127.0.0.1:8000/v1
```

## Main serve flags

Current vLLM `0.27` + RedHat DSpark profile:

```bash
--model unsloth/Qwen3.6-35B-A3B-NVFP4
--tensor-parallel-size 1
--trust-remote-code
--moe-backend marlin
--linear-backend auto
--attention-backend flashinfer
--kv-cache-dtype fp8
--kv-cache-memory-bytes 12G
--max-model-len 262144
--max-num-seqs 16
--max-num-batched-tokens 8192
--enable-chunked-prefill
--enable-prefix-caching
--async-scheduling
--speculative-config '{"method":"dspark","model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","num_speculative_tokens":8}'
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
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

- [`scripts/start-qwen-native-vllm027-redhat-dspark.sh`](scripts/start-qwen-native-vllm027-redhat-dspark.sh) — current vLLM 0.27 + DSpark launcher
- [`scripts/start-qwen-native-vllm026.sh`](scripts/start-qwen-native-vllm026.sh) — earlier native vLLM 0.26 + DFlash launcher
- [`scripts/stop-qwen.sh`](scripts/stop-qwen.sh) — stop helper
- [`scripts/benchmark-qwen.py`](scripts/benchmark-qwen.py) — OpenAI-compatible benchmark
- [`systemd/qwen-vllm.service`](systemd/qwen-vllm.service) — optional autostart template
- [`docs/CONFIG.md`](docs/CONFIG.md) — flag notes
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — common failures
- [`results/2026-08-11-native-vllm027-redhat-dspark-k8.md`](results/2026-08-11-native-vllm027-redhat-dspark-k8.md) — measured vLLM 0.27 + DSpark result
- [`results/2026-07-30-native-vllm026-dflash-k7.md`](results/2026-07-30-native-vllm026-dflash-k7.md) — measured native vLLM result
