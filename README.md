# Qwen3.6 35B NVFP4 on 1x DGX Spark

A practical recipe for running `unsloth/Qwen3.6-35B-A3B-NVFP4` on a single NVIDIA DGX Spark / GB10 with vLLM `0.26`, FP8 KV cache, native GB10 kernels, and speculative decoding.

This repo is the companion to my Laguna S 2.1 Spark recipe, but for the Qwen3.6 35B-A3B NVFP4 path. The goal is simple: a fast, useful, OpenAI-compatible local coding/agent model on one Spark.

## Why this setup matters

Qwen3.6 35B-A3B NVFP4 is a strong fit for Spark because it is MoE: the total model is large, but only a smaller active slice runs per token. With the right vLLM/FlashInfer path, it can be fast enough for real interactive use.

The important trick is not just `vllm serve`.

The Unsloth Qwen3.6 NVFP4 checkpoint mixes FP8 dense layers with NVFP4 MoE/linear layers. Forcing `flashinfer_b12x` everywhere can crash on unsupported layer types unless the runtime has a soft fallback path. MiaAI-Lab’s GB10 image solves that by using B12X where it works and falling back to auto selection for other layers.

## Tested / target profile

```text
Hardware: NVIDIA DGX Spark / GB10 / 128GB unified memory
Runtime: vLLM 0.26-class GB10 image
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Context: 262,144 tokens
KV cache: fp8
Linear backend: flashinfer_b12x with soft fallback
Attention backend: flashinfer
Spec decode: MTP, 2 speculative tokens
Tool parser: qwen3_coder
Reasoning parser: qwen3
Vision: enabled, up to 4 images/request
```

## Published reference benchmark

MiaAI-Lab reported the following for this stack on one DGX Spark:

| Load | TTFT | Aggregate | Stream |
|---:|---:|---:|---:|
| 1x | 103 ms | 95.1 tok/s | 95.1 tok/s |
| 2x | 165 ms | 132.0 tok/s | 67.3 tok/s |
| 3x | 142 ms | 149.4 tok/s | 50.3 tok/s |
| 4x | 214 ms | 171.6 tok/s | 49.7 tok/s |
| 6x | 233 ms | 235.3 tok/s | 40.5 tok/s |
| 8x | 242 ms | 317.0 tok/s | 41.1 tok/s |

My own local benchmark slot is in [`results/`](results/). Rerun the benchmark script after startup and replace the placeholder with your real box numbers.

## Quickstart

```bash
git clone https://github.com/sojufx/Unsloth-Qwen3.6-35B-NVFP4-DGX-Spark-Recipe.git
cd Unsloth-Qwen3.6-35B-NVFP4-DGX-Spark-Recipe

export HF_TOKEN="optional-if-needed"
export VLLM_API_KEY="change-me"

./scripts/start-qwen.sh
```

OpenAI-compatible endpoint:

```text
http://127.0.0.1:8888/v1
```

## Main serve flags

```bash
--model unsloth/Qwen3.6-35B-A3B-NVFP4
--tensor-parallel-size 1
--trust-remote-code
--moe-backend auto
--linear-backend flashinfer_b12x
--attention-backend flashinfer
--kv-cache-dtype fp8
--max-model-len 262144
--max-num-seqs 24
--max-num-batched-tokens 32768
--enable-chunked-prefill
--async-scheduling
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"triton"}'
--tool-call-parser qwen3_coder
--enable-auto-tool-choice
--reasoning-parser qwen3
```

## Important notes

- Use a GB10/SM121-compatible vLLM image.
- Set `CUTE_DSL_ARCH=sm_121a`.
- Use FP8 KV for 256K context.
- Do not blindly force B12X on stock vLLM unless the runtime supports soft fallback for unsupported layer types.
- This model can support image inputs when the runtime and client send multimodal chat messages correctly.
- Keep this separate from your main production model until you have benchmarked it on your own box.

## Credits

This recipe is inspired by our own earlier Spark experiments and the excellent work from:

- [Unsloth Qwen3.6 35B NVFP4 model](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)
- [MiaAI-Lab DGX Spark recipe](https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark)
- [vLLM](https://github.com/vllm-project/vllm)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

## Files

- [`scripts/start-qwen.sh`](scripts/start-qwen.sh) — Docker/vLLM serve wrapper
- [`scripts/stop-qwen.sh`](scripts/stop-qwen.sh) — stop helper
- [`scripts/benchmark-qwen.py`](scripts/benchmark-qwen.py) — OpenAI-compatible benchmark
- [`systemd/qwen-vllm.service`](systemd/qwen-vllm.service) — optional autostart template
- [`docs/CONFIG.md`](docs/CONFIG.md) — flag notes
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — common failures
- [`results/benchmark-placeholder.md`](results/benchmark-placeholder.md) — replace with your own measured run
