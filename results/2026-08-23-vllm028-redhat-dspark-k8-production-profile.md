# vLLM 0.28.0 + RedHat DSpark K=8 production profile

This is the exact production serve profile retained after migrating Qwen3.6 35B NVFP4 from the vLLM 0.27.2 path to vLLM 0.28.0 on one DGX Spark / GB10.

```text
Model: unsloth/Qwen3.6-35B-A3B-NVFP4
Draft: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark
Context: 262,144 tokens
KV cache: fp8_e4m3
GPU memory utilization: 0.75
Max sequences: 16
Max batched tokens: 8,192
Speculative decode: DSpark K=8
MoE backend: marlin
Attention: triton_attn
GDN prefill: triton
Reasoning: qwen3, medium thinking enabled by default
Tool calling: qwen3_coder with auto tool choice
```

The vLLM 0.27.2 benchmark tables remain in this repository as the measured throughput record. A fresh vLLM 0.28 benchmark was not retained, so this document intentionally does not attach old 0.27 numbers to the new runtime.

## Settings retained from production

```bash
--max-cudagraph-capture-size 32
--enable-chunked-prefill
--enable-prefix-caching
--prefix-match-unit 16
--async-scheduling
--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
```

Use `scripts/start-qwen-vllm028-redhat-dspark.sh` for the launchable version. It accepts model IDs by default and supports pinned local snapshots through `MODEL_ID` and `SPECULATOR_MODEL` environment variables.
