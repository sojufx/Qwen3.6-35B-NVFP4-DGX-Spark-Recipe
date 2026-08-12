# Troubleshooting

## `no kernel image is available for execution on the device`

Make sure this is set before serving:

```bash
export CUTE_DSL_ARCH=sm_121a
```

Also confirm the runtime was built or installed for ARM64 / SM121 / GB10.

## OOM or whole-box lockup

This is unified memory. A large prefill working set can push the whole Spark into a bad state.

Try safer values:

```bash
--gpu-memory-utilization 0.70
--max-num-batched-tokens 8192
--max-num-seqs 8
```

If the box gets power-limited after OOM, do a full shutdown, unplug power briefly, then boot again.

## vLLM 0.27 crashes with CUTLASS FP4 MoE

If vLLM 0.27 starts but crashes on first request with something like:

```text
RuntimeError: run_fp4_blockwise_scaled_group_mm_sm120
Failed to initialize GEMM: status=7
workspace_size=43008 num_experts=256 M=65536 N=1024 K=2048
```

force Marlin for MoE:

```bash
--moe-backend marlin
```

This avoids the CUTLASS FP4 MoE path that failed on my GB10 setup.

## Non-fatal xgrammar FSM warnings

During one vLLM 0.27 benchmark, the server logged:

```text
backend_xgrammar.py ... Failed to advance FSM
```

The request still returned HTTP 200. If tool calls or structured output behave strangely, inspect the tool parser / structured-output path before blaming the model weights.

## Container starts but endpoint is not ready

Check logs:

```bash
journalctl -u qwen-vllm -f
```

Check readiness:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

## Vision inputs

The recipe enables:

```bash
--limit-mm-per-prompt '{"image":4}'
--allowed-media-domains '*'
```

Your client must still send OpenAI-style multimodal messages.
