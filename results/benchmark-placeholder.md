# Benchmark placeholder

Replace this file after running:

```bash
python3 scripts/benchmark-qwen.py \
  --base-url http://127.0.0.1:8888/v1 \
  --api-key "$VLLM_API_KEY" \
  --model qwen36-35b \
  --concurrency 1,2,3,4,6,8 \
  --max-tokens 500
```

## Reference target

MiaAI-Lab reported roughly:

```text
C1: 95.1 tok/s
C8: 317.0 tok/s aggregate
TTFT: ~103-242 ms across C1-C8
```

Run your own benchmark before publishing claims for your own machine.
