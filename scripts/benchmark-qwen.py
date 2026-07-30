#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


PROMPT = """Write a complete single-file Python implementation of a tiny HTTP router.
Requirements: route decorators, path params, query parsing, JSON responses,
error handling, and a usage example. Return code only."""


def post(base_url: str, api_key: str | None, model: str, max_tokens: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - start
    usage = data.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "elapsed": elapsed,
        "completion_tokens": completion,
        "tok_s": completion / elapsed if elapsed else 0.0,
    }


def run_concurrency(base_url: str, api_key: str | None, model: str, max_tokens: int, concurrency: int) -> dict:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: post(base_url, api_key, model, max_tokens), range(concurrency)))
    elapsed = time.perf_counter() - start
    total_completion = sum(r["completion_tokens"] for r in results)
    speeds = [r["tok_s"] for r in results]
    return {
        "concurrency": concurrency,
        "elapsed": elapsed,
        "aggregate_tok_s": total_completion / elapsed if elapsed else 0.0,
        "stream_tok_s_mean": statistics.mean(speeds) if speeds else 0.0,
        "raw": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen36-35b")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--concurrency", default="1,2,3,4,6,8")
    args = parser.parse_args()

    concurrencies = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    for c in concurrencies:
        result = run_concurrency(args.base_url, args.api_key, args.model, args.max_tokens, c)
        print(
            f"C{c}: aggregate={result['aggregate_tok_s']:.1f} tok/s "
            f"stream_mean={result['stream_tok_s_mean']:.1f} tok/s "
            f"elapsed={result['elapsed']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
