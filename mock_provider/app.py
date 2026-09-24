"""A fake LLM that speaks the OpenAI chat format. Used for tests and benchmarks.

Why: real LLMs cost money, are slow, and fail at random times you can't control.
The mock is free, repeatable, and lets us inject failures on purpose.

Env vars (can also be changed live with POST /admin/config):
  MOCK_LATENCY_MS         how long each answer "thinks" (default 800)
  MOCK_FAIL_RATE          fraction of requests that fail, 0.0-1.0 (default 0)
  MOCK_FAIL_STATUS        HTTP status used for failures (default 503)
  MOCK_COMPLETION_TOKENS  approximate answer length in tokens (default 150)
"""
from __future__ import annotations

import asyncio
import os
import random
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Mock LLM provider")

CONFIG = {
    "latency_ms": float(os.getenv("MOCK_LATENCY_MS", "800")),
    "fail_rate": float(os.getenv("MOCK_FAIL_RATE", "0.0")),
    "fail_status": int(os.getenv("MOCK_FAIL_STATUS", "503")),
    "completion_tokens": int(os.getenv("MOCK_COMPLETION_TOKENS", "150")),
    "seed": os.getenv("MOCK_SEED"),
}
STATS = {"calls": 0, "failures": 0}
_rng = random.Random(CONFIG["seed"])

FILLER = ("this is a simulated answer from the mock provider used for repeatable tests "
          "and benchmarks of the gateway ").split()


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb: ~4 characters per token


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    STATS["calls"] += 1
    if CONFIG["latency_ms"] > 0:
        await asyncio.sleep(CONFIG["latency_ms"] / 1000)
    if _rng.random() < CONFIG["fail_rate"]:
        STATS["failures"] += 1
        return JSONResponse({"error": {"message": "injected failure", "type": "mock_error"}},
                            status_code=CONFIG["fail_status"])

    messages = body.get("messages", [])
    last = str(messages[-1].get("content", "")) if messages else ""
    words = [FILLER[i % len(FILLER)] for i in range(int(CONFIG["completion_tokens"] * 0.75))]
    answer = f"Mock answer to: {last[:200]} " + " ".join(words)
    prompt_tokens = sum(_tokens(str(m.get("content", ""))) for m in messages)
    completion_tokens = _tokens(answer)
    return {
        "id": "chatcmpl-mock-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock-1"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


@app.get("/health")
async def health():
    return {"status": "ok", **CONFIG}


@app.post("/admin/config")
async def set_config(request: Request):
    """Change latency / failure rate without restarting (used by the chaos benchmark)."""
    global _rng
    update = await request.json()
    for k in ("latency_ms", "fail_rate", "fail_status", "completion_tokens"):
        if k in update:
            CONFIG[k] = type(CONFIG[k])(update[k])
    if "seed" in update:
        CONFIG["seed"] = update["seed"]
        _rng = random.Random(update["seed"])
    return CONFIG


@app.get("/admin/stats")
async def stats():
    return STATS


@app.post("/admin/reset")
async def reset():
    STATS.update(calls=0, failures=0)
    return STATS
