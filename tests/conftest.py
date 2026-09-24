"""Shared test helpers.

Tests never touch the network: providers are faked with httpx.MockTransport, Redis is
fakeredis (or a real Redis if TEST_REDIS_URL is set, which CI does), and embeddings use
the tiny HashEmbedder. So the whole suite runs in seconds, offline, for free.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Callable

import fakeredis
import httpx
import pytest
from asgi_lifespan import LifespanManager

from gateway.config import Settings
from gateway.main import create_app

PRIMARY = "primary.test"
BACKUP = "backup.test"
KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def make_settings(**overrides) -> Settings:
    base = dict(
        GATEWAY_KEYS=f"{KEY},other-key",
        PROVIDER_ORDER="mock,mock2",
        MOCK_BASE_URL=f"http://{PRIMARY}/v1",
        MOCK2_BASE_URL=f"http://{BACKUP}/v1",
        EMBEDDER="hash",
        SEMANTIC_THRESHOLD=0.80,
        SEMANTIC_VERIFY_ENABLED=False,  # one-stage by default; verifier tests pass a fake verifier
        BUCKET_CAPACITY=1000,
        REFILL_PER_SEC=1000,
        RETRY_MAX=3,
        RETRY_BASE_S=0.001,
        RETRY_CAP_S=0.01,
        BREAKER_FAILS=5,
        BREAKER_COOLDOWN_S=30,
        CACHE_TTL_SECONDS=3600,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def ok_body(model: str = "mock", text: str = "hello") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


class FakeUpstream:
    """Fake LLM providers. Set a behaviour per host:
    an int status, a list of statuses (one per call, last one repeats), or an exception class.
    """

    def __init__(self):
        self.behaviour: dict[str, object] = {PRIMARY: 200, BACKUP: 200}
        self.calls: dict[str, int] = {PRIMARY: 0, BACKUP: 0}
        self.headers: dict[str, dict] = {}
        self.bodies: list[dict] = []

    def set(self, host: str, behaviour) -> None:
        self.behaviour[host] = behaviour

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        n = self.calls.get(host, 0)
        self.calls[host] = n + 1
        self.bodies.append(json.loads(request.content))
        b = self.behaviour.get(host, 404)
        if isinstance(b, list):
            b = b[min(n, len(b) - 1)]
        if isinstance(b, type) and issubclass(b, Exception):
            raise b("simulated", request=request)
        status = int(b)
        if status == 200:
            return httpx.Response(200, json=ok_body(host))
        return httpx.Response(status, json={"error": {"message": f"fake {status}"}},
                              headers=self.headers.get(host, {}))


async def fresh_redis():
    """An empty Redis: fakeredis by default, or a real one (flushed) if TEST_REDIS_URL is set."""
    url = os.getenv("TEST_REDIS_URL")
    if url:
        import redis.asyncio as aioredis

        r = aioredis.from_url(url)
        await r.flushdb()
        return r
    return fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())


@asynccontextmanager
async def gateway(settings: Settings | None = None, upstream: FakeUpstream | None = None,
                  redis_client=None, sleep: Callable | None = None, embedder=None, verifier=None):
    settings = settings or make_settings()
    upstream = upstream or FakeUpstream()
    r = redis_client if redis_client is not None else await fresh_redis()
    app = create_app(settings, redis_client=r, transport=httpx.MockTransport(upstream.handler),
                     sleep=sleep, embedder=embedder, verifier=verifier)
    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app),
                                     base_url="http://gateway") as client:
            client.upstream = upstream  # handy for assertions
            client.app = app
            client.redis = r
            yield client


def chat(content: str = "What is the capital of France?", temperature=0, system: str | None = None,
         model: str = "auto") -> dict:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    body = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    return body


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()
