"""The FastAPI app: wires every piece together in the order of the request flow.

    auth -> rate limit -> exact cache -> semantic cache -> provider (retry/fallback/breaker)
         -> store in caches -> log -> reply

Cheapest checks run first, so a rejected or cached request never costs a provider call.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from .auth import check_key, extract_key, key_id
from .cache import ExactCache, exact_key
from .config import Settings, get_settings
from .metrics import Metrics
from .ratelimit import RateLimiter
from .router import Router, build_providers
from .semantic_cache import CrossEncoderVerifier, Embedder, SemanticCache, Verifier, build_embedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gateway")


@dataclass
class Components:
    settings: Settings
    redis: object
    http: httpx.AsyncClient
    limiter: RateLimiter
    exact: ExactCache
    semantic: SemanticCache | None
    router: Router
    metrics: Metrics


def create_app(
    settings: Settings | None = None,
    *,
    redis_client=None,
    embedder: Embedder | None = None,
    verifier: Verifier | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep=None,
) -> FastAPI:
    """Build the app. Tests pass fakes (fake Redis, fake embedder, fake HTTP transport)."""
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        r = redis_client
        if r is None:
            r = aioredis.from_url(
                s.REDIS_URL, socket_timeout=s.REDIS_TIMEOUT_S, socket_connect_timeout=s.REDIS_TIMEOUT_S
            )
        # ONE shared HTTP client: reuses TCP/TLS connections instead of opening one per call.
        http = httpx.AsyncClient(
            timeout=httpx.Timeout(s.REQUEST_TIMEOUT_S, connect=5.0),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
            transport=transport,
        )
        semantic = None
        if s.CACHE_ENABLED and s.SEMANTIC_ENABLED:
            emb = embedder or build_embedder(s.EMBEDDER, s.EMBED_MODEL)  # loaded ONCE at startup
            if s.SEMANTIC_VERIFY_ENABLED:  # two stages: FAISS candidate, then cross-encoder check
                ver = verifier or CrossEncoderVerifier(s.VERIFY_MODEL)  # also loaded once
                semantic = SemanticCache(r, emb, s.SEMANTIC_CANDIDATE_THRESHOLD, s.CACHE_TTL_SECONDS,
                                         verifier=ver, verify_threshold=s.SEMANTIC_VERIFY_THRESHOLD)
            else:
                semantic = SemanticCache(r, emb, s.SEMANTIC_THRESHOLD, s.CACHE_TTL_SECONDS)
            n = await semantic.warm_start()
            log.info("semantic cache ready (%d vectors restored)", n)
        router_kwargs = {"sleep": sleep} if sleep else {}
        app.state.c = Components(
            settings=s,
            redis=r,
            http=http,
            limiter=RateLimiter(r, s.BUCKET_CAPACITY, s.REFILL_PER_SEC),
            exact=ExactCache(r, s.CACHE_TTL_SECONDS),
            semantic=semantic,
            router=Router(http, build_providers(s), s.RETRY_MAX, s.RETRY_BASE_S, s.RETRY_CAP_S, **router_kwargs),
            metrics=Metrics(s.PRICE_INPUT_PER_M, s.PRICE_OUTPUT_PER_M, s.LOG_FILE),
        )
        log.info("providers: %s", [p.name for p in app.state.c.router.providers])
        yield
        if semantic is not None:
            semantic.close()
        await http.aclose()
        if redis_client is None:
            await r.aclose()

    app = FastAPI(title="LLM API Gateway", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        return JSONResponse({"error": detail}, status_code=exc.status_code, headers=exc.headers)

    @app.get("/health")
    async def health(request: Request):
        c: Components = request.app.state.c
        try:
            redis_ok = bool(await c.redis.ping())
        except (RedisError, OSError):
            redis_ok = False
        return {
            "status": "ok",
            "redis": "up" if redis_ok else "down (failing open: no caching or rate limiting)",
            "providers": {
                p.name: (p.breaker.state.value if p.breaker else "no-breaker") for p in c.router.providers
            },
            "semantic_vectors": c.semantic.size() if c.semantic else None,
        }

    @app.get("/metrics-summary")
    async def metrics_summary(request: Request):
        c: Components = request.app.state.c
        check_key(extract_key(request), c.settings.keys)
        return c.metrics.summary()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        c: Components = request.app.state.c
        t0 = time.perf_counter()
        rid = uuid.uuid4().hex[:12]

        def finish(status: int, body: dict, cache: str, provider: str | None, usage=None,
                   similarity=None, extra_headers=None, **log_fields) -> JSONResponse:
            latency_ms = (time.perf_counter() - t0) * 1000
            headers = {
                "X-Request-Id": rid,
                "X-Cache": cache,
                "X-Provider": provider or "none",
                "X-Latency-Ms": f"{latency_ms:.2f}",
            }
            if similarity is not None:
                headers["X-Similarity"] = f"{similarity:.4f}"
            headers.update(extra_headers or {})
            c.metrics.record({
                "request_id": rid, "status": status, "cache": cache, "provider": provider,
                "similarity": None if similarity is None else round(similarity, 4),
                "latency_ms": round(latency_ms, 2), "usage": usage, **log_fields,
            })
            return JSONResponse(body, status_code=status, headers=headers)

        # 1. Auth
        key = check_key(extract_key(request), c.settings.keys)
        kid = key_id(key)

        # 2. Validate the body (OpenAI chat format)
        try:
            body = await request.json()
        except ValueError:
            return finish(400, {"error": {"message": "body must be JSON", "type": "invalid_request_error"}},
                          "NONE", None, key_id=kid)
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages):
            return finish(400, {"error": {"message": "'messages' must be a non-empty list of {role, content} objects",
                                          "type": "invalid_request_error"}}, "NONE", None, key_id=kid)
        if body.get("stream"):
            return finish(400, {"error": {"message": "streaming is not supported by this gateway",
                                          "type": "invalid_request_error"}}, "NONE", None, key_id=kid)

        # 3. Rate limit
        if c.settings.RATE_LIMIT_ENABLED:
            decision = await c.limiter.check(kid)
            if not decision.allowed:
                retry = max(1, math.ceil(decision.retry_after_s))
                return finish(429, {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
                              "NONE", None, extra_headers={"Retry-After": str(retry)}, key_id=kid)

        # Only deterministic (temperature 0) requests are cacheable. OpenAI's default is 1.
        bypass = request.headers.get("x-cache-bypass", "").lower() in {"1", "true", "yes"}
        try:
            temperature = float(body.get("temperature", 1.0))
        except (TypeError, ValueError):
            temperature = 1.0
        cacheable = c.settings.CACHE_ENABLED and temperature == 0 and not bypass
        forced = request.headers.get("x-provider-force")  # e.g. "ollama" for demos

        # 4. Exact cache
        ekey = exact_key(body) if cacheable else None
        if ekey:
            hit = await c.exact.get(ekey)
            if hit is not None:
                return finish(200, hit, "HIT-EXACT", "cache", usage=hit.get("usage"), key_id=kid)

        # 5. Semantic cache
        sq = None
        similarity = None
        verify = {}  # X-Verify-Score header, when the stage-2 verifier ran
        if cacheable and c.semantic is not None:
            sq = await c.semantic.embed(body)
            if sq is not None:
                res = await c.semantic.lookup(sq)
                similarity = res.similarity
                if res.verify_score is not None:
                    verify = {"X-Verify-Score": f"{res.verify_score:.4f}"}
                if res.response is not None:
                    # Also store under the exact key, so the next identical request is even cheaper.
                    if ekey:
                        await c.exact.set(ekey, res.response)
                    return finish(200, res.response, "HIT-SEMANTIC", "cache", usage=res.response.get("usage"),
                                  similarity=similarity, extra_headers=verify, key_id=kid,
                                  verify_score=res.verify_score)

        # 6. Provider call (retries, fallback, circuit breaker)
        result = await c.router.complete(body, only=forced)
        extra = {"X-Upstream-Ms": f"{result.upstream_ms:.2f}", "X-Attempts": str(result.attempts), **verify}
        if result.status == 503 and result.retry_after_s:
            extra["Retry-After"] = str(max(1, math.ceil(result.retry_after_s)))

        # 7. Store and reply
        if result.status == 200 and cacheable:
            await c.exact.set(ekey, result.body)
            if sq is not None:
                await c.semantic.store(sq, result.body)
        return finish(result.status, result.body, "MISS", result.provider,
                      usage=result.body.get("usage") if result.status == 200 else None,
                      similarity=similarity, extra_headers=extra, key_id=kid,
                      attempts=result.attempts, tried=result.tried, upstream_ms=round(result.upstream_ms, 2))

    return app


app = create_app()
