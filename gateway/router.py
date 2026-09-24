"""Router: send the request to a provider, retrying and falling back when things fail.

Three separate ideas (a classic interview question):
  * RETRY      same provider, try again after a short wait. For blips (timeout, 429, 5xx).
  * FALLBACK   give up on this provider, try the next one in PROVIDER_ORDER.
  * BREAKER    remember that a provider is down, and skip it instantly for a while.

Backoff uses "full jitter": wait = random(0, min(cap, base * 2**attempt)).
Without the randomness, 1,000 clients that failed together would all retry at the same
instant and knock the provider over again (the "thundering herd").
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

from .breaker import CircuitBreaker, State
from .config import Settings

log = logging.getLogger("gateway.router")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# The client sent a bad request: every provider would reject it too, so return it as-is.
CLIENT_ERROR_STATUS = {400, 413, 422}


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    breaker: CircuitBreaker | None = None

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class RouteResult:
    status: int
    body: dict
    provider: str | None
    attempts: int = 0
    upstream_ms: float = 0.0
    tried: list[str] = field(default_factory=list)
    retry_after_s: float = 0.0


def build_providers(s: Settings) -> list[Provider]:
    catalogue = {
        "gemini": (s.GEMINI_BASE_URL, s.GEMINI_MODEL, s.GEMINI_API_KEY),
        "ollama": (s.OLLAMA_BASE_URL, s.OLLAMA_MODEL, ""),
        "mock": (s.MOCK_BASE_URL, s.MOCK_MODEL, ""),
        "mock2": (s.MOCK2_BASE_URL, s.MOCK2_MODEL, ""),
    }
    providers = []
    for name in s.provider_order:
        if name not in catalogue:
            log.warning("unknown provider %r in PROVIDER_ORDER, ignoring", name)
            continue
        base, model, key = catalogue[name]
        if name == "gemini" and not key:
            log.warning("GEMINI_API_KEY is empty, skipping gemini")
            continue
        breaker = CircuitBreaker(s.BREAKER_FAILS, s.BREAKER_COOLDOWN_S) if s.BREAKER_ENABLED else None
        providers.append(Provider(name, base, model, key, breaker))
    return providers


def backoff_s(attempt: int, base: float, cap: float) -> float:
    """Full jitter. attempt 0 -> up to base, attempt 1 -> up to 2*base, ... never above cap."""
    return random.uniform(0, min(cap, base * (2**attempt)))


class Router:
    def __init__(
        self,
        client: httpx.AsyncClient,
        providers: list[Provider],
        retry_max: int,
        retry_base_s: float,
        retry_cap_s: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.client = client
        self.providers = providers
        self.retry_max = max(1, retry_max)
        self.retry_base_s = retry_base_s
        self.retry_cap_s = retry_cap_s
        self._sleep = sleep  # injectable so tests don't really wait

    async def complete(self, body: dict, only: str | None = None) -> RouteResult:
        providers = [p for p in self.providers if only is None or p.name == only]
        if not providers:
            return RouteResult(400, _err(f"provider {only!r} is not configured", "invalid_request_error"), None)

        attempts = 0
        upstream_ms = 0.0
        tried: list[str] = []
        last_error = "no provider available"
        skipped_open = 0
        min_retry_after = None

        for p in providers:
            payload = dict(body, model=p.model)  # each provider gets its own model name
            payload.pop("stream", None)
            for attempt in range(self.retry_max):
                if p.breaker is not None and not p.breaker.allow():
                    if attempt == 0:
                        skipped_open += 1
                        ra = p.breaker.retry_after_s()
                        min_retry_after = ra if min_retry_after is None else min(min_retry_after, ra)
                        log.info("breaker open for %s, skipping", p.name)
                    break  # go to the next provider
                attempts += 1
                if attempt == 0:
                    tried.append(p.name)
                t0 = time.perf_counter()
                outcome = await self._call_once(p, payload)
                upstream_ms += (time.perf_counter() - t0) * 1000
                kind, status, data, retry_hint, error = outcome
                if kind == "ok":
                    return RouteResult(200, data, p.name, attempts, upstream_ms, tried)
                if kind == "client_error":
                    return RouteResult(status, data, p.name, attempts, upstream_ms, tried)
                last_error = f"{p.name}: {error}"
                if kind != "retryable":
                    break  # e.g. 401/403/404: retrying won't help, fall back instead
                if p.breaker is not None and p.breaker.state == State.OPEN:
                    break  # the breaker just tripped: don't wait, move on
                if attempt < self.retry_max - 1:
                    wait = backoff_s(attempt, self.retry_base_s, self.retry_cap_s)
                    if retry_hint is not None:  # provider told us how long to wait
                        wait = min(max(wait, retry_hint), self.retry_cap_s)
                    await self._sleep(wait)
            log.warning("provider %s failed (%s), falling back", p.name, last_error)

        if skipped_open == len(providers):
            return RouteResult(
                503, _err("all providers are temporarily unavailable (circuit open)", "upstream_unavailable"),
                None, attempts, upstream_ms, tried, retry_after_s=min_retry_after or 1.0,
            )
        return RouteResult(502, _err(f"all providers failed; last error: {last_error}", "upstream_error"),
                           None, attempts, upstream_ms, tried)


    async def _call_once(self, p: Provider, payload: dict):
        """One HTTP call. Returns (kind, status, data, retry_hint, error) where kind is
        "ok", "client_error", "retryable" or "fatal". The breaker ALWAYS learns the outcome
        (the finally block), even if the call is cancelled or raises something unexpected,
        so a half-open breaker can never get stuck waiting for a result."""
        headers = {"Authorization": f"Bearer {p.api_key}"} if p.api_key else {}
        healthy: bool | None = None
        try:
            try:
                resp = await self.client.post(p.url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                healthy = False
                return "retryable", 0, None, None, type(exc).__name__
            except httpx.HTTPError as exc:  # e.g. a corrupt response body
                healthy = False
                return "fatal", 0, None, None, type(exc).__name__
            if resp.status_code < 300:
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if not isinstance(data, dict) or "choices" not in data:
                    healthy = False
                    return "fatal", resp.status_code, None, None, "invalid response body"
                healthy = True
                return "ok", 200, data, None, None
            if resp.status_code in CLIENT_ERROR_STATUS:
                healthy = True  # provider is fine; the request itself is bad
                return "client_error", resp.status_code, _safe_json(resp), None, None
            healthy = False
            kind = "retryable" if resp.status_code in RETRYABLE_STATUS else "fatal"
            return kind, resp.status_code, None, _retry_after_header(resp), f"HTTP {resp.status_code}"
        finally:
            if healthy:
                _ok(p)
            else:  # False, or None when cancelled / an unexpected exception escaped
                _fail(p)


def _ok(p: Provider) -> None:
    if p.breaker is not None:
        p.breaker.record_success()


def _fail(p: Provider) -> None:
    if p.breaker is not None:
        p.breaker.record_failure()


def _err(message: str, kind: str) -> dict:
    return {"error": {"message": message, "type": kind}}


def _safe_json(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except ValueError:
        return _err(resp.text[:500], "upstream_error")


def _retry_after_header(resp: httpx.Response) -> float | None:
    val = resp.headers.get("retry-after")
    try:
        return float(val) if val is not None else None
    except ValueError:
        return None
