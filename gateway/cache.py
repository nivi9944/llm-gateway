"""Exact-match response cache in Redis.

Key = SHA-256 of (model + normalised messages + temperature). Value = the full JSON
response, stored with a TTL so entries expire.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from redis.exceptions import RedisError

log = logging.getLogger("gateway.cache")

_WS = re.compile(r"\s+")

# Fields that never change the answer, so they are left out of the key.
IGNORED_FIELDS = {"stream", "stream_options", "user"}


def normalise_text(text: Any) -> str:
    """Trim leading and trailing whitespace only, so "Hi there " and "Hi there" share a key.

    Inner whitespace is kept because newlines and indentation change the meaning of code.
    Case is kept as well. Paraphrases and small wording changes are left to the semantic cache.
    """
    if not isinstance(text, str):
        # OpenAI "content parts" (lists) are serialised deterministically.
        return json.dumps(text, sort_keys=True, separators=(",", ":"))
    return text.strip()


def collapse_text(text: str) -> str:
    """Looser form used by the semantic cache: collapse all whitespace and lowercase."""
    return _WS.sub(" ", text).strip().lower()


def normalise_messages(messages: list[dict]) -> list[dict]:
    """Keep every field of every message (tool_calls, name, ...) and tidy only role and content."""
    out = []
    for m in messages:
        m = dict(m)
        m["role"] = str(m.get("role", "")).strip().lower()
        m["content"] = normalise_text(m.get("content", ""))
        out.append(m)
    return out


def request_params(body: dict) -> dict:
    """Return every request field that can change the answer, apart from the messages.

    This covers model, temperature, max_tokens, tools, response_format, stop, seed, top_p and
    any newer OpenAI fields, since only IGNORED_FIELDS are dropped.
    """
    params = {k: v for k, v in body.items() if k not in IGNORED_FIELDS and k != "messages"}
    params["temperature"] = float(body.get("temperature", 1.0))
    return params


def exact_key(body: dict) -> str:
    payload = {"params": request_params(body), "messages": normalise_messages(body.get("messages", []))}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "exact:" + hashlib.sha256(raw.encode()).hexdigest()


class ExactCache:
    def __init__(self, redis, ttl_s: int):
        self.redis = redis
        self.ttl_s = ttl_s

    async def get(self, key: str) -> dict | None:
        if self.redis is None:
            return None
        try:
            raw = await self.redis.get(key)
        except (RedisError, OSError) as exc:
            log.warning("exact cache get failed, treating as miss: %s", exc)
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, response: dict) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.set(key, json.dumps(response), ex=self.ttl_s)
        except (RedisError, OSError) as exc:
            log.warning("exact cache set failed, skipping: %s", exc)
