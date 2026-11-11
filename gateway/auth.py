"""API key authentication for incoming requests.

Accepts either `Authorization: Bearer <key>` (what OpenAI SDKs send) or `X-API-Key: <key>`.
"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import HTTPException, Request


def extract_key(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def check_key(key: str | None, valid_keys: set[str]) -> str:
    """Return the key if it is valid, otherwise raise a 401.

    hmac.compare_digest runs in constant time, so response timing does not leak
    how many leading characters of a guess were correct.
    """
    if key:
        for valid in valid_keys:
            if hmac.compare_digest(key.encode(), valid.encode()):
                return key
    raise HTTPException(
        status_code=401,
        detail={"message": "Missing or invalid API key", "type": "authentication_error"},
    )


def key_id(key: str) -> str:
    """Return a short, non-reversible id for a key, so raw keys never reach the logs."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]
