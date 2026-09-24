"""API-key check: the "show your ID at the front desk" step.

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
    """Return the key if valid, else raise 401.

    hmac.compare_digest compares in constant time, so an attacker can't guess a key
    character by character from how long the comparison takes.
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
    """Short, non-reversible id for logs, so raw keys never appear in log files."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]
