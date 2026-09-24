"""All settings in one place. Values come from environment variables or a .env file.

Plain-language: every knob of the gateway (keys, URLs, limits, thresholds) lives here,
so nothing is hard-coded in the rest of the code.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Auth ---
    GATEWAY_KEYS: str = "dev-key-1,dev-key-2"  # comma-separated list of client API keys

    # --- Redis ---
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TIMEOUT_S: float = 0.5  # keep small: if Redis is slow/down we fail open quickly

    # --- Providers ---
    PROVIDER_ORDER: str = "gemini,ollama"  # tried left to right; benchmarks use "mock"
    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_MODEL: str = "gemini-3.8-flash"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434/v1"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    MOCK_BASE_URL: str = "http://mock-provider:9000/v1"
    MOCK_MODEL: str = "mock-1"
    MOCK2_BASE_URL: str = "http://mock-provider-2:9000/v1"  # second mock, used as a fallback in chaos tests
    MOCK2_MODEL: str = "mock-2"
    REQUEST_TIMEOUT_S: float = 30.0

    # --- Caching ---
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 86400
    SEMANTIC_ENABLED: bool = True
    SEMANTIC_THRESHOLD: float = 0.90  # replace with the value from results/qqp_threshold.json
    EMBEDDER: str = "minilm"  # "minilm" (real) or "hash" (tiny stand-in for tests / offline trials)
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Two-stage semantic cache. When ON, FAISS only proposes a candidate at
    # SEMANTIC_CANDIDATE_THRESHOLD, and a cross-encoder must score the pair >= SEMANTIC_VERIFY_THRESHOLD.
    # When OFF, SEMANTIC_THRESHOLD alone decides (the one-stage cache).
    SEMANTIC_VERIFY_ENABLED: bool = True
    SEMANTIC_CANDIDATE_THRESHOLD: float = 0.85
    SEMANTIC_VERIFY_THRESHOLD: float = 0.5  # replace with the value from qqp_threshold.py --two-stage
    VERIFY_MODEL: str = "cross-encoder/quora-distilroberta-base"

    # --- Rate limiting (token bucket per key) ---
    RATE_LIMIT_ENABLED: bool = True
    BUCKET_CAPACITY: float = Field(10, gt=0)
    REFILL_PER_SEC: float = Field(0.1667, gt=0)  # ~10 requests per minute sustained; must be > 0

    # --- Reliability ---
    RETRY_MAX: int = 3  # total tries per provider (1 = no retry)
    RETRY_BASE_S: float = 0.25
    RETRY_CAP_S: float = 4.0
    BREAKER_ENABLED: bool = True
    BREAKER_FAILS: int = 5
    BREAKER_COOLDOWN_S: float = 30.0

    # --- Cost estimate (USD per 1M tokens), Gemini paid-tier list price ---
    # Source: ai.google.dev/gemini-api/docs/pricing (checked 2026-09-24, gemini-3.8-flash,
    # price valid through 2026-12-31). These are ESTIMATES: free-tier calls actually cost 0.
    PRICE_INPUT_PER_M: float = 0.75
    PRICE_OUTPUT_PER_M: float = 3.75

    # --- Logging / debug ---
    LOG_FILE: str = ""  # optional path; one JSON line per request is appended

    @property
    def keys(self) -> set[str]:
        return {k.strip() for k in self.GATEWAY_KEYS.split(",") if k.strip()}

    @property
    def provider_order(self) -> list[str]:
        return [p.strip().lower() for p in self.PROVIDER_ORDER.split(",") if p.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
