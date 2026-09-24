"""Observability: one JSON log line per request, plus running totals for /metrics-summary.

Cost is an ESTIMATE: tokens x the Gemini paid-tier list price in config. Free-tier calls
really cost 0, but pricing them at the paid rate gives a meaningful "money saved" figure.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, deque
from typing import Any

import numpy as np

log = logging.getLogger("gateway.requests")


def estimate_cost_usd(usage: dict | None, price_in_per_m: float, price_out_per_m: float) -> float:
    if not usage:
        return 0.0
    p = usage.get("prompt_tokens") or 0
    c = usage.get("completion_tokens") or 0
    return (p * price_in_per_m + c * price_out_per_m) / 1_000_000


class Metrics:
    def __init__(self, price_in_per_m: float, price_out_per_m: float, log_file: str = "", window: int = 5000):
        self.price_in = price_in_per_m
        self.price_out = price_out_per_m
        self.log_file = log_file
        self.started = time.time()
        self.total = 0
        self.by_cache: Counter[str] = Counter()
        self.by_provider: Counter[str] = Counter()
        self.by_status: Counter[int] = Counter()
        self.cost_spent = 0.0
        self.cost_saved = 0.0
        self.latencies: deque[float] = deque(maxlen=window)

    def cost(self, usage: dict | None) -> float:
        return estimate_cost_usd(usage, self.price_in, self.price_out)

    def record(self, entry: dict[str, Any]) -> None:
        """entry must hold: status, cache, provider, latency_ms, usage."""
        usage = entry.pop("usage", None)
        est = self.cost(usage)
        hit = entry.get("cache", "MISS").startswith("HIT")
        entry["prompt_tokens"] = (usage or {}).get("prompt_tokens")
        entry["completion_tokens"] = (usage or {}).get("completion_tokens")
        entry["est_cost_usd"] = 0.0 if hit else round(est, 8)
        entry["est_saved_usd"] = round(est, 8) if hit else 0.0

        self.total += 1
        self.by_cache[entry.get("cache", "MISS")] += 1
        self.by_provider[entry.get("provider") or "none"] += 1
        self.by_status[int(entry.get("status", 0))] += 1
        if entry.get("status") == 200:
            self.cost_spent += entry["est_cost_usd"]
            self.cost_saved += entry["est_saved_usd"]
        self.latencies.append(float(entry.get("latency_ms", 0.0)))

        line = json.dumps({"ts": round(time.time(), 3), **entry}, separators=(",", ":"))
        log.info(line)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def summary(self) -> dict:
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        hits = sum(v for k, v in self.by_cache.items() if k.startswith("HIT"))
        would_have_cost = self.cost_spent + self.cost_saved
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "requests": self.total,
            "by_cache": dict(self.by_cache),
            "by_provider": dict(self.by_provider),
            "by_status": {str(k): v for k, v in self.by_status.items()},
            "hit_rate": round(hits / self.total, 4) if self.total else 0.0,
            "est_cost_usd": round(self.cost_spent, 6),
            "est_saved_usd": round(self.cost_saved, 6),
            "est_saved_pct": round(100 * self.cost_saved / would_have_cost, 2) if would_have_cost else 0.0,
            "latency_ms": {
                "p50": round(float(np.percentile(lat, 50)), 2),
                "p95": round(float(np.percentile(lat, 95)), 2),
                "p99": round(float(np.percentile(lat, 99)), 2),
                "window": len(self.latencies),
            },
            "cost_note": "estimated at Gemini paid-tier list price; free-tier calls actually cost 0",
        }
