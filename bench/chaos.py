"""Benchmark 4: chaos test. Does the gateway keep answering when the provider misbehaves?

Two mock providers (50 ms latency each): "mock" is the primary, "mock2" the fallback.
Caching is OFF so every request really needs a provider. Real backoff waits are used.

Scenarios (2,000 requests each, 20 at a time):
  flaky_no_protection        primary fails 30%; 1 try, no fallback, no breaker
  flaky_protected            primary fails 30%; retries + fallback + breaker
  outage_no_breaker          primary fails 100%; retries + fallback, breaker OFF
  outage_protected           primary fails 100%; retries + fallback + breaker

The first pair gives the headline "success % with vs without". The second pair shows what
the breaker adds: during an outage it stops wasting retries (and time) on a dead provider.

Run:   python bench/chaos.py     (needs Redis at BENCH_REDIS_URL; no ML model needed)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._common import BENCH_KEY, BENCH_REDIS_URL, flush_bench_redis, pct, save_json, start_mock, stop  # noqa: E402

SCENARIOS = [
    {"name": "flaky_no_protection", "fail_rate": 0.3, "order": "mock", "retry_max": 1, "breaker": False},
    {"name": "flaky_protected", "fail_rate": 0.3, "order": "mock,mock2", "retry_max": 3, "breaker": True},
    {"name": "outage_no_breaker", "fail_rate": 1.0, "order": "mock,mock2", "retry_max": 3, "breaker": False},
    {"name": "outage_protected", "fail_rate": 1.0, "order": "mock,mock2", "retry_max": 3, "breaker": True},
]


async def run_scenario(sc: dict, args) -> dict:
    from asgi_lifespan import LifespanManager

    from gateway.config import Settings
    from gateway.main import create_app

    admin1 = f"http://127.0.0.1:{args.port1}"
    admin2 = f"http://127.0.0.1:{args.port2}"
    async with httpx.AsyncClient() as h:
        await h.post(f"{admin1}/admin/config", json={"fail_rate": sc["fail_rate"], "seed": args.seed})
        await h.post(f"{admin1}/admin/reset")
        await h.post(f"{admin2}/admin/reset")

    settings = Settings(
        _env_file=None, GATEWAY_KEYS=BENCH_KEY, REDIS_URL=BENCH_REDIS_URL, PROVIDER_ORDER=sc["order"],
        MOCK_BASE_URL=f"{admin1}/v1", MOCK2_BASE_URL=f"{admin2}/v1", CACHE_ENABLED=False,
        RATE_LIMIT_ENABLED=False, RETRY_MAX=sc["retry_max"], BREAKER_ENABLED=sc["breaker"],
    )
    app = create_app(settings)
    lat: list[float] = []
    codes: dict[int, int] = {}
    attempts: list[int] = []
    queue = list(range(args.n))
    t0 = time.perf_counter()
    async with LifespanManager(app, startup_timeout=120) as mgr:  # model load > default 5 s
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mgr.app), base_url="http://gw",
                                     timeout=120) as client:
            async def worker():
                while queue:
                    i = queue.pop()
                    body = {"model": "auto", "temperature": 0.7,
                            "messages": [{"role": "user", "content": f"chaos request {i}"}]}
                    s = time.perf_counter()
                    r = await client.post("/v1/chat/completions", json=body,
                                          headers={"Authorization": f"Bearer {BENCH_KEY}"})
                    lat.append((time.perf_counter() - s) * 1000)
                    codes[r.status_code] = codes.get(r.status_code, 0) + 1
                    attempts.append(int(r.headers.get("X-Attempts", 0)))

            await asyncio.gather(*[worker() for _ in range(args.concurrency)])
    wall = time.perf_counter() - t0
    async with httpx.AsyncClient() as h:
        s1 = (await h.get(f"{admin1}/admin/stats")).json()
        s2 = (await h.get(f"{admin2}/admin/stats")).json()
    ok = codes.get(200, 0)
    return {
        **sc,
        "requests": args.n,
        "success_pct": round(100 * ok / args.n, 2),
        "status_counts": {str(k): v for k, v in sorted(codes.items())},
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99)},
        "avg_provider_attempts": round(sum(attempts) / len(attempts), 3),
        "primary_calls": s1["calls"], "primary_failures": s1["failures"],
        "fallback_calls": s2["calls"],
        "wall_seconds": round(wall, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--latency-ms", type=float, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port1", type=int, default=9201)
    ap.add_argument("--port2", type=int, default=9202)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    logging.getLogger("gateway.requests").setLevel(logging.WARNING)
    logging.getLogger("gateway.router").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    flush_bench_redis()
    m1 = start_mock(args.port1, args.latency_ms, seed=args.seed, name="chaos-primary")
    m2 = start_mock(args.port2, args.latency_ms, seed=args.seed, name="chaos-fallback")
    results = []
    try:
        for sc in SCENARIOS:
            print(f"  {sc['name']} ...", flush=True)
            res = asyncio.run(run_scenario(sc, args))
            print(f"    success {res['success_pct']}%  p95 {res['latency_ms']['p95']} ms  "
                  f"primary calls {res['primary_calls']}")
            results.append(res)
    finally:
        stop(m1, m2)

    by = {r["name"]: r for r in results}
    save_json(args.out, "chaos.json", {
        "benchmark": "chaos",
        "setup": {"requests_per_scenario": args.n, "concurrency": args.concurrency,
                  "provider_latency_ms": args.latency_ms, "cache": "off", "seed": args.seed,
                  "retry_policy": "full-jitter backoff, base 0.25 s, cap 4 s",
                  "breaker": "opens after 5 consecutive failures, 30 s cooldown"},
        "headline": {
            "primary_failure_rate": 0.3,
            "success_pct_without_protection": by["flaky_no_protection"]["success_pct"],
            "success_pct_with_protection": by["flaky_protected"]["success_pct"],
        },
        "breaker_effect_during_outage": {
            "primary_calls_without_breaker": by["outage_no_breaker"]["primary_calls"],
            "primary_calls_with_breaker": by["outage_protected"]["primary_calls"],
            "p95_ms_without_breaker": by["outage_no_breaker"]["latency_ms"]["p95"],
            "p95_ms_with_breaker": by["outage_protected"]["latency_ms"]["p95"],
        },
        "scenarios": results,
    })


if __name__ == "__main__":
    main()
