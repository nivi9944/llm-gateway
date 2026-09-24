import redis.asyncio as aioredis

from .conftest import AUTH, chat, gateway, make_settings


def dead_redis():
    # Nothing listens on port 1, so every Redis call fails fast.
    return aioredis.from_url("redis://127.0.0.1:1/0", socket_timeout=0.2, socket_connect_timeout=0.2)


async def test_gateway_still_answers_when_redis_is_down():
    async with gateway(make_settings(BUCKET_CAPACITY=1), redis_client=dead_redis()) as gw:
        # capacity 1, but the limiter fails open, so both requests pass
        r1 = await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        r2 = await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r2.headers["X-Cache"] == "MISS"  # no cache available, still served
        assert gw.upstream.calls["primary.test"] == 2
        health = (await gw.get("/health")).json()
        assert health["redis"].startswith("down")


async def test_metrics_summary_counts_hits_and_savings():
    async with gateway() as gw:
        for _ in range(3):
            await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        m = (await gw.get("/metrics-summary", headers=AUTH)).json()
        assert m["requests"] == 3
        assert m["by_cache"] == {"MISS": 1, "HIT-EXACT": 2}
        assert abs(m["est_saved_pct"] - 66.67) < 0.01
