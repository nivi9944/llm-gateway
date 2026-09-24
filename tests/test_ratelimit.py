import asyncio

from .conftest import AUTH, chat, gateway, make_settings


async def test_50_concurrent_requests_exactly_capacity_pass():
    s = make_settings(BUCKET_CAPACITY=10, REFILL_PER_SEC=0.0001, CACHE_ENABLED=False)
    async with gateway(s) as gw:
        results = await asyncio.gather(*[
            gw.post("/v1/chat/completions", json=chat(f"q{i}"), headers=AUTH) for i in range(50)
        ])
        codes = [r.status_code for r in results]
        assert codes.count(200) == 10  # atomic Lua script: no race lets an 11th through
        assert codes.count(429) == 40


async def test_429_has_retry_after_header():
    s = make_settings(BUCKET_CAPACITY=1, REFILL_PER_SEC=0.1, CACHE_ENABLED=False)
    async with gateway(s) as gw:
        await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        r = await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        assert r.status_code == 429
        assert r.json()["error"]["type"] == "rate_limit_error"
        # 1 token at 0.1 tokens/s -> about 10 s to wait
        assert 1 <= int(r.headers["Retry-After"]) <= 10


async def test_each_key_has_its_own_bucket():
    s = make_settings(BUCKET_CAPACITY=1, REFILL_PER_SEC=0.0001, CACHE_ENABLED=False)
    async with gateway(s) as gw:
        a1 = await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        a2 = await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        b1 = await gw.post("/v1/chat/completions", json=chat(), headers={"X-API-Key": "other-key"})
        assert (a1.status_code, a2.status_code, b1.status_code) == (200, 429, 200)


async def test_bucket_refills_over_time():
    s = make_settings(BUCKET_CAPACITY=1, REFILL_PER_SEC=10, CACHE_ENABLED=False)
    async with gateway(s) as gw:
        assert (await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)).status_code == 200
        assert (await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)).status_code == 429
        await asyncio.sleep(0.15)  # 10 tokens/s -> 1 token after 0.1 s
        assert (await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)).status_code == 200
