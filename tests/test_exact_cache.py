import asyncio

from gateway.cache import exact_key

from .conftest import AUTH, chat, gateway, make_settings


async def post(gw, body, headers=None):
    return await gw.post("/v1/chat/completions", json=body, headers={**AUTH, **(headers or {})})


async def test_miss_then_hit():
    async with gateway() as gw:
        r1 = await post(gw, chat())
        r2 = await post(gw, chat())
        assert r1.headers["X-Cache"] == "MISS"
        assert r2.headers["X-Cache"] == "HIT-EXACT"
        assert r2.headers["X-Provider"] == "cache"
        assert r1.json() == r2.json()
        assert gw.upstream.calls["primary.test"] == 1  # second answer cost nothing


async def test_leading_and_trailing_whitespace_still_hit():
    async with gateway() as gw:
        await post(gw, chat("  What is the capital of France? \n"))
        r = await post(gw, chat("What is the capital of France?"))
        assert r.headers["X-Cache"] == "HIT-EXACT"


def test_inner_whitespace_matters_for_the_exact_key():
    # Indentation changes the meaning of code, so these must NOT share a key.
    a = chat("fix this:\nif x:\n    y()\nz()")
    b = chat("fix this:\nif x:\n    y()\n    z()")
    assert exact_key(a) != exact_key(b)


def test_parameters_that_change_the_answer_are_in_the_key():
    base = chat()
    for field, value in [("max_tokens", 5), ("tools", [{"type": "function"}]), ("stop", ["\n"]),
                         ("response_format", {"type": "json_object"}), ("seed", 1), ("top_p", 0.1)]:
        assert exact_key({**base, field: value}) != exact_key(base), field
    assert exact_key({**base, "user": "abc", "stream": False}) == exact_key(base)  # these don't matter
    with_tool_msg = {**base, "messages": base["messages"] + [{"role": "tool", "tool_call_id": "1", "content": "x"}]}
    other_tool_msg = {**base, "messages": base["messages"] + [{"role": "tool", "tool_call_id": "2", "content": "x"}]}
    assert exact_key(with_tool_msg) != exact_key(other_tool_msg)


async def test_entries_get_a_ttl_and_expire():
    async with gateway(make_settings(CACHE_TTL_SECONDS=1, SEMANTIC_ENABLED=False)) as gw:
        await post(gw, chat())
        ttl = await gw.redis.ttl(exact_key(chat()))
        assert 0 < ttl <= 1
        await asyncio.sleep(1.3)
        r = await post(gw, chat())
        assert r.headers["X-Cache"] == "MISS"
        assert gw.upstream.calls["primary.test"] == 2


async def test_no_caching_when_temperature_above_zero():
    async with gateway() as gw:
        await post(gw, chat(temperature=0.7))
        r = await post(gw, chat(temperature=0.7))
        assert r.headers["X-Cache"] == "MISS"
        assert gw.upstream.calls["primary.test"] == 2


async def test_no_caching_when_temperature_missing():
    # OpenAI's default temperature is 1, so a request without it is not cached.
    async with gateway() as gw:
        await post(gw, chat(temperature=None))
        r = await post(gw, chat(temperature=None))
        assert r.headers["X-Cache"] == "MISS"


async def test_bypass_header_skips_cache():
    async with gateway() as gw:
        await post(gw, chat())
        r = await post(gw, chat(), {"X-Cache-Bypass": "true"})
        assert r.headers["X-Cache"] == "MISS"


async def test_different_model_is_a_different_entry():
    async with gateway(make_settings(SEMANTIC_ENABLED=False)) as gw:
        await post(gw, chat(model="a"))
        r = await post(gw, chat(model="b"))
        assert r.headers["X-Cache"] == "MISS"


async def test_failed_calls_are_not_cached(upstream):
    upstream.set("primary.test", 500)
    upstream.set("backup.test", 500)
    async with gateway(upstream=upstream) as gw:
        r1 = await post(gw, chat())
        assert r1.status_code == 502
        upstream.set("primary.test", 200)
        r2 = await post(gw, chat())
        assert r2.status_code == 200 and r2.headers["X-Cache"] == "MISS"
