import pytest

from gateway.semantic_cache import HashEmbedder, SemanticCache

from .conftest import AUTH, chat, fresh_redis, gateway, make_settings


async def post(gw, body):
    return await gw.post("/v1/chat/completions", json=body, headers=AUTH)


async def test_paraphrase_hits_above_threshold():
    async with gateway() as gw:
        await post(gw, chat("How do I learn Python quickly?"))
        r = await post(gw, chat("How can I learn Python quickly?"))
        assert r.headers["X-Cache"] == "HIT-SEMANTIC"
        assert float(r.headers["X-Similarity"]) >= 0.80
        assert gw.upstream.calls["primary.test"] == 1


async def test_unrelated_question_misses():
    async with gateway() as gw:
        await post(gw, chat("How do I learn Python quickly?"))
        r = await post(gw, chat("What is the boiling point of water at sea level?"))
        assert r.headers["X-Cache"] == "MISS"
        assert gw.upstream.calls["primary.test"] == 2


async def test_different_system_prompt_never_matches():
    async with gateway() as gw:
        await post(gw, chat("How do I learn Python quickly?", system="Answer like a pirate"))
        r = await post(gw, chat("How can I learn Python quickly?", system="Answer formally"))
        assert r.headers["X-Cache"] == "MISS"


async def test_semantic_hit_is_promoted_to_exact_cache():
    async with gateway() as gw:
        await post(gw, chat("How do I learn Python quickly?"))
        await post(gw, chat("How can I learn Python quickly?"))
        r = await post(gw, chat("How can I learn Python quickly?"))
        assert r.headers["X-Cache"] == "HIT-EXACT"


async def test_embed_and_search_run_on_the_cache_own_thread_pool():
    import threading

    class SpyEmbedder(HashEmbedder):
        def encode(self, texts):
            seen.append(threading.current_thread().name)
            return super().encode(texts)

    seen: list[str] = []
    cache = SemanticCache(await fresh_redis(), SpyEmbedder(), 0.9, 60)
    q = await cache.embed(chat("How do I learn Python quickly?"))
    await cache.store(q, {"id": "x"})
    assert (await cache.lookup(q)).response == {"id": "x"}
    assert seen and all(name.startswith("semantic") for name in seen)
    cache.close()


async def test_warm_start_rebuilds_index_from_redis():
    r = await fresh_redis()
    async with gateway(redis_client=r) as gw:
        await post(gw, chat("How do I learn Python quickly?"))
    # "Restart" the gateway with the same Redis: FAISS is empty in memory, then rebuilt.
    async with gateway(redis_client=r) as gw2:
        assert (await gw2.get("/health")).json()["semantic_vectors"] == 1
        resp = await post(gw2, chat("How can I learn Python quickly?"))
        assert resp.headers["X-Cache"] == "HIT-SEMANTIC"


async def test_expired_answer_is_a_miss_and_vector_is_dropped():
    r = await fresh_redis()
    cache = SemanticCache(r, HashEmbedder(), threshold=0.8, ttl_s=3600)
    q = await cache.embed(chat("How do I learn Python quickly?"))
    await cache.store(q, {"answer": 1})
    await r.flushdb()  # simulate TTL expiry of the answer
    res = await cache.lookup(q)
    assert res.response is None
    assert cache.size() == 0


async def test_different_max_tokens_never_matches():
    async with gateway() as gw:
        await post(gw, chat("How do I learn Python quickly?"))
        r = await post(gw, {**chat("How can I learn Python quickly?"), "max_tokens": 5})
        assert r.headers["X-Cache"] == "MISS"


async def test_reused_id_after_redis_data_loss_is_not_trusted():
    """If Redis loses data, the id counter restarts and FAISS may point an old id at a NEW entry.
    The lookup must notice the entry doesn't belong to this question."""
    r = await fresh_redis()
    cache = SemanticCache(r, HashEmbedder(), threshold=0.8, ttl_s=3600)
    q_france = await cache.embed(chat("What is the capital of France?"))
    await cache.store(q_france, {"answer": "Paris"})
    await r.flushdb()  # Redis loses everything; FAISS still has id 1
    q_bank = await cache.embed(chat("What is my account balance?", system="You are a bank bot"))
    await cache.store(q_bank, {"answer": "secret balance"})  # gets id 1 again
    res = await cache.lookup(q_france)
    assert res.response is None


@pytest.mark.slow
async def test_real_minilm_paraphrase_vs_unrelated():
    """Runs with the real model (on your laptop / in CI). Skipped if it can't be downloaded."""
    try:
        from gateway.semantic_cache import MiniLMEmbedder

        emb = MiniLMEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:  # no internet / model not available
        pytest.skip(f"MiniLM unavailable: {exc}")
    v = emb.encode(["How do I learn Python quickly?", "What is the fastest way to learn Python?",
                    "What is the boiling point of water?"])
    para, unrelated = float(v[0] @ v[1]), float(v[0] @ v[2])
    assert para > 0.75
    assert unrelated < 0.3
    async with gateway(make_settings(EMBEDDER="minilm", SEMANTIC_THRESHOLD=0.75), embedder=emb) as gw:
        await post(gw, chat("How do I learn Python quickly?"))
        r = await post(gw, chat("What is the fastest way to learn Python?"))
        assert r.headers["X-Cache"] == "HIT-SEMANTIC"
