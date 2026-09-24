"""Two-stage semantic cache: FAISS proposes a candidate, a (fake) cross-encoder decides."""
from .conftest import AUTH, chat, gateway, make_settings

FIRST = "How do I learn Python quickly?"
PARAPHRASE = "How can I learn Python quickly?"  # hash-embedder similarity is above 0.80, below 0.999


class FakeVerifier:
    """Returns a fixed score and remembers every pair it was asked about."""

    def __init__(self, score: float = 0.9, error: Exception | None = None):
        self.fixed = score
        self.error = error
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs):
        self.pairs.extend(pairs)
        if self.error:
            raise self.error
        return [self.fixed for _ in pairs]


def two_stage(**overrides):
    # SEMANTIC_THRESHOLD is set so high that a one-stage cache would never hit: any hit below
    # must come from the candidate threshold (0.80) plus the verifier.
    return make_settings(SEMANTIC_VERIFY_ENABLED=True, SEMANTIC_CANDIDATE_THRESHOLD=0.80,
                         SEMANTIC_VERIFY_THRESHOLD=0.5, SEMANTIC_THRESHOLD=0.999, **overrides)


async def post(gw, text):
    return await gw.post("/v1/chat/completions", json=chat(text), headers=AUTH)


async def test_verifier_pass_is_a_hit():
    v = FakeVerifier(score=0.9)
    async with gateway(two_stage(), verifier=v) as gw:
        await post(gw, FIRST)
        r = await post(gw, PARAPHRASE)
        assert r.headers["X-Cache"] == "HIT-SEMANTIC"
        assert r.headers["X-Verify-Score"] == "0.9000"
        assert gw.upstream.calls["primary.test"] == 1
    # The verifier saw (new question, cached question), both normalised like the cache stores them.
    assert v.pairs == [("how can i learn python quickly?", "how do i learn python quickly?")]


async def test_verifier_fail_is_a_miss():
    v = FakeVerifier(score=0.2)
    async with gateway(two_stage(), verifier=v) as gw:
        await post(gw, FIRST)
        r = await post(gw, PARAPHRASE)
        assert r.headers["X-Cache"] == "MISS"
        assert r.headers["X-Verify-Score"] == "0.2000"
        assert gw.upstream.calls["primary.test"] == 2
    assert len(v.pairs) == 1


async def test_verifier_error_is_a_miss_not_an_unchecked_hit():
    v = FakeVerifier(error=RuntimeError("model crashed"))
    async with gateway(two_stage(), verifier=v) as gw:
        await post(gw, FIRST)
        r = await post(gw, PARAPHRASE)
        assert r.status_code == 200
        assert r.headers["X-Cache"] == "MISS"


async def test_below_candidate_threshold_never_calls_verifier():
    v = FakeVerifier(score=1.0)
    async with gateway(two_stage(), verifier=v) as gw:
        await post(gw, FIRST)
        r = await post(gw, "What is the boiling point of water at sea level?")
        assert r.headers["X-Cache"] == "MISS"
    assert v.pairs == []


async def test_verify_disabled_keeps_one_stage_behaviour():
    v = FakeVerifier(score=0.0)  # would reject everything, if it were used
    settings = make_settings(SEMANTIC_VERIFY_ENABLED=False, SEMANTIC_THRESHOLD=0.80)
    async with gateway(settings, verifier=v) as gw:
        await post(gw, FIRST)
        r = await post(gw, PARAPHRASE)
        assert r.headers["X-Cache"] == "HIT-SEMANTIC"
        assert "X-Verify-Score" not in r.headers
    assert v.pairs == []
