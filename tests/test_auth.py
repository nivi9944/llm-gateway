from .conftest import AUTH, chat, gateway


async def test_missing_key_is_401():
    async with gateway() as gw:
        r = await gw.post("/v1/chat/completions", json=chat())
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"
        assert gw.upstream.calls["primary.test"] == 0  # rejected before any provider call


async def test_wrong_key_is_401():
    async with gateway() as gw:
        r = await gw.post("/v1/chat/completions", json=chat(), headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401


async def test_valid_bearer_key_passes():
    async with gateway() as gw:
        r = await gw.post("/v1/chat/completions", json=chat(), headers=AUTH)
        assert r.status_code == 200
        assert r.headers["X-Cache"] == "MISS"
        assert r.headers["X-Provider"] == "mock"


async def test_x_api_key_header_also_works():
    async with gateway() as gw:
        r = await gw.post("/v1/chat/completions", json=chat(), headers={"X-API-Key": "other-key"})
        assert r.status_code == 200


async def test_metrics_summary_needs_key_but_health_does_not():
    async with gateway() as gw:
        assert (await gw.get("/metrics-summary")).status_code == 401
        assert (await gw.get("/metrics-summary", headers=AUTH)).status_code == 200
        h = await gw.get("/health")
        assert h.status_code == 200 and h.json()["redis"] == "up"


async def test_bad_body_is_400():
    async with gateway() as gw:
        r = await gw.post("/v1/chat/completions", json={"model": "x", "messages": []}, headers=AUTH)
        assert r.status_code == 400
        r = await gw.post("/v1/chat/completions", json={**chat(), "stream": True}, headers=AUTH)
        assert r.status_code == 400
        r = await gw.post("/v1/chat/completions", json={"model": "x", "messages": ["hi"]}, headers=AUTH)
        assert r.status_code == 400  # not a 500


def test_zero_refill_rate_is_rejected_at_startup():
    import pytest
    from pydantic import ValidationError

    from .conftest import make_settings

    with pytest.raises(ValidationError):
        make_settings(REFILL_PER_SEC=0)
