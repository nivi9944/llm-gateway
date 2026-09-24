import httpx

from gateway.router import backoff_s

from .conftest import AUTH, BACKUP, PRIMARY, chat, gateway, make_settings


class SleepRecorder:
    def __init__(self):
        self.waits: list[float] = []

    async def __call__(self, s: float):
        self.waits.append(s)


async def post(gw, body=None, headers=None):
    return await gw.post("/v1/chat/completions", json=body or chat(temperature=0.5),
                         headers={**AUTH, **(headers or {})})


async def test_retries_on_500_with_backoff_then_succeeds(upstream):
    upstream.set(PRIMARY, [500, 500, 200])
    sleeper = SleepRecorder()
    s = make_settings(RETRY_BASE_S=0.5, RETRY_CAP_S=4.0)
    async with gateway(s, upstream, sleep=sleeper) as gw:
        r = await post(gw)
        assert r.status_code == 200
        assert r.headers["X-Provider"] == "mock"
        assert r.headers["X-Attempts"] == "3"
        assert upstream.calls[PRIMARY] == 3 and upstream.calls[BACKUP] == 0
        # two waits, each within its full-jitter window: [0, 0.5] then [0, 1.0]
        assert len(sleeper.waits) == 2
        assert 0 <= sleeper.waits[0] <= 0.5 and 0 <= sleeper.waits[1] <= 1.0


async def test_no_retry_on_400(upstream):
    upstream.set(PRIMARY, 400)
    async with gateway(upstream=upstream) as gw:
        r = await post(gw)
        assert r.status_code == 400
        assert upstream.calls[PRIMARY] == 1
        assert upstream.calls[BACKUP] == 0  # a bad request would fail everywhere


async def test_fallback_when_primary_always_fails(upstream):
    upstream.set(PRIMARY, 503)
    async with gateway(upstream=upstream) as gw:
        r = await post(gw)
        assert r.status_code == 200
        assert r.headers["X-Provider"] == "mock2"
        assert upstream.calls[PRIMARY] == 3 and upstream.calls[BACKUP] == 1


async def test_timeouts_are_retried(upstream):
    upstream.set(PRIMARY, [httpx.ReadTimeout, 200])
    async with gateway(upstream=upstream) as gw:
        r = await post(gw)
        assert r.status_code == 200 and upstream.calls[PRIMARY] == 2


async def test_401_from_provider_falls_back_without_retry(upstream):
    upstream.set(PRIMARY, 401)  # e.g. a bad provider key: retrying can't fix it
    async with gateway(upstream=upstream) as gw:
        r = await post(gw)
        assert r.status_code == 200 and r.headers["X-Provider"] == "mock2"
        assert upstream.calls[PRIMARY] == 1


async def test_provider_retry_after_is_respected(upstream):
    upstream.set(PRIMARY, [429, 200])
    upstream.headers[PRIMARY] = {"Retry-After": "2"}
    sleeper = SleepRecorder()
    s = make_settings(RETRY_BASE_S=0.01, RETRY_CAP_S=5.0)
    async with gateway(s, upstream, sleep=sleeper) as gw:
        r = await post(gw)
        assert r.status_code == 200
        assert sleeper.waits == [2.0]


async def test_all_providers_fail_gives_502(upstream):
    upstream.set(PRIMARY, 500)
    upstream.set(BACKUP, 500)
    async with gateway(upstream=upstream) as gw:
        r = await post(gw)
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "upstream_error"


async def test_force_provider_header(upstream):
    async with gateway(upstream=upstream) as gw:
        r = await post(gw, headers={"X-Provider-Force": "mock2"})
        assert r.headers["X-Provider"] == "mock2" and upstream.calls[PRIMARY] == 0
        bad = await post(gw, headers={"X-Provider-Force": "nope"})
        assert bad.status_code == 400


async def test_provider_gets_its_own_model_name(upstream):
    async with gateway(upstream=upstream) as gw:
        await post(gw, chat(model="whatever", temperature=0.5))
        assert upstream.bodies[-1]["model"] == "mock-1"


def test_full_jitter_stays_inside_window():
    for attempt in range(6):
        for _ in range(200):
            w = backoff_s(attempt, base=0.25, cap=4.0)
            assert 0 <= w <= min(4.0, 0.25 * 2**attempt)
