from gateway.breaker import CircuitBreaker, State

from .conftest import AUTH, BACKUP, PRIMARY, chat, gateway, make_settings


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_opens_after_n_consecutive_failures():
    b = CircuitBreaker(fail_threshold=5, cooldown_s=30, clock=FakeClock())
    for _ in range(4):
        b.record_failure()
    assert b.state == State.CLOSED and b.allow()
    b.record_failure()
    assert b.state == State.OPEN and not b.allow()


def test_success_resets_the_count():
    b = CircuitBreaker(5, 30, clock=FakeClock())
    for _ in range(4):
        b.record_failure()
    b.record_success()
    for _ in range(4):
        b.record_failure()
    assert b.state == State.CLOSED  # failures must be CONSECUTIVE


def test_half_opens_after_cooldown_and_allows_one_trial():
    clock = FakeClock()
    b = CircuitBreaker(2, 30, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.t += 29.9
    assert not b.allow()
    clock.t += 0.2
    assert b.allow()            # the single test request
    assert b.state == State.HALF_OPEN
    assert not b.allow()        # everyone else still waits


def test_closes_on_trial_success():
    clock = FakeClock()
    b = CircuitBreaker(2, 30, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.t += 31
    assert b.allow()
    b.record_success()
    assert b.state == State.CLOSED and b.allow()


def test_reopens_on_trial_failure():
    clock = FakeClock()
    b = CircuitBreaker(2, 30, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.t += 31
    assert b.allow()
    b.record_failure()
    assert b.state == State.OPEN and not b.allow()
    assert 29 < b.retry_after_s() <= 30


def test_stuck_half_open_trial_is_replaced_after_cooldown():
    clock = FakeClock()
    b = CircuitBreaker(2, 30, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.t += 31
    assert b.allow()        # test request goes out... and never reports back
    clock.t += 10
    assert not b.allow()
    clock.t += 21
    assert b.allow()        # a new test request is allowed instead of waiting forever


async def test_unexpected_error_on_trial_still_counts_as_failure(upstream):
    import httpx

    upstream.set(PRIMARY, httpx.DecodingError)  # not a timeout/transport error
    s = make_settings(BREAKER_FAILS=1, RETRY_MAX=3)
    async with gateway(s, upstream) as gw:
        r = await gw.post("/v1/chat/completions", json=chat(temperature=1), headers=AUTH)
        assert r.status_code == 200 and r.headers["X-Provider"] == "mock2"  # fell back, no 500
        assert upstream.calls[PRIMARY] == 1
        assert (await gw.get("/health")).json()["providers"]["mock"] == "open"


async def test_open_breaker_skips_provider_instantly(upstream):
    upstream.set(PRIMARY, 503)
    s = make_settings(BREAKER_FAILS=3, RETRY_MAX=3)
    async with gateway(s, upstream) as gw:
        r1 = await gw.post("/v1/chat/completions", json=chat(temperature=1), headers=AUTH)
        assert r1.headers["X-Provider"] == "mock2" and upstream.calls[PRIMARY] == 3
        r2 = await gw.post("/v1/chat/completions", json=chat(temperature=1), headers=AUTH)
        assert r2.headers["X-Provider"] == "mock2"
        assert upstream.calls[PRIMARY] == 3  # breaker open: primary not called again
        assert (await gw.get("/health")).json()["providers"]["mock"] == "open"


async def test_all_breakers_open_gives_503_with_retry_after(upstream):
    upstream.set(PRIMARY, 503)
    upstream.set(BACKUP, 503)
    s = make_settings(BREAKER_FAILS=3, RETRY_MAX=3)
    async with gateway(s, upstream) as gw:
        r1 = await gw.post("/v1/chat/completions", json=chat(temperature=1), headers=AUTH)
        assert r1.status_code == 502
        r2 = await gw.post("/v1/chat/completions", json=chat(temperature=1), headers=AUTH)
        assert r2.status_code == 503
        assert int(r2.headers["Retry-After"]) >= 1
        assert upstream.calls[PRIMARY] == 3 and upstream.calls[BACKUP] == 3
