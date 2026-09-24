"""Token-bucket rate limiter, one bucket per API key, stored in Redis.

Picture a bucket that holds BUCKET_CAPACITY tokens and refills at REFILL_PER_SEC.
Each request takes one token. Empty bucket -> 429 Too Many Requests.

Why a Lua script? "Read tokens, compute, write tokens" is three steps. If two gateway
replicas did those steps at the same time, both could read "1 token left" and both let a
request through. Redis runs a Lua script as ONE atomic step, so that race can't happen.
The script also reads Redis's own clock (TIME), so replicas with slightly different
clocks still agree.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.exceptions import RedisError

log = logging.getLogger("gateway.ratelimit")

TOKEN_BUCKET_LUA = """
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])   -- tokens per second
local cost     = tonumber(ARGV[3])   -- tokens this request needs (1)

local t   = redis.call('TIME')       -- {seconds, microseconds} from Redis's clock
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local data   = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts     = tonumber(data[2])
if tokens == nil then
  tokens = capacity                  -- new key starts with a full bucket
  ts = now
end

-- refill for the time that passed, but never above capacity
local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = (cost - tokens) / rate   -- seconds until enough tokens exist
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
-- forget idle buckets once they would be full again anyway
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)

-- Redis turns Lua numbers into integers, so send decimals back as strings
return {allowed, tostring(tokens), tostring(retry_after)}
"""


@dataclass
class RateDecision:
    allowed: bool
    remaining: float
    retry_after_s: float
    degraded: bool = False  # True when Redis failed and we let the request through


class RateLimiter:
    def __init__(self, redis, capacity: float, refill_per_sec: float):
        self.redis = redis
        self.capacity = capacity
        self.rate = refill_per_sec
        self._script = redis.register_script(TOKEN_BUCKET_LUA) if redis is not None else None

    async def check(self, key_id: str, cost: float = 1) -> RateDecision:
        if self._script is None:
            return RateDecision(True, self.capacity, 0.0, degraded=True)
        try:
            allowed, tokens, retry_after = await self._script(
                keys=[f"rl:{key_id}"], args=[self.capacity, self.rate, cost]
            )
            return RateDecision(bool(int(allowed)), float(tokens), float(retry_after))
        except (RedisError, OSError) as exc:
            # Fail open: a broken Redis must not take the whole gateway down.
            log.warning("rate limiter unavailable, failing open: %s", exc)
            return RateDecision(True, 0.0, 0.0, degraded=True)
