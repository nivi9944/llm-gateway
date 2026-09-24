# Design decisions

Each entry: what was decided, why, and what was given up. Useful for interviews: most
"why did you…?" questions are answered here.

## 1. FastAPI + fully async + one shared `httpx.AsyncClient`
**Why:** a gateway mostly *waits* (on providers, on Redis). Async lets one process handle many
waiting requests at once instead of one thread per request. One shared client reuses TCP/TLS
connections instead of opening a new one per call.
**Trade-off:** CPU work (embedding) would block the event loop, so it runs in a worker thread
(`asyncio.to_thread`).

## 2. OpenAI-compatible API (`/v1/chat/completions`)
**Why:** Gemini and Ollama both expose OpenAI-compatible endpoints, so one request shape works for
every provider, and apps can use any OpenAI SDK by changing `base_url`.
**Detail:** the gateway replaces `model` with each provider's own model name.

## 3. Gemini free tier as primary, local Ollama as fallback
**Why:** both are free; cloud → local fallback makes the reliability demo real.
**Not used:** the Claude API for runtime calls (not included in the Max plan).

## 4. Benchmarks against a mock provider
**Why:** free, repeatable (seeded), and failures can be injected on purpose. Real providers add
noise (network, quotas) that would make numbers unrepeatable.

## 5. Exact cache: SHA-256 of messages + all answer-changing parameters, in Redis with TTL
**Why:** hashing gives a fixed-size key. The key covers the full messages (including `tool_calls`,
`name`, …) and every request field except `stream`, `stream_options` and `user`, so `max_tokens`,
`tools`, `response_format`, `stop`, `seed`, `top_p` and any future field all separate entries.
(The original plan hashed only model + messages + temperature; a review showed two requests differing
only in `max_tokens` or `tools` would then share an answer.)
Only leading/trailing whitespace is trimmed: newlines and indentation change the meaning of code.
Case is kept too. Looser matching is the semantic cache's job.
**Only when `temperature == 0`:** other temperatures ask for varied answers. A request that omits
`temperature` gets OpenAI's default (1) and is not cached.

## 6. Semantic cache: MiniLM + FAISS `IndexFlatIP`, answers in Redis
**Why:** people paraphrase; exact matching misses them. MiniLM (384-d) is small and CPU-friendly.
Vectors are normalised, so inner product = cosine similarity. A flat (exact) index is simple and
fast at this scale (thousands to low millions of vectors).
**Safety:** only the *last user message* is embedded (whitespace collapsed, lowercased), and it is
compared only with requests that share the same parameters and the same earlier messages (a
"namespace" hash). A question under a different system prompt or `max_tokens` can never match.
On a hit, the stored entry's namespace and vector are re-checked, so an id reused after Redis lost its
data can never return another conversation's answer.
**Restart:** each Redis entry also stores its vector, so a restarted gateway rebuilds FAISS from Redis.
**Trade-off:** FAISS is per-process. Several replicas would each hold their own index (see README
"Limitations"). An expired Redis answer is detected at lookup time and its vector removed.
**Ids** come from a Redis `INCR`, so they are unique across replicas.

## 7. Threshold chosen from data, not guessed
**Rule:** the lowest threshold whose precision is ≥ 95% on labelled QQP pairs (validation split).
Lowest, because a lower threshold catches more paraphrases (more savings) while still meeting the bar.
**Note:** QQP labels are noisy (some "duplicates" are debatable). If no threshold reaches 95%, the
script says so (`target_met: false`) and reports the best one instead of hiding it.

**Result (laptop run, `results/qqp_threshold.json`, 20,000 sampled QQP validation pairs, MiniLM):**
the 95% target was **not met**. The best precision was at threshold **0.98**: precision 0.9238,
recall 0.0759, false-hit rate 0.0036 (558 true hits, 46 false hits). 0.99 scored lower (0.9195), and
0.97 fell to 0.8941, so 0.98 is also the pick for a 90% target. We set `SEMANTIC_THRESHOLD=0.98`:
a wrong cached answer is worse than a cache miss, so we chose the strictest setting and quote its
real precision instead of changing the data or method. Trade-off: it catches only 7.59% of paraphrases (recall 0.0759);
MiniLM cannot separate "same meaning" from "similar wording" at 95% on QQP.

## 8. Rate limiting: token bucket per key, as a Redis Lua script
**Why token bucket:** allows short bursts (capacity) while enforcing an average rate (refill).
**Why Lua:** "read tokens → compute → write tokens" must be atomic. Two replicas doing it at the same
time could both see "1 token left" and both pass. Redis runs a script as one step. The script
reads Redis's own clock (`TIME`), so replicas with slightly different clocks still agree.
**Alternatives:** fixed window (bursty at window edges), sliding-window log (exact but stores every
timestamp), sliding-window counter (a good approximation). Token bucket is the usual choice for APIs.
**Response:** 429 with `Retry-After` = seconds until one token exists.

## 9. Retry policy
Retry only timeouts, connection errors and HTTP 429/500/502/503/504, at most 3 tries per provider.
400/413/422 are the client's fault: returned immediately, no retry, no fallback. 401/403/404 are
provider-specific (e.g. a bad key): no retry, but fall back.
**Full jitter:** wait = random(0, min(cap, base × 2^attempt)). Without randomness, clients that failed
together retry together and overload the provider again (thundering herd).
A provider's `Retry-After` header is honoured (up to the cap).

## 10. Circuit breaker per provider
CLOSED → OPEN after 5 *consecutive* failed calls (each retry counts) → HALF-OPEN after 30 s → a single
test request decides (success → CLOSED, failure → OPEN again). Every call reports its outcome in a
`finally` block, and a test request that never reports back is replaced after another cooldown, so
the breaker can't get stuck half-open.
**Why:** during an outage, retries alone keep hammering a dead provider and add seconds of waiting to
every request. The chaos benchmark measures this (`breaker_effect_during_outage`).
If every provider's breaker is open, the gateway answers 503 with `Retry-After`.

## 11. Fail open when Redis is down
**Why:** availability over cost. A cache/limiter outage should not become a full outage. Requests go
straight to providers and a warning is logged. Redis calls use a short timeout (0.5 s) so failing is fast.
**Trade-off:** no rate limiting while Redis is down. For a paid public API you might fail *closed* for the
limiter instead.

## 12. Cost = tokens × Gemini paid-tier list price, labelled "estimated"
Free-tier calls cost 0, which would make every saving 0. Pricing at the paid list price
(`gemini-3.8-flash`: $0.75 / $3.75 per 1M input / output tokens on 2026-09-24) gives a meaningful
"% saved". The percentage barely depends on the exact price, since both sides use the same price.

## 13. Benchmark design choices
* **Replay mix** (40% new, 20% near-miss, 20% exact repeat, 20% paraphrase) is an *assumption* about a
  FAQ-like workload. Hit rate depends on it, so the mix is saved in `replay.json` and must be quoted
  with the numbers. Replay uses QQP *train* minus any pair sharing a question with *validation* (used
  for the threshold), so tuning and testing never share text. Every hit (exact or semantic) is scored:
  correct only if the new question and the one that produced the answer are linked by QQP duplicate
  labels (chained: A=B and B=C means A=C).
* **Load test** users start gradually (5/s) and the ramp-up is excluded. Starting all 50 at once makes
  them move in lock-step waves that hit the gateway as one burst every 800 ms: a test artifact.
  Every request is unique so it takes the full (most expensive) path. Percentiles come from exact
  per-request times (Locust's own CSV rounds values above 100 ms).
* **Load/chaos numbers for the resume** must come from one's own laptop (they depend on hardware).

## 14. Tests run offline
Providers are faked with `httpx.MockTransport`, Redis with `fakeredis` (Lua supported via `lupa`), and
embeddings with a tiny hashing "bag of words" embedder. The suite runs in ~10 s with no network.
CI runs it twice: once on fakeredis, once on a real Redis service (to check the Lua script for real).
One test uses the real MiniLM model and is skipped if it can't be downloaded.

## 15. Packaging
One Docker image serves both the gateway and the mock (different commands). CPU-only PyTorch keeps it
~2 GB smaller. The MiniLM model is downloaded at build time so containers start offline.
Ollama runs on the Windows host; containers reach it at `host.docker.internal:11434`.
