# Design decisions

Each entry states what was chosen, why, and what it costs. Numbers come from `results/*.json`
(file named next to each one).

## 1. FastAPI, fully async, one shared `httpx.AsyncClient`
**Why:** a gateway spends most of its time waiting (on providers, on Redis). Async lets one process
hold many waiting requests without a thread each. One shared HTTP client reuses TCP/TLS connections
instead of opening a new one per call.
**Trade-off:** CPU work (embedding, cross-encoder scoring) would block the event loop, so it runs on a
dedicated thread pool (section 8).

## 2. OpenAI-compatible API (`/v1/chat/completions`)
**Why:** Gemini and Ollama both expose OpenAI-compatible endpoints, so one request shape works for
every provider, and any OpenAI SDK works by changing `base_url`.
**Detail:** the gateway replaces `model` with each provider's own model name.

## 3. Gemini Flash as primary, local Ollama (Qwen2.5) as fallback
**Why:** a hosted model for quality and a local model that keeps answering when the cloud provider
fails, rate-limits or is unreachable. Both are free to run for this project.

## 4. Benchmarks run against a mock provider
**Why:** free, repeatable (seeded), and failures can be injected on purpose. Real providers add network
and quota noise that would make results unrepeatable. A separate live check (`bench/smoke_live.py`)
confirms the real Gemini and Ollama paths work.

## 5. Exact cache: SHA-256 of messages and every answer-changing parameter, in Redis with TTL
**Why:** hashing gives a fixed-size key. The key covers the full messages (including `tool_calls`,
`name`) and every request field except `stream`, `stream_options` and `user`, so `max_tokens`, `tools`,
`response_format`, `stop`, `seed`, `top_p` and any future field all separate entries. Two requests that
differ only in `max_tokens` must never share an answer.
Only leading and trailing whitespace is trimmed: newlines and indentation change the meaning of code,
and case is kept. Looser matching is the semantic cache's job.
**Only when `temperature == 0`:** other temperatures ask for varied answers. A request that omits
`temperature` gets OpenAI's default (1) and is not cached.

## 6. Semantic cache: MiniLM + FAISS `IndexFlatIP`, answers in Redis
**Why:** people paraphrase, and exact matching misses them. MiniLM (384 dimensions) is small and
CPU-friendly. Vectors are normalised, so inner product equals cosine similarity. A flat (exact) index
is simple and fast at this scale (thousands to low millions of vectors).
**Safety:** only the last user message is embedded (whitespace collapsed, lowercased), and it is
compared only with requests that share the same parameters and the same earlier messages (a
"namespace" hash). A question under a different system prompt or `max_tokens` can never match.
On a hit, the stored entry's namespace and vector are re-checked, so an id reused after Redis lost its
data can never return another conversation's answer.
**Restart:** each Redis entry also stores its vector, so a restarted gateway rebuilds FAISS from Redis.
**Trade-off:** FAISS is per-process, so several replicas would each hold their own index (see README,
"Limitations"). An expired answer is detected at lookup time and its vector removed. Ids come from a
Redis `INCR`, so they are unique across replicas.

## 7. Two-stage semantic cache: embedding finds a candidate, a cross-encoder decides
**Problem:** an embedding model turns each question into a vector on its own, so it mostly measures
shared wording. With MiniLM alone, no threshold reached 95% precision on 20,000 labelled QQP
validation pairs. The best setting (0.98) had precision 0.9238 and caught only 7.59% of paraphrases
(`results/qqp_threshold_single_stage.json`).
**Design:** FAISS proposes the single closest cached question if its similarity is at least 0.85. A
cross-encoder (`cross-encoder/quora-distilroberta-base`) then reads the new and cached questions
together and must score the pair at least 0.98 before the answer is reused. Reading both together lets
it notice the small word that changes the meaning. It is too slow to compare against every cached
question, which is why stage 1 exists.
**How the thresholds were chosen:** the candidate threshold is fixed at 0.85; the cross-encoder
threshold is swept from 0.05 to 0.99, and the lowest one with precision of at least 95% is used
(lowest, because it catches the most paraphrases while meeting the bar).
**Result (`results/qqp_threshold.json`):** precision 0.9549, recall 0.3803, false-hit rate 0.0104.
Stage 1 alone at 0.85 has recall 0.6116, which is the ceiling for the two stages together.
**Trade-offs:**
* More reuse also means more mistakes. In the replay benchmark the two-stage cache served 51 wrong
  answers in 10,000 requests, against 10 for MiniLM alone, while its hit rate rose from 0.2255 to
  0.2882 (`results/replay.json` vs `results/replay_single_stage.json`).
* Latency: p95 gateway overhead is 332.4 ms with the cross-encoder and 268.9 ms without it
  (`results/load_optimized.json` vs `results/load_threadpool_batching.json`).
* Only FAISS's best candidate is checked. If it is rejected, a second-best match is not tried.
**Data caveat:** this cross-encoder was trained on QQP *train*. It is evaluated only on QQP
*validation*, and the replay workload drops train pairs that share a question with validation, but
the model has seen the same kind of Quora questions, so real traffic may score lower.
**Switch:** `SEMANTIC_VERIFY_ENABLED=false` returns to the embedding-only cache (`SEMANTIC_THRESHOLD`).
A verifier error is treated as a miss, never as an unchecked hit.
**Note on labels:** QQP labels are noisy (some "duplicates" are debatable), so precision figures carry
that noise too.

## 8. CPU work on a dedicated thread pool, with micro-batching
**Problem:** with 50 concurrent users, each embedding call tried to use every core and the calls fought
each other; p95 gateway overhead was 1270.1 ms (`results/load_baseline.json`).
**Thread pool:** each torch call is limited to 1 thread, and the semantic cache has its own 4-thread
pool, separate from asyncio's shared default pool. p95 overhead: 529.4 ms
(`results/load_threadpool_only.json`). 8 workers with 2 threads each was slower, at 1272.1 ms
(`results/load_trial_8x2.json`).
**Micro-batching:** embedding requests that arrive within 5 ms (up to 32) are sent to the model as one
batch, and each caller gets its own vector back. One matrix multiply for 32 sentences is much cheaper
than 32 small ones. p95 overhead: 268.9 ms (`results/load_threadpool_batching.json`).
**Trade-off:** a lone request can wait up to 5 ms for others to join, which is small next to an LLM call.

## 9. Rate limiting: token bucket per key, as a Redis Lua script
**Why token bucket:** allows short bursts (capacity) while enforcing an average rate (refill).
**Why Lua:** "read tokens, compute, write tokens" must be atomic. Two replicas doing it at the same
time could both see "1 token left" and both pass; Redis runs a script as one step. The script reads
Redis's own clock (`TIME`), so replicas with slightly different clocks still agree.
**Alternatives:** fixed window (bursty at window edges), sliding-window log (exact but stores every
timestamp), sliding-window counter (a good approximation). Token bucket is the usual choice for APIs.
**Response:** 429 with `Retry-After` set to the seconds until one token exists.

## 10. Retry policy
Retry only timeouts, connection errors and HTTP 429/500/502/503/504, at most 3 tries per provider.
400/413/422 are the client's fault: returned immediately, with no retry and no fallback. 401/403/404
are provider-specific (for example a bad key): no retry, but fall back.
**Full jitter:** wait = random(0, min(cap, base × 2^attempt)). Without randomness, clients that failed
together retry together and overload the provider again (thundering herd).
A provider's `Retry-After` header is honoured, up to the cap.

## 11. Circuit breaker per provider
CLOSED, then OPEN after 5 consecutive failed calls (each retry counts), then HALF-OPEN after 30 s, when
a single test request decides (success closes it, failure opens it again). Every call reports its
outcome in a `finally` block, and a test request that never reports back is replaced after another
cooldown, so the breaker cannot get stuck half-open.
**Why:** during an outage, retries alone keep hammering a dead provider and add seconds to every
request. In the chaos benchmark the breaker cut calls to a dead provider from 6,000 to 20
(`results/chaos.json`). If every provider's breaker is open, the gateway answers 503 with `Retry-After`.

## 12. Fail open when Redis is down
**Why:** availability over cost. A cache or limiter outage should not become a full outage. Requests go
straight to providers and a warning is logged. Redis calls use a short timeout (0.5 s) so failing is fast.
**Trade-off:** no rate limiting while Redis is down. A paid public API might fail closed for the limiter
instead.

## 13. Cost is tokens × Gemini paid-tier list price, labelled "estimated"
Free-tier calls cost nothing, which would make every saving zero. Pricing at the paid list price
($0.75 / $3.75 per 1M input / output tokens, set in `gateway/config.py`) gives a meaningful "% saved".
The percentage barely depends on the exact price, since both sides of the comparison use it.

## 14. Benchmark design
* **Replay mix** (40% new, 20% near-miss, 20% exact repeat, 20% paraphrase) is an assumption about an
  FAQ-like workload. Hit rate depends on it, so the realised mix is saved in `replay.json` and quoted
  with the results. Replay uses QQP *train* minus any pair sharing a question with *validation* (used for
  tuning), so tuning and testing never share text. Every hit is scored: it is correct only if QQP's
  duplicate labels link the new question to the one that produced the answer (chained: A=B and B=C
  means A=C).
* **Load test** users start gradually (5 per second) and the ramp-up is excluded. Starting all 50 at
  once makes them move in lock-step waves that hit the gateway as one burst every 800 ms, which is a
  test artifact. Every request is unique, so it takes the full, most expensive path. Percentiles come
  from exact per-request times (Locust's own CSV rounds values above 100 ms).
* **Hardware:** load and chaos results depend on the machine; each result file records it.

## 15. Tests run offline
Providers are faked with `httpx.MockTransport`, Redis with `fakeredis` (Lua supported via `lupa`),
embeddings with a tiny hashing "bag of words" embedder, and the cross-encoder with a fake verifier. The
suite needs no network. CI runs it twice: once on fakeredis and once on a real Redis service, to check
the Lua script for real. One test uses the real MiniLM model and is skipped if it cannot be downloaded.

## 16. Packaging
One Docker image serves both the gateway and the mock provider (different commands). CPU-only PyTorch
keeps it about 2 GB smaller. Both models are downloaded at build time, so containers start offline.
Ollama runs on the host; containers reach it at `host.docker.internal:11434`.
