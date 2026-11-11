"""Semantic cache: reuse a stored answer when a new question means the same as an old one.

"How do I learn Python fast?" and "Quickest way to learn Python?" share few exact words,
so the exact cache misses them. The question is embedded with MiniLM, and FAISS finds the
closest stored vector by cosine similarity. In one-stage mode a similarity >= SEMANTIC_THRESHOLD
is a hit. In two-stage mode FAISS only proposes a candidate (SEMANTIC_CANDIDATE_THRESHOLD)
and a cross-encoder must confirm the pair (SEMANTIC_VERIFY_THRESHOLD).

Guards against serving a wrong answer:
  * only the last user message is embedded, and matches are searched only among requests
    with the same model and the same earlier conversation (the namespace), so a question
    under a different system prompt never matches;
  * thresholds are chosen from labelled data (bench/qqp_threshold.py);
  * only temperature-0 requests are cached (checked in main.py), and entries expire (TTL).

FAISS (in memory) holds the vectors for search. Redis holds the answers plus a copy of each
vector, so a restarted gateway can rebuild its FAISS index.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from redis.exceptions import RedisError

from .cache import collapse_text, normalise_messages, request_params

log = logging.getLogger("gateway.semantic")

ID_COUNTER_KEY = "semid:next"
ENTRY_PREFIX = "sem:"
# CPU work (embedding + FAISS search) runs on this many dedicated threads, each torch call on
# TORCH_THREADS cores. Many small requests in parallel do better than one request using every core.
POOL_WORKERS = 4
TORCH_THREADS = 1


# ---------------------------------------------------------------- embedders
class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, rows unit-length
        ...


class MiniLMEmbedder:
    """Production embedder: sentence-transformers/all-MiniLM-L6-v2 (384-dim, runs well on CPU)."""

    dim = 384

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer  # heavy import, done once

        # By default every encode() uses all cores. With several requests in parallel the
        # threads contend with each other and latency climbs sharply under load.
        torch.set_num_threads(TORCH_THREADS)

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Small dependency-free embedder for unit tests and offline trial runs.

    Bag-of-words with the hashing trick: each word adds +-1 to one of 384 slots.
    It only captures shared words, so it is never used for reported results.
    """

    dim = 384
    _TOKEN = re.compile(r"[a-z0-9]+")

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def build_embedder(kind: str, model_name: str) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    return MiniLMEmbedder(model_name)


# ---------------------------------------------------------------- verifier (stage 2)
class Verifier(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:  # 0..1, higher = same meaning
        ...


class CrossEncoderVerifier:
    """Stage 2 of the semantic cache: a cross-encoder that reads both questions together.

    MiniLM (stage 1) embeds each question on its own, so it mostly reflects shared wording.
    A cross-encoder scores the pair jointly and can catch small words that flip the meaning
    ("with" vs "without"). It is slower, so it only checks the single best FAISS candidate.
    """

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import CrossEncoder  # heavy import, done once at startup

        torch.set_num_threads(TORCH_THREADS)
        self.model = CrossEncoder(model_name, device="cpu")

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        # This model has a single sigmoid output, so scores are already in 0..1.
        return [float(s) for s in self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)]


# ---------------------------------------------------------------- batching
class EmbedBatcher:
    """Micro-batcher that merges embedding requests arriving close together into one encode().

    Encoding 32 sentences in one call costs far less than 32 separate calls (the model runs
    one matrix multiply instead of 32 small ones). Each caller waits at most `max_wait_s`
    (5 ms) for others to join; a full batch (`max_size`) is sent at once without waiting.
    Every caller gets back only its own vector; if encode() fails, every caller gets the error.
    """

    def __init__(self, encode, pool: ThreadPoolExecutor, max_wait_s: float = 0.005, max_size: int = 32):
        self._encode = encode
        self._pool = pool
        self.max_wait_s = max_wait_s
        self.max_size = max_size
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._timer: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task] = set()  # keep running batches referenced until done

    async def embed(self, text: str) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending.append((text, fut))
        if len(self._pending) >= self.max_size:
            self._flush()
        elif self._timer is None:
            self._timer = loop.call_later(self.max_wait_s, self._flush)
        return await fut

    def _flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        batch, self._pending = self._pending, []
        if batch:
            task = asyncio.get_running_loop().create_task(self._run(batch))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run(self, batch: list[tuple[str, asyncio.Future]]) -> None:
        loop = asyncio.get_running_loop()
        try:
            vecs = await loop.run_in_executor(self._pool, self._encode, [text for text, _ in batch])
        except Exception as exc:  # every waiting caller gets the same error
            for _, fut in batch:
                if not fut.done():  # the caller may have been cancelled meanwhile
                    fut.set_exception(exc)
            return
        for (_, fut), vec in zip(batch, vecs):
            if not fut.done():
                fut.set_result(vec)


# ---------------------------------------------------------------- cache
@dataclass
class SemanticQuery:
    namespace: str
    text: str
    vector: np.ndarray  # shape (dim,)


@dataclass
class SemanticResult:
    response: dict | None
    similarity: float | None
    match_text: str | None = None
    verify_score: float | None = None  # stage-2 score, when the verifier ran


def build_query_parts(body: dict) -> tuple[str, str] | None:
    """Return (namespace, text) or None if this request can't use the semantic cache.

    namespace = hash of everything except the last user message (model, parameters, tools,
    system prompt, earlier turns). Only requests with the same namespace can match.
    """
    messages = body.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    content = last.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    extra = {k: v for k, v in last.items() if k not in ("role", "content")}  # e.g. "name"
    context = {"params": request_params(body), "history": normalise_messages(messages[:-1]), "last_extra": extra}
    ns = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return ns, collapse_text(content)


class SemanticCache:
    def __init__(self, redis, embedder: Embedder, threshold: float, ttl_s: int,
                 verifier: Verifier | None = None, verify_threshold: float = 0.5):
        """`threshold` is the FAISS cut-off. Without a verifier it alone decides a hit (one
        stage). With a verifier it only selects a candidate, and the answer is reused only if
        the verifier's score is >= `verify_threshold` (two stages).
        """
        import faiss  # imported here so the rest of the gateway loads without it

        faiss.omp_set_num_threads(1)  # searches are one query at a time, where OpenMP threads only add overhead
        self._faiss = faiss
        self.redis = redis
        self.embedder = embedder
        self.threshold = threshold
        self.verifier = verifier
        self.verify_threshold = verify_threshold
        self.ttl_s = ttl_s
        self._indexes: dict[str, object] = {}  # namespace -> faiss.IndexIDMap2(IndexFlatIP)
        self._lock = threading.Lock()  # FAISS indexes are not safe for concurrent writes
        # A dedicated pool, so embedding neither starves nor is starved by asyncio's default pool.
        self._pool = ThreadPoolExecutor(max_workers=POOL_WORKERS, thread_name_prefix="semantic")
        self._batcher = EmbedBatcher(embedder.encode, self._pool)

    # ---- helpers
    def _index_for(self, ns: str):
        idx = self._indexes.get(ns)
        if idx is None:
            # Inner product of unit vectors equals cosine similarity.
            idx = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.embedder.dim))
            self._indexes[ns] = idx
        return idx

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def size(self) -> int:
        with self._lock:
            return sum(i.ntotal for i in self._indexes.values())

    def _search(self, ns: str, vec: np.ndarray) -> tuple[float, int] | None:
        with self._lock:
            idx = self._indexes.get(ns)
            if idx is None or idx.ntotal == 0:
                return None
            sims, ids = idx.search(vec.reshape(1, -1), 1)
        return float(sims[0][0]), int(ids[0][0])

    def _add(self, ns: str, vec: np.ndarray, entry_id: int) -> None:
        with self._lock:
            self._index_for(ns).add_with_ids(vec.reshape(1, -1), np.array([entry_id], dtype=np.int64))

    def _remove(self, ns: str, entry_id: int) -> None:
        with self._lock:
            idx = self._indexes.get(ns)
            if idx is not None:
                idx.remove_ids(np.array([entry_id], dtype=np.int64))

    # ---- public API
    async def embed(self, body: dict) -> SemanticQuery | None:
        parts = build_query_parts(body)
        if parts is None:
            return None
        ns, text = parts
        # Embedding is CPU work (~5-15 ms). The batcher runs it on the worker pool, keeping the
        # event loop free, and merges requests that arrive together into one encode() call.
        vec = await self._batcher.embed(text)
        return SemanticQuery(ns, text, vec)

    async def lookup(self, q: SemanticQuery) -> SemanticResult:
        loop = asyncio.get_running_loop()
        found = await loop.run_in_executor(self._pool, self._search, q.namespace, q.vector)
        if found is None:
            return SemanticResult(None, None)
        sim, entry_id = found
        if sim < self.threshold:
            return SemanticResult(None, sim)
        try:
            raw = await self.redis.get(f"{ENTRY_PREFIX}{entry_id}") if self.redis is not None else None
        except (RedisError, OSError) as exc:
            log.warning("semantic answer fetch failed, treating as miss: %s", exc)
            return SemanticResult(None, sim)
        if raw is None:  # the answer expired (TTL) -> drop its vector too
            self._remove(q.namespace, entry_id)
            return SemanticResult(None, sim)
        entry = json.loads(raw)
        # If Redis lost data, the id counter restarts and an old FAISS id could point at a new,
        # unrelated entry. The entry is trusted only if its namespace and vector match the query.
        stored = np.frombuffer(base64.b64decode(entry["vec"]), dtype=np.float32)
        if entry.get("ns") != q.namespace or stored.shape != q.vector.shape \
                or float(stored @ q.vector) < self.threshold:
            self._remove(q.namespace, entry_id)
            return SemanticResult(None, sim)
        if self.verifier is None:
            return SemanticResult(entry["response"], sim, entry.get("text"))
        # Stage 2: the cross-encoder scores (new question, cached question) as a pair.
        try:
            score = (await loop.run_in_executor(
                self._pool, self.verifier.score, [(q.text, entry.get("text") or "")]))[0]
        except Exception as exc:  # a broken verifier must never serve an unchecked answer
            log.warning("semantic verify failed, treating as miss: %s", exc)
            return SemanticResult(None, sim)
        if score < self.verify_threshold:
            return SemanticResult(None, sim, entry.get("text"), score)
        return SemanticResult(entry["response"], sim, entry.get("text"), score)

    async def store(self, q: SemanticQuery, response: dict) -> None:
        if self.redis is None:
            return
        try:
            entry_id = int(await self.redis.incr(ID_COUNTER_KEY))  # unique across replicas
            entry = {
                "ns": q.namespace,
                "text": q.text,
                "vec": base64.b64encode(q.vector.astype(np.float32).tobytes()).decode(),
                "response": response,
            }
            await self.redis.set(f"{ENTRY_PREFIX}{entry_id}", json.dumps(entry), ex=self.ttl_s)
        except (RedisError, OSError) as exc:
            # No answer was stored, so the vector is not indexed either.
            log.warning("semantic store failed, skipping: %s", exc)
            return
        self._add(q.namespace, q.vector, entry_id)

    async def warm_start(self) -> int:
        """Rebuild FAISS from Redis after a restart. Returns how many vectors were loaded."""
        if self.redis is None:
            return 0
        loaded = 0
        try:
            async for key in self.redis.scan_iter(match=f"{ENTRY_PREFIX}*", count=500):
                raw = await self.redis.get(key)
                if raw is None:
                    continue
                entry = json.loads(raw)
                vec = np.frombuffer(base64.b64decode(entry["vec"]), dtype=np.float32)
                if vec.shape[0] != self.embedder.dim:
                    continue
                k = key.decode() if isinstance(key, bytes) else key
                self._add(entry["ns"], vec, int(k[len(ENTRY_PREFIX):]))
                loaded += 1
        except (RedisError, OSError) as exc:
            log.warning("semantic warm start skipped: %s", exc)
        return loaded
