"""Semantic cache: reuse an answer when a NEW question MEANS the same as an old one.

"How do I learn Python fast?" and "Quickest way to learn Python?" share few exact words,
so the exact cache misses. Here we turn the question into a vector (an "embedding") with
MiniLM. Similar meanings give vectors pointing in similar directions. FAISS finds the
closest stored vector; if cosine similarity >= SEMANTIC_THRESHOLD we reuse its answer.

Safety rules that keep wrong answers out:
  * only the LAST user message is embedded, and matches are only searched among requests
    with the same model and the same earlier conversation (the "namespace"), so a question
    under a different system prompt never matches;
  * the threshold is picked from labelled data (bench/qqp_threshold.py);
  * only temperature-0 requests are cached (checked in main.py), and entries expire (TTL).

Storage split: FAISS (in memory) holds vectors for fast search; Redis holds the answers
(and a copy of each vector, so a restarted gateway can rebuild its FAISS index).
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
# TORCH_THREADS cores. Many small parallel requests beat one request grabbing every core.
POOL_WORKERS = 4
TORCH_THREADS = 1


# ---------------------------------------------------------------- embedders
class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim) float32, rows unit-length
        ...


class MiniLMEmbedder:
    """The real embedder: sentence-transformers/all-MiniLM-L6-v2 (384-dim, CPU friendly)."""

    dim = 384

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer  # heavy import, done once

        # Without this, every encode() tries to use ALL cores; with several requests in
        # parallel the threads fight each other and latency explodes under load.
        torch.set_num_threads(TORCH_THREADS)

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """A tiny, dependency-free stand-in used by unit tests and offline trial runs.

    Bag-of-words with the "hashing trick": each word adds +-1 to one of 384 slots.
    It understands shared WORDS, not meaning, so it is NOT used for real numbers.
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


def build_query_parts(body: dict) -> tuple[str, str] | None:
    """Return (namespace, text) or None if this request can't use the semantic cache.

    namespace = hash of everything EXCEPT the last user message (model, parameters, tools,
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
    def __init__(self, redis, embedder: Embedder, threshold: float, ttl_s: int):
        import faiss  # imported here so the rest of the gateway loads without it

        faiss.omp_set_num_threads(1)  # single-query searches: OpenMP threads only add overhead
        self._faiss = faiss
        self.redis = redis
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s
        self._indexes: dict[str, object] = {}  # namespace -> faiss.IndexIDMap2(IndexFlatIP)
        self._lock = threading.Lock()  # FAISS indexes are not safe for concurrent writes
        # Own pool, so embedding can't starve (or be starved by) asyncio's shared default pool.
        self._pool = ThreadPoolExecutor(max_workers=POOL_WORKERS, thread_name_prefix="semantic")

    # ---- helpers
    def _index_for(self, ns: str):
        idx = self._indexes.get(ns)
        if idx is None:
            # Inner product on unit vectors == cosine similarity.
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
        # Embedding is CPU work (~5-15 ms). Running it on our worker threads keeps the async
        # event loop free to serve other requests meanwhile.
        loop = asyncio.get_running_loop()
        vec = (await loop.run_in_executor(self._pool, self.embedder.encode, [text]))[0]
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
        # Defence in depth: if Redis lost data, the id counter restarts and an old FAISS id could
        # point at a NEW, unrelated entry. Only trust the entry if it really is what we matched.
        stored = np.frombuffer(base64.b64decode(entry["vec"]), dtype=np.float32)
        if entry.get("ns") != q.namespace or stored.shape != q.vector.shape \
                or float(stored @ q.vector) < self.threshold:
            self._remove(q.namespace, entry_id)
            return SemanticResult(None, sim)
        return SemanticResult(entry["response"], sim, entry.get("text"))

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
            # No answer stored -> don't index the vector either.
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
