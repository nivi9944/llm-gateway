"""Benchmark 2: replay a 10,000-request workload and measure cache hit rate + cost saved.

The workload is built from QQP *train* pairs. Any pair containing a question that also appears in
the validation split (used to tune the threshold) is dropped, so tuning and testing never share text.
Each request is one of four kinds (default mix, change with --mix):

  unique      a question not seen before                         -> should MISS
  exact       an earlier question repeated word for word          -> should HIT-EXACT
  paraphrase  the human-labelled duplicate of an earlier question -> should HIT-SEMANTIC
  nearmiss    a question QQP labels as DIFFERENT from an earlier,
              similar-looking one ("hard negative")               -> should MISS

The hit rate depends on this mix, so the mix is saved in replay.json. Quote it together
with the numbers ("on a workload with 20% exact repeats and 20% paraphrases").

Correctness of every cache hit is checked: each provider answer has a unique id, and a hit is
"correct" only if the question that produced that answer and the new question mean the same thing
by QQP's labels (duplicate pairs are chained together, so if A=B and B=C then A=C).

The gateway runs in-process; the provider is the mock (real HTTP, 0 ms latency).
Cost = tokens x Gemini paid-tier list price (an estimate).

Run:   python bench/replay.py            (needs Redis at BENCH_REDIS_URL, default db 1)
       python bench/replay.py --smoke    (offline check, hash embedder, tiny data)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._common import BENCH_KEY, BENCH_REDIS_URL, flush_bench_redis, save_json, start_mock, stop  # noqa: E402

KINDS = ("unique", "nearmiss", "exact", "paraphrase")


def parse_mix(text: str) -> dict[str, float]:
    mix = {k: 0.0 for k in KINDS}
    for part in text.split(","):
        k, v = part.split("=")
        mix[k.strip()] = float(v)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


class Meaning:
    """Union-find over QQP duplicate pairs: two texts with the same root mean the same thing."""

    def __init__(self, dup_pairs):
        from gateway.cache import collapse_text

        self.norm = collapse_text
        self.parent: dict[str, str] = {}
        for a, b in dup_pairs:
            self._union(self.norm(a), self.norm(b))

    def _find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        self.parent[self._find(a)] = self._find(b)

    def of(self, text: str) -> str:
        return self._find(self.norm(text))


def load_pools(args):
    if args.smoke:
        from bench.smoke_data import PAIRS

        dup = [(a, b) for a, b, y in PAIRS if y == 1]
        non = [(a, b) for a, b, y in PAIRS if y == 0]
        return dup, non, Meaning(dup), {"dataset": "bench/smoke_data.py (smoke only)"}
    from datasets import load_dataset

    from gateway.cache import collapse_text

    val = load_dataset("nyu-mll/glue", "qqp", split="validation")
    val_questions = {collapse_text(q) for col in ("question1", "question2") for q in list(val[col]) if q}
    ds = load_dataset("nyu-mll/glue", "qqp", split="train")
    all_dup, dup, non, dropped = [], [], [], 0
    for q1, q2, y in zip(list(ds["question1"]), list(ds["question2"]), list(ds["label"])):
        if not q1 or not q2 or collapse_text(q1) == collapse_text(q2):
            continue
        if y == 1:
            all_dup.append((q1, q2))  # used for meaning-groups even if the pair is dropped below
        if collapse_text(q1) in val_questions or collapse_text(q2) in val_questions:
            dropped += 1
            continue
        (dup if y == 1 else non).append((q1, q2))
    return dup, non, Meaning(all_dup), {
        "dataset": "GLUE QQP (train split)", "dup_pairs_available": len(dup), "nondup_pairs_available": len(non),
        "pairs_dropped_for_overlap_with_validation": dropped}


def build_workload(n: int, mix: dict, dup, non, seed: int) -> list[dict]:
    """Returns a list of {text, kind}."""
    rng = random.Random(seed)
    rng.shuffle(dup)
    rng.shuffle(non)
    di = ni = 0
    sent: list[dict] = []
    open_dup: list[int] = []   # dup pairs whose first question was sent, partner not yet
    open_non: list[int] = []   # same for non-dup pairs
    kinds, weights = zip(*mix.items())
    for _ in range(n):
        kind = rng.choices(kinds, weights)[0]
        if kind == "exact" and sent:
            item = {"text": rng.choice(sent)["text"], "kind": "exact"}
        elif kind == "paraphrase" and open_dup:
            i = open_dup.pop(rng.randrange(len(open_dup)))
            item = {"text": dup[i][1], "kind": "paraphrase"}
        elif kind == "nearmiss" and open_non:
            j = open_non.pop(rng.randrange(len(open_non)))
            item = {"text": non[j][1], "kind": "nearmiss"}
        else:  # unique (also the fallback early on, when nothing can be repeated yet)
            if rng.random() < 0.5 and di < len(dup):
                item = {"text": dup[di][0], "kind": "unique"}
                open_dup.append(di)
                di += 1
            elif ni < len(non):
                item = {"text": non[ni][0], "kind": "unique"}
                open_non.append(ni)
                ni += 1
            else:  # smoke data can run out; recycle as exact repeats
                item = {"text": rng.choice(sent)["text"], "kind": "exact"}
        sent.append(item)
    return sent


async def run(args, workload, threshold, embedder_kind, meaning: Meaning, verify: dict | None = None):
    import httpx
    from asgi_lifespan import LifespanManager

    from gateway.config import Settings
    from gateway.main import create_app

    settings = Settings(
        _env_file=None, GATEWAY_KEYS=BENCH_KEY, REDIS_URL=BENCH_REDIS_URL, PROVIDER_ORDER="mock",
        MOCK_BASE_URL=f"http://127.0.0.1:{args.mock_port}/v1", EMBEDDER=embedder_kind,
        SEMANTIC_THRESHOLD=threshold, RATE_LIMIT_ENABLED=False,
        SEMANTIC_VERIFY_ENABLED=verify is not None,
        SEMANTIC_CANDIDATE_THRESHOLD=verify["candidate"] if verify else threshold,
        SEMANTIC_VERIFY_THRESHOLD=verify["verify"] if verify else 0.5,
    )
    logging.getLogger("gateway.requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app = create_app(settings)
    answer_meaning: dict[str, str] = {}  # provider answer id -> meaning of the question that produced it
    rows = []
    async with LifespanManager(app, startup_timeout=120) as mgr:  # model load > default 5 s
        c = app.state.c
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mgr.app), base_url="http://gw",
                                     timeout=60) as client:
            t0 = time.time()
            for i, item in enumerate(workload):
                body = {"model": "auto", "temperature": 0,
                        "messages": [{"role": "user", "content": item["text"]}]}
                r = await client.post("/v1/chat/completions", json=body,
                                      headers={"Authorization": f"Bearer {BENCH_KEY}"})
                cache = r.headers.get("X-Cache")
                data = r.json() if r.status_code == 200 else {}
                row = {"kind": item["kind"], "status": r.status_code, "cache": cache,
                       "cost": c.metrics.cost(data.get("usage")), "correct": None}
                if cache == "MISS" and r.status_code == 200:
                    answer_meaning[data["id"]] = meaning.of(item["text"])
                elif cache and cache.startswith("HIT"):
                    row["correct"] = answer_meaning.get(data.get("id")) == meaning.of(item["text"])
                rows.append(row)
                if (i + 1) % 500 == 0:
                    print(f"\r  {i + 1}/{len(workload)} requests ({time.time() - t0:.0f}s)", end="", flush=True)
            print()
    return rows, round(time.time() - t0, 1)


def summarise(rows) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["status"] == 200]
    exact = sum(r["cache"] == "HIT-EXACT" for r in rows)
    sem = [r for r in rows if r["cache"] == "HIT-SEMANTIC"]
    hits = [r for r in rows if (r["cache"] or "").startswith("HIT")]
    wrong_sem = sum(r["correct"] is False for r in sem)
    wrong = sum(r["correct"] is False for r in hits)
    cost_all = sum(r["cost"] for r in ok)
    cost_paid = sum(r["cost"] for r in ok if r["cache"] == "MISS")

    def kind_stats(kind):
        ks = [r for r in rows if r["kind"] == kind]
        return {"count": len(ks),
                "hit_exact": sum(r["cache"] == "HIT-EXACT" for r in ks),
                "hit_semantic": sum(r["cache"] == "HIT-SEMANTIC" for r in ks),
                "miss": sum(r["cache"] == "MISS" for r in ks)}

    return {
        "requests": n,
        "errors": n - len(ok),
        "hit_rate": round((exact + len(sem)) / n, 4),
        "exact_hit_rate": round(exact / n, 4),
        "semantic_hit_rate": round(len(sem) / n, 4),
        "semantic_hit_precision": round(1 - wrong_sem / len(sem), 4) if sem else None,
        "all_hit_precision": round(1 - wrong / len(hits), 4) if hits else None,
        "wrong_answers_served": wrong,
        "wrong_answer_rate_all_requests": round(wrong / n, 4),
        "paraphrase_catch_rate": _rate(rows, "paraphrase", "HIT-SEMANTIC"),
        "est_cost_without_cache_usd": round(cost_all, 4),
        "est_cost_with_cache_usd": round(cost_paid, 4),
        "est_cost_saved_pct": round(100 * (1 - cost_paid / cost_all), 2) if cost_all else 0.0,
        "by_kind": {k: kind_stats(k) for k in KINDS},
    }


def _rate(rows, kind, cache):
    ks = [r for r in rows if r["kind"] == kind]
    return round(sum(r["cache"] == cache for r in ks) / len(ks), 4) if ks else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--mix", default="unique=0.4,nearmiss=0.2,exact=0.2,paraphrase=0.2")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: chosen value from results/qqp_threshold.json")
    ap.add_argument("--no-verify", action="store_true", help="one-stage cache even if a two-stage result exists")
    ap.add_argument("--mock-port", type=int, default=9110)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ("trial_results" if args.smoke else "results"))
    if args.smoke:
        args.n = min(args.n, 300)

    threshold = args.threshold
    src = "--threshold"
    verify = None  # {"candidate": .., "verify": ..} when the two-stage cache is used
    if threshold is None:
        f = Path("results/qqp_threshold.json")
        if f.exists() and not args.smoke:
            chosen = json.loads(f.read_text())["chosen"]
            if "verify_threshold" in chosen and not args.no_verify:
                verify = {"candidate": chosen["candidate_threshold"], "verify": chosen["verify_threshold"]}
                threshold, src = chosen["candidate_threshold"], "results/qqp_threshold.json (two-stage)"
            elif "verify_threshold" in chosen:  # --no-verify: fall back to the one-stage result
                single = Path("results/qqp_threshold_v2_single_stage.json")
                threshold = json.loads(single.read_text())["chosen"]["threshold"]
                src = str(single).replace("\\", "/")
            else:
                threshold, src = chosen["threshold"], "results/qqp_threshold.json"
        else:
            threshold, src = 0.90, "default 0.90 (run qqp_threshold.py first)"
    mix = parse_mix(args.mix)
    dup, non, meaning, info = load_pools(args)
    workload = build_workload(args.n, mix, dup, non, args.seed)
    realised = {k: round(sum(w["kind"] == k for w in workload) / len(workload), 4) for k in KINDS}
    print(f"{len(workload)} requests, threshold {threshold} ({src}), verify {verify}, realised mix {realised}")

    flush_bench_redis()
    mock = start_mock(args.mock_port, latency_ms=0, name="replay-mock")
    try:
        rows, seconds = asyncio.run(run(args, workload, threshold, "hash" if args.smoke else "minilm", meaning,
                                        verify))
    finally:
        stop(mock)
    summary = summarise(rows)
    save_json(out, "replay.json", {
        "benchmark": "replay",
        "smoke_only_not_for_resume": bool(args.smoke),
        **info,
        "embedder": "hash (smoke)" if args.smoke else "sentence-transformers/all-MiniLM-L6-v2",
        "threshold": threshold, "threshold_source": src,
        "mode": "two_stage" if verify else "single_stage",
        "verify": {"candidate_threshold": verify["candidate"], "verify_threshold": verify["verify"],
                   "model": "cross-encoder/quora-distilroberta-base"} if verify else None,
        "requested_mix": mix, "realised_mix": realised, "seed": args.seed,
        "provider": "mock (0 ms)", "price_basis": "Gemini paid-tier list price from gateway config",
        "seconds": seconds,
        **summary,
    })
    print(json.dumps({k: summary[k] for k in ("hit_rate", "exact_hit_rate", "semantic_hit_rate",
                                              "semantic_hit_precision", "est_cost_saved_pct")}, indent=2))


if __name__ == "__main__":
    main()
