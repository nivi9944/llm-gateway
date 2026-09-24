"""Live smoke test against a RUNNING gateway (docker compose up -d first).

  1. a normal request                      -> should go to Gemini (X-Provider: gemini)
  2. the same kind of request forced onto Ollama (X-Provider-Force: ollama)
  3. request 1 again                       -> should be X-Cache: HIT-EXACT
  4. a paraphrase of request 1             -> hopefully X-Cache: HIT-SEMANTIC (shows similarity)

Run:   python bench/smoke_live.py --url http://localhost:8000 --key dev-key-1
Only public, harmless questions are sent (the Gemini free tier may use prompts for training).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._common import save_json  # noqa: E402

Q1 = "In one sentence, what is a token bucket rate limiter?"
Q1_PARAPHRASE = "In one sentence, what does a token-bucket rate limiter do?"
Q2 = "In one sentence, what is exponential backoff?"


def ask(client, url, key, question, force=None):
    headers = {"Authorization": f"Bearer {key}"}
    if force:
        headers["X-Provider-Force"] = force
    body = {"model": "auto", "temperature": 0, "messages": [{"role": "user", "content": question}]}
    r = client.post(f"{url}/v1/chat/completions", json=body, headers=headers)
    info = {"question": question, "forced": force, "status": r.status_code,
            **{h: r.headers.get(h) for h in ("X-Cache", "X-Provider", "X-Similarity", "X-Latency-Ms")}}
    if r.status_code == 200:
        info["answer"] = r.json()["choices"][0]["message"]["content"][:200]
    else:
        info["error"] = r.text[:300]
    print(f"- {info['X-Cache']:<13} provider={info['X-Provider']:<7} "
          f"sim={info['X-Similarity']}  {info['X-Latency-Ms']} ms  status={r.status_code}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="dev-key-1")
    args = ap.parse_args()
    # A per-run tag keeps questions new, so a rerun doesn't just hit the cache from last time.
    tag = f"[check {random.randint(1000, 9999)}] "
    q1, q1p, q2 = tag + Q1, tag + Q1_PARAPHRASE, tag + Q2
    with httpx.Client(timeout=120) as c:
        print("health:", c.get(f"{args.url}/health").json())
        steps = [
            ask(c, args.url, args.key, q1),
            ask(c, args.url, args.key, q2, force="ollama"),
            ask(c, args.url, args.key, q1),
            ask(c, args.url, args.key, q1p),
        ]
        summary = c.get(f"{args.url}/metrics-summary", headers={"Authorization": f"Bearer {args.key}"}).json()
    checks = {
        "gemini_answered": steps[0]["X-Provider"] == "gemini",
        "ollama_answered_when_forced": steps[1]["X-Provider"] == "ollama",
        "repeat_was_exact_hit": steps[2]["X-Cache"] == "HIT-EXACT",
        "paraphrase_was_semantic_hit": steps[3]["X-Cache"] == "HIT-SEMANTIC",
    }
    print(checks)
    save_json("results", "smoke_live.json", {"benchmark": "smoke_live", "steps": steps, "checks": checks,
                                             "metrics_summary": summary})


if __name__ == "__main__":
    main()
