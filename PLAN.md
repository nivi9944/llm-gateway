# Plan and status

| Phase | Who | Status |
|---|---|---|
| A: Build | Claude (cloud) | **Done 2026-09-24.** Repo written, tests green (50 passed, 1 skipped: needs the real model), load + chaos trial-run in the cloud. Threshold + replay benchmarks and the Docker build could not run there (HuggingFace and Docker Hub blocked) and move to Phase C. |
| B: Setup | Nithin | Laptop tools, Gemini key, Ollama model, `gh auth login`, Claude Code |
| C: Run for real | Claude Code on the laptop | Follow `HANDOVER.md` steps 1–12 |
| D: Own it | Nithin + Claude | Walkthrough, mock interview, resume bullets from `results/` |

## Phase D walkthrough outline (~90 min)
1. Request flow end to end (`gateway/main.py`), with live requests and headers.
2. Exact cache vs semantic cache; the threshold plot (`results/qqp_threshold.png`).
3. Token bucket + Lua script; why atomic; 50-concurrent test.
4. Router: retry vs fallback vs breaker; live demo by stopping Ollama / setting a bad Gemini key.
5. How each number was measured (open each `results/*.json`).
6. 10-question mock interview, scored out of 5 each.

## Later
* Project 2 (KPI Root-Cause Agent) sends its LLM calls through this gateway →
  `results/agent_traffic.json` (benchmark 5).
* Possible extras: streaming, Redis vector search instead of per-process FAISS, Prometheus metrics.
