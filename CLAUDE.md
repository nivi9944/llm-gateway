# CLAUDE.md

Guide for Claude Code working in this repo.

## What this is
An OpenAI-compatible LLM gateway: auth → token-bucket rate limit (Redis Lua) → exact cache (Redis)
→ semantic cache (MiniLM + FAISS) → router (Gemini → Ollama, retries with full jitter, circuit
breaker) → cache store → JSON log. Read `README.md` for the flow and `DECISIONS.md` for the why.

## Owner context
Built by a student preparing for placements who is new to backend engineering. Explain in plain,
beginner-level language. He must be able to explain every line in an interview. Prefer doing over
describing. Windows laptop, PowerShell commands.

## Commands
```powershell
.\.venv\Scripts\Activate.ps1
pytest -q                                   # offline, ~10 s
docker compose up -d --build                # gateway :8000, redis :6379, mock :9000
python bench/<name>.py --help               # benchmarks write results/<name>.json
python bench/summarize.py                   # results -> README table + draft resume bullets
```

## Rules
* **Never invent, estimate or hand-round a number.** Every number in README/resume must trace to
  `results/*.json`. Use `bench/summarize.py`.
* **Never print or read aloud `.env`** (holds the Gemini key). Edit single lines with PowerShell `-replace`.
* Only public data through Gemini (free tier may use prompts for training).
* Keep tests offline: fake providers via `httpx.MockTransport`, `fakeredis`, `HashEmbedder`.
* Every new behaviour gets a test. Keep `pytest -q` green.
* **No em dashes (U+2014) anywhere:** files, code comments, docstrings, commit messages, GitHub text.
  Use a comma, a colon, parentheses, or a new sentence instead.

## Map
| File | Role |
|---|---|
| `gateway/main.py` | app factory `create_app()`, request flow, headers |
| `gateway/config.py` | all settings (env / .env) |
| `gateway/ratelimit.py` | token bucket Lua script |
| `gateway/cache.py` | exact cache + message normalisation |
| `gateway/semantic_cache.py` | embedders, FAISS index per namespace, Redis answers, warm start |
| `gateway/router.py` | providers, retry/backoff, fallback |
| `gateway/breaker.py` | circuit breaker state machine |
| `gateway/metrics.py` | JSON log line, cost estimate, `/metrics-summary` |
| `mock_provider/app.py` | fake LLM; `/admin/config` changes latency/failure live |
| `bench/` | benchmarks 1–4, `smoke_live.py`, `test_report.py`, `summarize.py` |
| `tests/conftest.py` | `gateway()` helper that builds an app with fakes |

## Status
See `PLAN.md` and `HANDOVER.md` (next steps).
