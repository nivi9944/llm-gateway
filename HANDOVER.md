# HANDOVER: Phase C (run on the laptop, then publish)

**For Claude Code:** follow the steps below in order on this Windows laptop (PowerShell).
Stop and ask the user before anything that costs money or touches files outside this folder.

## What Phase A (cloud) already did

* Wrote the whole repo: gateway, mock provider, benchmarks, tests, Docker, CI, docs.
* `pytest`: 50 passed, 1 skipped (the real-MiniLM test; the cloud could not download the model).
  Also passed against a real Redis.
* Trial runs of the load and chaos benchmarks on a 2-CPU cloud machine, using a stand-in embedder.
  Those numbers are **not** for the resume and are not in `results/`.

## What Phase A could NOT do (the cloud network blocks HuggingFace and Docker Hub)

1. Download the MiniLM model and the QQP dataset → **benchmarks 1 (threshold) and 2 (replay) have
   never run for real.** Their scripts passed an offline `--smoke` check. Run them here (steps 6–7).
2. Build the Docker image → **first real build happens here** (step 4).

## Hard rules

* **Never print, cat, or echo `.env`.** It holds the Gemini key. To change a setting in it, replace
  just that line (see step 6). The user pastes the key themselves.
* **Never invent or round numbers by hand.** Numbers come only from `results/*.json`, copied by
  `python bench/summarize.py`.
* Only public data goes through Gemini (the free tier may train on prompts).

## Step 0: prerequisites (the user did these in Phase B)

`git`, `py` / `python` 3.11+, Docker Desktop (running), `ollama` with `qwen2.5:7b` or `qwen2.5:3b`
pulled, `gh` logged in. Check quickly:

```powershell
git --version; python --version; docker version --format '{{.Server.Version}}'; ollama list; gh auth status
```

## Step 1: `.env` (USER does this)

```powershell
Copy-Item .env.example .env
notepad .env      # user pastes GEMINI_API_KEY; if the laptop has 8 GB RAM set OLLAMA_MODEL=qwen2.5:3b
```

Check the Gemini model name in AI Studio: it must be a Flash model listed as free for this key.

## Step 2: Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements-dev.txt   # Windows PyPI torch is CPU-only, which is what we want
```

## Step 3: tests (includes the real-MiniLM test, first run downloads ~90 MB)

```powershell
pytest -q
```

Expect **all passed, 0 skipped**. Fix any failure before continuing.

## Step 4: start the stack

```powershell
docker compose up -d --build          # first build takes a few minutes (torch + model download)
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

`/health` should show `redis: up` and providers `gemini` and `ollama` as `closed`.
If `gemini` is missing, the key is empty in `.env`. If port 8000 is busy, change the gateway port in
`docker-compose.yml` to `"8080:8000"` and use 8080 below.

## Step 5: tests against the real Redis from compose

```powershell
$env:TEST_REDIS_URL="redis://localhost:6379/2"; pytest -q; Remove-Item Env:TEST_REDIS_URL
python bench/test_report.py           # writes results/tests.json
```

## Step 6: benchmark 1, choose the threshold (~5–10 min, CPU)

```powershell
python bench/qqp_threshold.py
```

It prints the chosen threshold. Put it into `.env` **without printing the file** (replace 0.87 with
the printed value), then restart the gateway so it picks it up:

```powershell
(Get-Content .env) -replace '^SEMANTIC_THRESHOLD=.*', 'SEMANTIC_THRESHOLD=0.87' | Set-Content .env
docker compose up -d gateway
```

If it says **target NOT met**: tell the user. Options: (a) accept the best threshold and quote its
real precision, or (b) rerun with `--target 0.90`. Do not change the data or the method to force 95%.

## Step 7: benchmark 2, replay (~3–6 min)

```powershell
python bench/replay.py                # reads the threshold from results/qqp_threshold.json
```

## Step 8: live smoke test (Gemini, forced Ollama, cache hits)

```powershell
python bench/smoke_live.py --url http://localhost:8000 --key dev-key-1
```

Expect `gemini_answered`, `ollama_answered_when_forced` and `repeat_was_exact_hit` to be true.
`paraphrase_was_semantic_hit` depends on the threshold; report it either way.
If Gemini returns 429 (free-tier limit), wait a minute and retry.

## Step 9: benchmarks 3 and 4 on this laptop (~2.5 + ~1.5 min)

Close heavy apps first (browser tabs, games): the laptop's load affects the numbers.

```powershell
python bench/load.py
python bench/chaos.py
```

## Step 10: fill the README and draft bullets

```powershell
python bench/summarize.py
```

This updates the Results table in `README.md` and prints draft resume bullets with their lengths.
Show both to the user.

Also replace `<your-username>` in `README.md` with the user's GitHub username (ask if unknown:
`gh api user --jq .login`).

## Step 11: commit and publish (ASK the user before pushing)

```powershell
git init -b main
git add .
git status                            # make sure .env is NOT listed
git commit -m "LLM API Gateway: caching, rate limiting, fallback, benchmarks"
gh repo create llm-gateway --public --source . --push
```

Then check the Actions tab: the CI badge should go green.

## Step 12: final report

Print a table of every number in `results/*.json` (`python bench/summarize.py --no-readme` does this).

## Troubleshooting

| Problem | Fix |
|---|---|
| "WSL 2 installation is incomplete" | `wsl --update` |
| Virtualization disabled | enable in BIOS (F2 on ASUS) |
| Scripts disabled when activating venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `python` not found | use `py`, or reinstall Python with "Add to PATH" ticked |
| Container can't reach Ollama | Ollama must be running on Windows; URL `http://host.docker.internal:11434/v1` |
| Port 8000 busy | map the gateway to 8080 |
| Benchmarks: "Connection refused" to Redis | `docker compose up -d redis` |
| Benchmark port already in use | a previous run is still alive: `Get-Process python` and stop it, or pass `--mock-port` / `--gateway-port` |
| Gemini 429 | free-tier limit; wait, or lower request rate |
| Laptop slow / hot | switch to `qwen2.5:3b` |
| `faiss` import error on Windows | `pip install --force-reinstall faiss-cpu` |
