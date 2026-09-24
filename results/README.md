# results/

The only source of numbers quoted in this repository. Each file is written by a script in `bench/`
and records the machine it ran on, the time, and the exact settings used.
`python bench/summarize.py` copies them into the main README.

## Current configuration

| File | Written by |
|---|---|
| `qqp_threshold.json`, `qqp_threshold.png` | `bench/qqp_threshold.py --two-stage` |
| `replay.json` | `bench/replay.py` |
| `load.json` | `bench/load.py` |
| `chaos.json` | `bench/chaos.py` |
| `smoke_live.json` | `bench/smoke_live.py` (live Gemini / Ollama check) |
| `tests.json` | `bench/test_report.py` |

## Comparison runs

| File | Configuration |
|---|---|
| `qqp_threshold_single_stage.json` | semantic cache with MiniLM alone (`bench/qqp_threshold.py`) |
| `replay_single_stage.json` | replay with MiniLM alone (`bench/replay.py --no-verify`) |
| `load_baseline.json` | load test, embedding on asyncio's default thread pool, torch using all cores |
| `load_threadpool_only.json` | + dedicated 4-thread pool, 1 torch thread each |
| `load_trial_8x2.json` | alternative tried: 8 workers x 2 torch threads (slower, not used) |
| `load_threadpool_batching.json` | + micro-batched embeddings (MiniLM-only semantic cache) |
| `load_optimized.json` | + cross-encoder check: the final configuration (same run as `load.json`) |
