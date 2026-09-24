# results/

The **only** place numbers come from. Each file is written by a script in `bench/` and includes
the machine it ran on, the time, and the exact settings used.

| File | Written by |
|---|---|
| `tests.json` | `bench/test_report.py` |
| `qqp_threshold.json`, `qqp_threshold.png` | `bench/qqp_threshold.py --two-stage` (current, v3) |
| `replay.json` | `bench/replay.py` (current, v3) |
| `load.json` | `bench/load.py` (current, v3) |
| `chaos.json` | `bench/chaos.py` |
| `smoke_live.json` | `bench/smoke_live.py` |

Earlier versions, kept unchanged so improvements can be compared file to file:

| File | Version |
|---|---|
| `load_v0_original.json` | v0: original code, before any speed work |
| `load_v1_threadpool.json` | v1: own 4-thread pool, 1 torch thread each |
| `load_v1_trial_8workers_2threads.json` | v1 trial: 8 workers x 2 threads (slower, not used) |
| `load_v2_batching.json` | v2: v1 + micro-batched embeddings |
| `load_v3_two_stage.json` | v3: v2 + cross-encoder verify (same run as `load.json`) |
| `qqp_threshold_v2_single_stage.json` | v0 to v2: one-stage cache (MiniLM alone) |
| `replay_v2_single_stage.json` | v0 to v2: one-stage cache (MiniLM alone) |
