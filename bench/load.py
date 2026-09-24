"""Benchmark 3: load test. How much latency does the gateway ADD, and how many req/s?

Setup: mock provider with 800 ms latency. Locust (headless) runs 50 users for 60 s,
every request unique (cache MISS = the full, most expensive path).

Two runs with identical settings:
  A. Locust -> mock directly          (baseline: what the app would see with no gateway)
  B. Locust -> gateway -> mock        (with the gateway in the middle)
Gateway overhead = B minus A, at p50 / p95 / p99.

A second, independent measure comes from the gateway's own log: for each request,
latency_ms (time inside the gateway) minus upstream_ms (time waiting on the provider).

Run:   python bench/load.py                 (real: MiniLM embedder)
       python bench/load.py --embedder hash (offline trial; writes to trial_results/)
Needs Redis at BENCH_REDIS_URL (default redis://localhost:6379/1).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._common import (BENCH_KEY, BENCH_REDIS_URL, ROOT, flush_bench_redis, pct, save_json,  # noqa: E402
                           start_mock, start_server, stop)


def run_locust(host: str, users: int, seconds: int, spawn_rate: float, prefix: str) -> dict:
    # Users start gradually (spawn_rate per second). Starting all 50 at the same instant makes
    # them move in lock-step waves that hit the gateway as one burst every 800 ms: a test
    # artifact, not real traffic. The ramp-up time is excluded from the stats.
    ramp = int(users / spawn_rate) + 1
    cmd = [sys.executable, "-m", "locust", "-f", str(ROOT / "bench" / "locustfile.py"), "--headless",
           "-u", str(users), "-r", str(spawn_rate), "-t", f"{seconds + ramp}s", "--host", host,
           "--reset-stats", "--csv", prefix, "--only-summary", "--loglevel", "WARNING"]
    raw = f"{prefix}_raw.txt"
    env = {**os.environ, "GATEWAY_KEY": BENCH_KEY, "LOCUST_RAW_OUT": raw}
    print(f"  locust -> {host} ({users} users, {seconds}s)")
    subprocess.run(cmd, cwd=ROOT, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(f"{prefix}_stats.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    n = int(row["Request Count"])
    fails = int(row["Failure Count"])
    times = [float(x) for x in Path(raw).read_text().split()] if Path(raw).exists() else []
    return {
        "requests": n,
        "requests_recorded_exact": len(times),
        "failures": fails,
        "success_pct": round(100 * (n - fails) / n, 2) if n else 0.0,
        "rps": round(float(row["Requests/s"]), 2),
        "p50_ms": pct(times, 50),
        "p95_ms": pct(times, 95),
        "p99_ms": pct(times, 99),
        "avg_ms": round(sum(times) / len(times), 1) if times else None,
        "percentile_source": f"exact per-request times ({len(times)} successful requests)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--spawn-rate", type=float, default=5, help="users started per second")
    ap.add_argument("--latency-ms", type=float, default=800)
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "hash"])
    ap.add_argument("--mock-port", type=int, default=9120)
    ap.add_argument("--gateway-port", type=int, default=8120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    trial = args.embedder == "hash"
    out = Path(args.out or ("trial_results" if trial else "results"))

    flush_bench_redis()
    tmp = Path(tempfile.mkdtemp())
    gw_log = tmp / "gateway_requests.jsonl"
    mock = start_mock(args.mock_port, latency_ms=args.latency_ms, name="load-mock")
    gw = None
    try:
        gw = start_server("gateway.main:app", args.gateway_port, {
            "GATEWAY_KEYS": BENCH_KEY, "REDIS_URL": BENCH_REDIS_URL, "PROVIDER_ORDER": "mock",
            "MOCK_BASE_URL": f"http://127.0.0.1:{args.mock_port}/v1", "EMBEDDER": args.embedder,
            "BUCKET_CAPACITY": 1_000_000_000, "REFILL_PER_SEC": 1_000_000,  # limiter ON but never blocks
            "LOG_FILE": str(gw_log),
        }, "load-gateway")
        direct = run_locust(f"http://127.0.0.1:{args.mock_port}", args.users, args.seconds, args.spawn_rate,
                            str(tmp / "direct"))
        via = run_locust(f"http://127.0.0.1:{args.gateway_port}", args.users, args.seconds, args.spawn_rate,
                         str(tmp / "gateway"))
    finally:
        stop(gw, mock)

    internal = []
    cache_counts: dict[str, int] = {}
    for line in gw_log.read_text().splitlines():
        e = json.loads(line)
        cache_counts[e.get("cache")] = cache_counts.get(e.get("cache"), 0) + 1
        if e.get("status") == 200 and e.get("upstream_ms") is not None:
            internal.append(e["latency_ms"] - e["upstream_ms"])

    result = {
        "benchmark": "load",
        "trial_only_not_for_resume": trial,
        "setup": {"tool": "locust (headless)", "users": args.users, "seconds": args.seconds,
                  "spawn_rate_per_s": args.spawn_rate, "ramp_up_excluded": True,
                  "mock_latency_ms": args.latency_ms, "embedder": args.embedder,
                  "cache": "every request unique -> all MISS (full path)",
                  "gateway_workers": 1, "rate_limiter": "on (limit set high)"},
        "direct_to_mock": direct,
        "through_gateway": via,
        "gateway_overhead_ms": {
            "p50": round(via["p50_ms"] - direct["p50_ms"], 1),
            "p95": round(via["p95_ms"] - direct["p95_ms"], 1),
            "p99": round(via["p99_ms"] - direct["p99_ms"], 1),
            "method": "through_gateway percentile minus direct_to_mock percentile (same load)",
        },
        "gateway_cache_status_counts": cache_counts,
        "gateway_internal_overhead_ms": {
            "requests": len(internal),
            "p50": pct(internal, 50), "p95": pct(internal, 95), "p99": pct(internal, 99),
            "method": "per request, from the gateway log: latency_ms - upstream_ms",
        },
    }
    save_json(out, "load.json", result)
    print(json.dumps({"rps": via["rps"], "p95_ms": via["p95_ms"],
                      "overhead": result["gateway_overhead_ms"],
                      "internal": result["gateway_internal_overhead_ms"]}, indent=2))


if __name__ == "__main__":
    main()
