"""Helpers shared by the benchmark scripts: start/stop servers, record machine info, save JSON."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BENCH_REDIS_URL = os.getenv("BENCH_REDIS_URL", "redis://localhost:6379/1")  # db 1: never your dev cache
BENCH_KEY = "bench-key"


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info


def save_json(out_dir: str | Path, name: str, data: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "machine": machine_info(), **data}
    path = out / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return path


def flush_bench_redis(url: str = BENCH_REDIS_URL) -> None:
    import redis

    r = redis.Redis.from_url(url)
    r.ping()
    r.flushdb()


def wait_healthy(url: str, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"{url} did not become healthy in {timeout_s}s")


def start_server(app: str, port: int, env: dict[str, str], log_name: str) -> subprocess.Popen:
    """Start `uvicorn <app>` as a child process and wait until /health answers."""
    full_env = {**os.environ, **{k: str(v) for k, v in env.items()}}
    log_dir = ROOT / "bench" / "logs"
    log_dir.mkdir(exist_ok=True)
    log = open(log_dir / f"{log_name}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning", "--no-access-log"],
        cwd=ROOT, env=full_env, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        wait_healthy(f"http://127.0.0.1:{port}/health")
    except Exception:
        proc.terminate()
        raise
    return proc


def start_mock(port: int, latency_ms: float, fail_rate: float = 0.0, seed: int | None = 42,
               name: str = "mock") -> subprocess.Popen:
    env = {"MOCK_LATENCY_MS": latency_ms, "MOCK_FAIL_RATE": fail_rate}
    if seed is not None:
        env["MOCK_SEED"] = seed
    return start_server("mock_provider.app:app", port, env, name)


def stop(*procs: subprocess.Popen) -> None:
    for p in procs:
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def pct(values, q: float) -> float:
    import numpy as np

    return round(float(np.percentile(values, q)), 2) if len(values) else 0.0
