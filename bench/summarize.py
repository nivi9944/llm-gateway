"""Read every results/*.json, print a summary, fill the README results table, and draft resume
bullets. Every number comes straight from the JSON files, so nothing is typed by hand.

Run:   python bench/summarize.py            (prints + updates README.md)
       python bench/summarize.py --dir trial_results --no-readme
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load(d: Path, name: str) -> dict | None:
    f = d / name
    return json.loads(f.read_text()) if f.exists() else None


def fmt_pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def build(d: Path) -> tuple[list[tuple[str, str, str]], dict]:
    rows, vals = [], {}
    q = load(d, "qqp_threshold.json")
    if q:
        c = q["chosen"]
        vals.update(threshold=c["threshold"], precision=c["precision"], recall=c["recall"],
                    false_hit=c["false_hit_rate"], qqp_pairs=q.get("sampled_pairs"))
        note = "" if c["target_met"] else " (95% target NOT met)"
        rows += [
            ("Semantic threshold (chosen)", f"{c['threshold']:.2f}{note}", "qqp_threshold.json"),
            ("Hit precision at threshold", fmt_pct(c["precision"]), "qqp_threshold.json"),
            ("Paraphrase recall at threshold", fmt_pct(c["recall"]), "qqp_threshold.json"),
            ("False-hit rate (different-meaning pairs matched)", fmt_pct(c["false_hit_rate"], 2),
             "qqp_threshold.json"),
            ("QQP pairs evaluated", f"{q.get('sampled_pairs', 0):,}", "qqp_threshold.json"),
        ]
    r = load(d, "replay.json")
    if r:
        vals.update(hit_rate=r["hit_rate"], saved=r["est_cost_saved_pct"], replay_n=r["requests"],
                    sem_prec=r["semantic_hit_precision"], mix=r["realised_mix"])
        mix = r["realised_mix"]
        rows += [
            ("Replay: overall cache hit rate", fmt_pct(r["hit_rate"]), "replay.json"),
            ("Replay: exact / semantic hit rate",
             f"{fmt_pct(r['exact_hit_rate'])} / {fmt_pct(r['semantic_hit_rate'])}", "replay.json"),
            ("Replay: semantic hits that were correct", fmt_pct(r["semantic_hit_precision"]), "replay.json"),
            ("Replay: all cache hits that were correct", fmt_pct(r.get("all_hit_precision")), "replay.json"),
            ("Replay: estimated LLM spend saved", f"{r['est_cost_saved_pct']:.1f}%", "replay.json"),
            ("Replay workload mix (unique / near-miss / exact / paraphrase)",
             " / ".join(f"{100 * mix[k]:.0f}%" for k in ("unique", "nearmiss", "exact", "paraphrase")),
             "replay.json"),
        ]
    ld = load(d, "load.json")
    if ld:
        g, o = ld["through_gateway"], ld["gateway_overhead_ms"]
        vals.update(rps=g["rps"], p95_overhead=o["p95"], p95_internal=ld["gateway_internal_overhead_ms"]["p95"],
                    users=ld["setup"]["users"])
        rows += [
            (f"Load: throughput ({ld['setup']['users']} users, {ld['setup']['mock_latency_ms']:.0f} ms provider)",
             f"{g['rps']:.1f} req/s", "load.json"),
            ("Load: end-to-end p50 / p95 / p99", f"{g['p50_ms']:.0f} / {g['p95_ms']:.0f} / {g['p99_ms']:.0f} ms",
             "load.json"),
            ("Load: gateway overhead p50 / p95 / p99 (vs direct)",
             f"{o['p50']:.1f} / {o['p95']:.1f} / {o['p99']:.1f} ms", "load.json"),
            ("Load: in-gateway time p95 (log: latency - upstream)",
             f"{ld['gateway_internal_overhead_ms']['p95']:.1f} ms", "load.json"),
            ("Load: success rate", f"{g['success_pct']:.2f}%", "load.json"),
        ]
    ch = load(d, "chaos.json")
    if ch:
        h, b = ch["headline"], ch["breaker_effect_during_outage"]
        vals.update(success_with=h["success_pct_with_protection"], success_without=h["success_pct_without_protection"],
                    fault_rate=h["primary_failure_rate"])
        rows += [
            (f"Chaos ({100 * h['primary_failure_rate']:.0f}% failures): success without / with protection",
             f"{h['success_pct_without_protection']:.1f}% / {h['success_pct_with_protection']:.1f}%", "chaos.json"),
            ("Chaos (outage): calls to dead provider, no breaker / breaker",
             f"{b['primary_calls_without_breaker']:,} / {b['primary_calls_with_breaker']:,}", "chaos.json"),
            ("Chaos (outage): p95 latency, no breaker / breaker",
             f"{b['p95_ms_without_breaker']:.0f} / {b['p95_ms_with_breaker']:.0f} ms", "chaos.json"),
        ]
    t = load(d, "tests.json")
    if t:
        vals.update(tests=t["passed"])
        rows.append(("Automated tests passing", f"{t['passed']} of {t['total']}", "tests.json"))
    return rows, vals


def machine_line(d: Path) -> str:
    for name in ("load.json", "chaos.json", "replay.json", "qqp_threshold.json"):
        j = load(d, name)
        if j:
            m = j["machine"]
            return (f"Measured on: {m.get('platform')}, {m.get('cpu_count')} logical CPUs, "
                    f"{m.get('ram_gb', '?')} GB RAM, Python {m.get('python')}, {j['generated_at']}.")
    return ""


def bullets(v: dict) -> list[str]:
    out = []
    if {"saved", "hit_rate"} <= v.keys():
        out.append(f"Built an async LLM gateway with exact + semantic caching, cutting estimated LLM spend "
                   f"**{v['saved']:.0f}%** at **{100 * v['hit_rate']:.0f}%** hit rate.")
    if {"precision", "false_hit"} <= v.keys() and v.get("qqp_pairs"):
        out.append(f"Tuned the semantic-cache threshold on **{v['qqp_pairs'] // 1000}K** Quora question pairs to "
                   f"**{math.floor(100 * v['precision'])}%** hit precision, under "
                   f"**{math.ceil(1000 * v['false_hit']) / 10:.1f}%** false hits.")
    if {"success_with", "fault_rate"} <= v.keys():
        sw = f"{v['success_with']:.1f}".rstrip("0").rstrip(".")
        out.append("Added atomic Redis + Lua rate limiting, retries, fallback and a circuit breaker, keeping "
                   f"**{sw}%** success at {100 * v['fault_rate']:.0f}% faults.")
    if {"rps", "p95_overhead", "tests", "users"} <= v.keys():
        # RPS here is capped by the test design (users / provider latency), so the bullet leads with
        # the overhead number, which is what the load test really measures.
        out.append(f"Kept p95 gateway overhead at **{v['p95_overhead']:.0f} ms** under {v['users']} concurrent users "
                   f"(**{v['rps']:.0f}** req/s), verified by Locust and **{v['tests']}** pytest cases.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    d = ROOT / args.dir
    rows, vals = build(d)
    if not rows:
        print(f"no result files in {d}")
        return
    table = ["| Metric | Value | Source |", "|---|---|---|"]
    table += [f"| {a} | {b} | `{args.dir}/{c}` |" for a, b, c in rows]
    block = "\n".join([START, "", machine_line(d), "", *table, "", END])
    print(block)
    print("\nDraft resume bullets (check length and wording before using):")
    for b in bullets(vals):
        print(f"- {b}  ({len(b.replace('**', ''))} chars)")
    if not args.no_readme and args.dir == "results":
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
        print("\nREADME.md results table updated.")


if __name__ == "__main__":
    main()
