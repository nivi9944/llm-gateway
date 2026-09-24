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
                    false_hit=c["false_hit_rate"], qqp_pairs=q.get("sampled_pairs"),
                    two_stage=q.get("mode") == "two_stage")
        note = "" if c["target_met"] else f" ({q['target_precision']:.0%} target NOT met)"
        if q.get("mode") == "two_stage":
            rows.append(("Semantic cache: MiniLM candidate / cross-encoder verify threshold",
                         f"{c['candidate_threshold']:.2f} / {c['verify_threshold']:.2f}{note}", "qqp_threshold.json"))
        else:
            rows.append(("Semantic threshold (chosen)", f"{c['threshold']:.2f}{note}", "qqp_threshold.json"))
        rows += [
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
            ("Replay: paraphrases served from cache", fmt_pct(r.get("paraphrase_catch_rate")), "replay.json"),
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


# Version history: which result file holds each version's numbers. The cache logic did not change
# between v0 and v2 (only speed work), so those versions share the single-stage cache runs.
VERSIONS = [
    ("v0 original", {"load": "load_v0_original.json", "qqp": "qqp_threshold_v2_single_stage.json",
                     "replay": "replay_v2_single_stage.json"}),
    ("v1 thread pool", {"load": "load_v1_threadpool.json", "qqp": "qqp_threshold_v2_single_stage.json",
                        "replay": "replay_v2_single_stage.json"}),
    ("v2 batching", {"load": "load_v2_batching.json", "qqp": "qqp_threshold_v2_single_stage.json",
                     "replay": "replay_v2_single_stage.json"}),
    ("v3 two-stage", {"load": "load_v3_two_stage.json", "qqp": "qqp_threshold.json", "replay": "replay.json"}),
]
VERSION_METRICS = [
    ("QQP hit precision", "qqp", lambda j: fmt_pct(j["chosen"]["precision"])),
    ("QQP paraphrase recall", "qqp", lambda j: fmt_pct(j["chosen"]["recall"])),
    ("QQP false-hit rate", "qqp", lambda j: fmt_pct(j["chosen"]["false_hit_rate"], 2)),
    ("Replay hit rate", "replay", lambda j: fmt_pct(j["hit_rate"])),
    ("Replay paraphrases served from cache", "replay", lambda j: fmt_pct(j["paraphrase_catch_rate"])),
    ("Replay semantic hits correct", "replay", lambda j: fmt_pct(j["semantic_hit_precision"])),
    ("Replay all hits correct", "replay", lambda j: fmt_pct(j["all_hit_precision"])),
    ("Replay est. spend saved", "replay", lambda j: f"{j['est_cost_saved_pct']:.1f}%"),
    ("Load throughput", "load", lambda j: f"{j['through_gateway']['rps']:.1f} req/s"),
    ("Load overhead p50", "load", lambda j: f"{j['gateway_overhead_ms']['p50']:.1f} ms"),
    ("Load overhead p95", "load", lambda j: f"{j['gateway_overhead_ms']['p95']:.1f} ms"),
    ("Load in-gateway time p95", "load", lambda j: f"{j['gateway_internal_overhead_ms']['p95']:.1f} ms"),
]


def final_version(d: Path) -> str | None:
    ld = load(d, "load.json")
    if ld is None:
        return None
    return "v3 two-stage" if ld.get("semantic_cache", {}).get("verify_enabled") else "v2 batching"


def versions_table(d: Path) -> list[str]:
    files = {name: {k: load(d, f) for k, f in m.items()} for name, m in VERSIONS}
    names = [n for n, _ in VERSIONS if all(files[n].values())]
    if len(names) < 2:
        return []
    final = final_version(d)
    head = [f"{n} (final)" if n == final else n for n in names]
    out = ["| Metric | " + " | ".join(head) + " |", "|---|" + "---|" * len(names)]
    for label, kind, fn in VERSION_METRICS:
        out.append(f"| {label} | " + " | ".join(fn(files[n][kind]) for n in names) + " |")
    return out


def bullets(v: dict, d: Path | None = None) -> list[str]:
    out = []
    first = {k: load(d, f) for k, f in VERSIONS[0][1].items()} if d else {}
    if {"saved", "hit_rate"} <= v.keys():
        kind = "two-stage semantic" if v.get("two_stage") else "semantic"
        out.append(f"Built an async LLM gateway with exact + {kind} caching, cutting estimated LLM spend "
                   f"**{v['saved']:.1f}%** at **{100 * v['hit_rate']:.1f}%** hit rate.")
    if v.get("two_stage") and first.get("qqp"):
        r0, r1 = first["qqp"]["chosen"]["recall"], v["recall"]
        out.append(f"Added a cross-encoder check to the semantic cache, lifting paraphrase recall "
                   f"**{r1 / r0:.1f}x** (**{100 * r0:.1f}%** to **{100 * r1:.1f}%**) at "
                   f"**{math.floor(1000 * v['precision']) / 10:.1f}%** precision.")
    elif {"precision", "false_hit"} <= v.keys() and v.get("qqp_pairs"):
        out.append(f"Tuned the semantic-cache threshold on **{v['qqp_pairs'] // 1000}K** Quora question pairs to "
                   f"**{math.floor(100 * v['precision'])}%** hit precision, under "
                   f"**{math.ceil(1000 * v['false_hit']) / 10:.1f}%** false hits.")
    if {"success_with", "fault_rate"} <= v.keys():
        sw = f"{v['success_with']:.1f}".rstrip("0").rstrip(".")
        out.append("Added atomic Redis + Lua rate limiting, retries, fallback and a circuit breaker, keeping "
                   f"**{sw}%** success at {100 * v['fault_rate']:.0f}% faults.")
    if first.get("load") and {"p95_overhead", "users"} <= v.keys():
        o0, o1 = first["load"]["gateway_overhead_ms"]["p95"], v["p95_overhead"]
        out.append(f"Cut p95 gateway overhead **{math.floor(100 * (1 - o1 / o0))}%** (**{o0:.0f}** to **{o1:.0f} ms**) "
                   f"at {v['users']} concurrent users with a thread pool and micro-batching.")
    elif {"rps", "p95_overhead", "tests", "users"} <= v.keys():
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
    history = versions_table(d)
    if history:
        history = ["", "Version history (each column is read from its own results file):", "", *history]
    block = "\n".join([START, "", machine_line(d), "", *table, *history, "", END])
    print(block)
    print("\nDraft resume bullets (check length and wording before using):")
    for b in bullets(vals, d):
        print(f"- {b}  ({len(b.replace('**', ''))} chars)")
    if not args.no_readme and args.dir == "results":
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
        print("\nREADME.md results table updated.")


if __name__ == "__main__":
    main()
