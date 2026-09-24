"""Read results/*.json, print a summary, fill the README tables, and draft resume bullets.
Every number comes straight from the JSON files, so nothing is typed by hand.

README blocks it rewrites (between HTML comment markers):
  KEY      headline numbers                     (<!-- KEY:START --> ... <!-- KEY:END -->)
  RESULTS  every measured number, with source   (<!-- RESULTS:START --> ... <!-- RESULTS:END -->)
  COMPARE  single-stage vs two-stage cache, and baseline vs optimized load path

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

# Comparison files (kept in results/ next to the current ones):
SINGLE_QQP = "qqp_threshold_single_stage.json"  # semantic cache with MiniLM alone
SINGLE_REPLAY = "replay_single_stage.json"
LOAD_STEPS = [  # same load test, each optimisation added on top of the previous one
    ("Baseline", "load_baseline.json"),
    ("+ dedicated thread pool", "load_threadpool_only.json"),
    ("+ micro-batching", "load_threadpool_batching.json"),
    ("+ cross-encoder check (final)", "load_optimized.json"),
]


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
            rows.append(("Semantic cache: MiniLM candidate / cross-encoder threshold",
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
                    sem_prec=r["semantic_hit_precision"], all_prec=r.get("all_hit_precision"),
                    mix=r["realised_mix"])
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
                    users=ld["setup"]["users"], latency_ms=ld["setup"]["mock_latency_ms"])
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
        vals.update(tests=t["passed"], tests_total=t["total"])
        rows.append(("Automated tests passing", f"{t['passed']} of {t['total']}", "tests.json"))
    single = load(d, SINGLE_QQP)
    if single:
        vals.update(single_recall=single["chosen"]["recall"], single_precision=single["chosen"]["precision"])
    base = load(d, LOAD_STEPS[0][1])
    if base:
        vals.update(base_p95_overhead=base["gateway_overhead_ms"]["p95"])
    return rows, vals


def machine_line(d: Path) -> str:
    for name in ("load.json", "chaos.json", "replay.json", "qqp_threshold.json"):
        j = load(d, name)
        if j:
            m = j["machine"]
            return (f"Measured on: {m.get('platform')}, {m.get('cpu_count')} logical CPUs, "
                    f"{m.get('ram_gb', '?')} GB RAM, Python {m.get('python')}.")
    return ""


def overhead_cut(v: dict) -> int:
    """Whole-percent reduction in p95 overhead, rounded DOWN so it never overstates."""
    return math.floor(100 * (1 - v["p95_overhead"] / v["base_p95_overhead"]))


def key_table(v: dict, src: str) -> list[str]:
    need = {"precision", "recall", "single_recall", "hit_rate", "saved", "all_prec", "success_with",
            "success_without", "fault_rate", "p95_overhead", "base_p95_overhead", "rps", "users", "tests"}
    if not need <= v.keys():
        return []
    s = lambda *names: ", ".join(f"`{src}/{n}`" for n in names)  # noqa: E731
    rows = [
        (f"Semantic-cache hit precision ({v['qqp_pairs']:,} labelled question pairs)",
         fmt_pct(v["precision"]), s("qqp_threshold.json")),
        ("Paraphrase recall: single-stage embedding → two-stage",
         f"{fmt_pct(v['single_recall'])} → {fmt_pct(v['recall'])}", s(SINGLE_QQP, "qqp_threshold.json")),
        (f"Cache hit rate / estimated LLM spend saved ({v['replay_n']:,}-request replay)",
         f"{fmt_pct(v['hit_rate'])} / {v['saved']:.1f}%", s("replay.json")),
        ("Cache hits that returned a correct answer", fmt_pct(v["all_prec"]), s("replay.json")),
        (f"Success rate at {100 * v['fault_rate']:.0f}% provider faults: protected vs unprotected",
         f"{v['success_with']:.1f}% vs {v['success_without']:.1f}%", s("chaos.json")),
        (f"p95 gateway overhead at {v['users']} concurrent users: baseline → optimized",
         f"{v['base_p95_overhead']:.0f} → {v['p95_overhead']:.0f} ms ({overhead_cut(v)}% lower)",
         s(LOAD_STEPS[0][1], "load.json")),
        (f"Throughput ({v['users']} users, {v['latency_ms']:.0f} ms provider latency)",
         f"{v['rps']:.1f} req/s", s("load.json")),
        ("Automated tests", f"{v['tests']} passing", s("tests.json")),
    ]
    return ["| Metric | Result | Source |", "|---|---|---|", *[f"| {a} | {b} | {c} |" for a, b, c in rows]]


def compare_tables(d: Path, src: str) -> list[str]:
    out = []
    single, two = load(d, SINGLE_QQP), load(d, "qqp_threshold.json")
    rs, rt = load(d, SINGLE_REPLAY), load(d, "replay.json")
    if single and two and rs and rt:
        sc, tc = single["chosen"], two["chosen"]
        rows = [
            ("Threshold(s)", f"{sc['threshold']:.2f}",
             f"{tc.get('candidate_threshold', 0):.2f} / {tc['threshold']:.2f}"),
            ("QQP hit precision", fmt_pct(sc["precision"]), fmt_pct(tc["precision"])),
            ("QQP paraphrase recall", fmt_pct(sc["recall"]), fmt_pct(tc["recall"])),
            ("QQP false-hit rate", fmt_pct(sc["false_hit_rate"], 2), fmt_pct(tc["false_hit_rate"], 2)),
            ("Replay hit rate", fmt_pct(rs["hit_rate"]), fmt_pct(rt["hit_rate"])),
            ("Replay paraphrases served from cache", fmt_pct(rs["paraphrase_catch_rate"]),
             fmt_pct(rt["paraphrase_catch_rate"])),
            ("Replay semantic hits correct", fmt_pct(rs["semantic_hit_precision"]),
             fmt_pct(rt["semantic_hit_precision"])),
            ("Replay all hits correct", fmt_pct(rs["all_hit_precision"]), fmt_pct(rt["all_hit_precision"])),
            ("Replay wrong answers served", f"{rs['wrong_answers_served']:,} of {rs['requests']:,}",
             f"{rt['wrong_answers_served']:,} of {rt['requests']:,}"),
            ("Replay estimated spend saved", f"{rs['est_cost_saved_pct']:.1f}%", f"{rt['est_cost_saved_pct']:.1f}%"),
        ]
        out += ["**Semantic cache: single-stage vs two-stage** "
                f"(`{src}/{SINGLE_QQP}`, `{src}/{SINGLE_REPLAY}` vs `{src}/qqp_threshold.json`, `{src}/replay.json`)",
                "",
                "| Metric | Single-stage (MiniLM only) | Two-stage (MiniLM + cross-encoder) |", "|---|---|---|",
                *[f"| {a} | {b} | {c} |" for a, b, c in rows], ""]
    steps = [(name, load(d, f), f) for name, f in LOAD_STEPS]
    steps = [s for s in steps if s[1]]
    if len(steps) >= 2:
        metrics = [
            ("Throughput", lambda j: f"{j['through_gateway']['rps']:.1f} req/s"),
            ("Gateway overhead p50", lambda j: f"{j['gateway_overhead_ms']['p50']:.1f} ms"),
            ("Gateway overhead p95", lambda j: f"{j['gateway_overhead_ms']['p95']:.1f} ms"),
            ("Gateway overhead p99", lambda j: f"{j['gateway_overhead_ms']['p99']:.1f} ms"),
            ("In-gateway time p95", lambda j: f"{j['gateway_internal_overhead_ms']['p95']:.1f} ms"),
            ("Success rate", lambda j: f"{j['through_gateway']['success_pct']:.1f}%"),
        ]
        out += ["**Load path: baseline vs optimized** (each column adds one change to the one before; "
                "files: " + ", ".join(f"`{src}/{f}`" for _, _, f in steps) + ")",
                "",
                "| Metric | " + " | ".join(n for n, _, _ in steps) + " |", "|---|" + "---|" * len(steps),
                *[f"| {label} | " + " | ".join(fn(j) for _, j, _ in steps) + " |" for label, fn in metrics]]
    return out


def bullets(v: dict) -> list[str]:
    out = []
    if {"saved", "hit_rate"} <= v.keys():
        kind = "two-stage semantic" if v.get("two_stage") else "semantic"
        out.append(f"Built an async LLM gateway with exact + {kind} caching, cutting estimated LLM spend "
                   f"**{v['saved']:.1f}%** at **{100 * v['hit_rate']:.1f}%** hit rate.")
    if v.get("two_stage") and "single_recall" in v:
        r0, r1 = v["single_recall"], v["recall"]
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
    if {"base_p95_overhead", "p95_overhead", "users"} <= v.keys():
        out.append(f"Cut p95 gateway overhead **{overhead_cut(v)}%** (**{v['base_p95_overhead']:.0f}** to "
                   f"**{v['p95_overhead']:.0f} ms**) at {v['users']} concurrent users with a thread pool and "
                   "micro-batching.")
    return out


def replace_block(text: str, name: str, body: list[str]) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    block = "\n".join([start, "", *body, "", end])
    return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)


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
    blocks = {
        "KEY": key_table(vals, args.dir),
        "RESULTS": [machine_line(d), "", "| Metric | Value | Source |", "|---|---|---|",
                    *[f"| {a} | {b} | `{args.dir}/{c}` |" for a, b, c in rows]],
        "COMPARE": compare_tables(d, args.dir),
    }
    for name, body in blocks.items():
        if body:
            print(f"\n[{name}]\n" + "\n".join(body))
    print("\nDraft resume bullets (check length and wording before using):")
    for b in bullets(vals):
        print(f"- {b}  ({len(b.replace('**', ''))} chars)")
    if not args.no_readme and args.dir == "results":
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        for name, body in blocks.items():
            if body:
                text = replace_block(text, name, body)
        readme.write_text(text, encoding="utf-8")
        print("\nREADME.md tables updated.")


if __name__ == "__main__":
    main()
