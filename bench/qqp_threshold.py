"""Benchmark 1: choose the semantic-cache threshold from labelled data.

Quora Question Pairs (GLUE QQP) has question pairs labelled by humans as
"same meaning" (1) or "different" (0). For each pair we compute the MiniLM cosine
similarity, then sweep thresholds. At threshold t, "similarity >= t" means "the cache
would reuse the answer". We then measure:

  precision  of the pairs the cache WOULD match, the share that truly mean the same.
             (1 - precision = share of cache hits that would return a WRONG answer.)
  recall     of the truly-same pairs, the share the cache would catch.
  false-hit rate  of the truly-DIFFERENT pairs, the share wrongly matched.

We pick the LOWEST threshold whose precision >= target (default 95%): lowest, because a
lower threshold catches more paraphrases (more savings) while still meeting the bar.

Uses the QQP *validation* split. The replay benchmark uses *train* pairs and drops any pair that
shares a question with validation, so tuning and evaluation never see the same text.

Two-stage mode (--two-stage): MiniLM similarity >= a FIXED candidate threshold (0.85) picks the
candidate, then a cross-encoder scores each candidate pair. We sweep the cross-encoder threshold
instead, with the same rule (lowest one with precision >= target). A pair counts as a hit only if
it passes BOTH stages, exactly like the gateway.

Run:   python bench/qqp_threshold.py                  (real run, ~20k pairs)
       python bench/qqp_threshold.py --two-stage      (two-stage cache: sweep the verify threshold)
       python bench/qqp_threshold.py --smoke          (tiny offline check, hash embedder)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench._common import save_json  # noqa: E402

THRESHOLDS = np.round(np.arange(0.70, 0.9901, 0.01), 2)
VERIFY_THRESHOLDS = np.round(np.arange(0.05, 0.9901, 0.01), 2)  # cross-encoder scores are 0..1


def load_pairs(args) -> tuple[list[str], list[str], np.ndarray, dict]:
    if args.smoke:
        from bench.smoke_data import PAIRS

        q1, q2, y = zip(*PAIRS)
        return list(q1), list(q2), np.array(y), {"dataset": "bench/smoke_data.py (smoke only)"}
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "qqp", split=args.split)
    total = len(ds)
    n = min(args.sample, total) if args.sample else total
    idx = np.random.default_rng(args.seed).choice(total, size=n, replace=False)
    sub = ds.select(idx.tolist())
    info = {"dataset": f"GLUE QQP ({args.split} split)", "split_size": total, "sampled_pairs": n,
            "seed": args.seed}
    return list(sub["question1"]), list(sub["question2"]), np.array(list(sub["label"]), dtype=int), info


def encode_all(texts: list[str], embedder, batch: int = 256) -> np.ndarray:
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        out.append(embedder.encode(texts[i:i + batch]))
        done = min(i + batch, len(texts))
        print(f"\r  encoded {done}/{len(texts)}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return np.vstack(out)


def sweep(sims: np.ndarray, y: np.ndarray, thresholds=THRESHOLDS) -> list[dict]:
    rows = []
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    for t in thresholds:
        pred = sims >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        rows.append({
            "threshold": float(t),
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / pos, 4) if pos else None,
            "false_hit_rate": round(fp / neg, 4) if neg else None,
            "match_rate": round(float(pred.mean()), 4),
            "tp": tp, "fp": fp,
        })
    return rows


def choose(rows: list[dict], target: float) -> dict:
    ok = [r for r in rows if r["precision"] is not None and r["precision"] >= target]
    if ok:
        best = min(ok, key=lambda r: r["threshold"])
        # If even the lowest threshold in the sweep passes, the true optimum may be lower still.
        return {**best, "target_met": True, "at_sweep_floor": best["threshold"] == rows[0]["threshold"]}
    best = max((r for r in rows if r["precision"] is not None), key=lambda r: r["precision"])
    return {**best, "target_met": False}


def plot(rows: list[dict], chosen: dict, target: float, path: Path, title: str,
         xlabel: str = "Similarity threshold") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    t = [r["threshold"] for r in rows]
    prec = [r["precision"] if r["precision"] is not None else np.nan for r in rows]
    rec = [r["recall"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.plot(t, prec, color=blue, lw=2, marker="o", ms=4, label="Precision (hits that are correct)")
    ax.plot(t, rec, color=orange, lw=2, marker="o", ms=4, label="Recall (paraphrases caught)")
    ax.axhline(target, color=ink2, lw=1, ls="--")
    ax.text(t[0], target + 0.01, f"target precision {target:.0%}", color=ink2, fontsize=8)
    ax.axvline(chosen["threshold"], color=ink2, lw=1, ls=":")
    label = "chosen" if chosen["target_met"] else "best (target not met)"
    right = chosen["threshold"] > 0.94
    ax.annotate(f"{label} {chosen['threshold']:.2f}\nprecision {chosen['precision']:.1%}\nrecall {chosen['recall']:.1%}",
                xy=(chosen["threshold"], chosen["precision"]), xytext=(-8 if right else 8, -48),
                textcoords="offset points", ha="right" if right else "left", fontsize=8, color=ink)
    ax.set_xlabel(xlabel, color=ink2)
    ax.set_ylabel("Rate", color=ink2)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=ink, fontsize=11, loc="left")
    ax.grid(color=grid, lw=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=20000, help="pairs to sample (0 = whole split)")
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", type=float, default=0.95, help="minimum precision")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--smoke", action="store_true", help="offline check with tiny data + hash embedder")
    ap.add_argument("--two-stage", action="store_true", help="FAISS candidate + cross-encoder verify")
    ap.add_argument("--candidate", type=float, default=0.85, help="two-stage: fixed MiniLM candidate threshold")
    ap.add_argument("--verify-model", default="cross-encoder/quora-distilroberta-base")
    ap.add_argument("--out", default=None, help="output folder (default results/, smoke: trial_results/)")
    args = ap.parse_args()
    out = Path(args.out or ("trial_results" if args.smoke else "results"))

    q1, q2, y, info = load_pairs(args)
    print(f"{len(y)} pairs, {int(y.sum())} duplicates ({y.mean():.1%})")

    from gateway.cache import collapse_text
    from gateway.semantic_cache import build_embedder

    embedder = build_embedder("hash" if args.smoke else "minilm", args.model)
    # Same text preparation as the gateway's semantic cache: collapse whitespace, lowercase.
    prep = lambda s: collapse_text(s or "")  # noqa: E731
    t0 = time.time()
    a = encode_all([prep(s) for s in q1], embedder)
    b = encode_all([prep(s) for s in q2], embedder)
    sims = (a * b).sum(axis=1)
    definitions = {
        "precision": "TP / (TP + FP): share of would-be cache hits whose questions truly mean the same",
        "recall": "TP / all duplicate pairs: share of paraphrases the cache would catch",
        "false_hit_rate": "FP / all non-duplicate pairs: share of different-meaning pairs wrongly matched",
        "match_rate": "share of all pairs with similarity >= threshold",
        "rule": "lowest threshold with precision >= target",
    }

    if args.two_stage:
        run_two_stage(args, out, q1, q2, y, sims, prep, info, definitions, t0)
        return

    rows = sweep(sims, y)
    chosen = choose(rows, args.target)

    data = {
        "benchmark": "qqp_threshold",
        "mode": "single_stage",
        "smoke_only_not_for_resume": bool(args.smoke),
        **info,
        "embedder": "hash (smoke)" if args.smoke else args.model,
        "duplicate_share": round(float(y.mean()), 4),
        "target_precision": args.target,
        "chosen": chosen,
        "sweep": rows,
        "encode_seconds": round(time.time() - t0, 1),
        "definitions": definitions,
    }
    save_json(out, "qqp_threshold.json", data)
    plot(rows, chosen, args.target, out / "qqp_threshold.png",
         f"SMOKE ONLY: threshold sweep on {len(y)} hand-written pairs (hash embedder)" if args.smoke
         else f"Semantic-cache threshold on {len(y):,} QQP pairs (MiniLM)")
    flag = "" if chosen["target_met"] else "  (target NOT met: showing best precision)"
    print(f"chosen threshold {chosen['threshold']}: precision {chosen['precision']}, "
          f"recall {chosen['recall']}, false-hit rate {chosen['false_hit_rate']}{flag}")
    if not args.smoke:
        print(f"-> set SEMANTIC_THRESHOLD={chosen['threshold']} in .env")


def run_two_stage(args, out, q1, q2, y, sims, prep, info, definitions, t0) -> None:
    """Stage 1 fixed (MiniLM >= args.candidate); sweep stage 2 (cross-encoder score)."""
    import os

    import torch

    from gateway.semantic_cache import CrossEncoderVerifier

    cand = sims >= args.candidate
    idx = np.flatnonzero(cand)
    print(f"{len(idx)} of {len(y)} pairs pass stage 1 (MiniLM >= {args.candidate}); scoring them with "
          f"{args.verify_model}")
    verifier = CrossEncoderVerifier(args.verify_model)
    torch.set_num_threads(os.cpu_count() or 1)  # offline benchmark: speed only, results are identical
    scores = np.full(len(y), -np.inf)  # pairs that fail stage 1 can never be hits
    t1 = time.time()
    batch = 128
    for i in range(0, len(idx), batch):
        part = idx[i:i + batch]
        scores[part] = verifier.score([(prep(q2[j]), prep(q1[j])) for j in part])
        print(f"\r  verified {min(i + batch, len(idx))}/{len(idx)}  ({time.time() - t1:.0f}s)", end="", flush=True)
    print()
    s = scores[idx]
    print(f"  cross-encoder score range on candidates: {s.min():.4f} .. {s.max():.4f}")

    rows = sweep(scores, y, VERIFY_THRESHOLDS)  # hit = passes stage 1 AND score >= t
    chosen = {"candidate_threshold": args.candidate, "verify_threshold": None,
              **choose(rows, args.target)}
    chosen["verify_threshold"] = chosen["threshold"]
    stage1 = sweep(sims, y, np.array([args.candidate]))[0]  # stage 1 alone = recall ceiling

    data = {
        "benchmark": "qqp_threshold",
        "mode": "two_stage",
        "smoke_only_not_for_resume": False,
        **info,
        "embedder": args.model,
        "verifier": args.verify_model,
        "verifier_training_note": "cross-encoder/quora-distilroberta-base was trained on QQP train; "
                                  "this evaluation uses QQP validation pairs only",
        "duplicate_share": round(float(y.mean()), 4),
        "target_precision": args.target,
        "candidate_threshold": args.candidate,
        "stage1_only_at_candidate": stage1,
        "chosen": chosen,
        "sweep": rows,
        "sweep_variable": "verify_threshold (cross-encoder score), candidate threshold fixed",
        "candidates_scored": int(len(idx)),
        "encode_seconds": round(time.time() - t0, 1),
        "verify_seconds": round(time.time() - t1, 1),
        "definitions": {**definitions,
                        "match_rate": "share of all pairs passing BOTH stages",
                        "rule": "lowest verify threshold with precision >= target (candidate fixed)"},
    }
    save_json(out, "qqp_threshold.json", data)
    plot(rows, chosen, args.target, out / "qqp_threshold.png",
         f"Two-stage cache on {len(y):,} QQP pairs: MiniLM >= {args.candidate}, then cross-encoder",
         xlabel="Cross-encoder verify threshold")
    flag = "" if chosen["target_met"] else "  (target NOT met: showing best precision)"
    print(f"stage 1 alone at {args.candidate}: precision {stage1['precision']}, recall {stage1['recall']}")
    print(f"chosen verify threshold {chosen['verify_threshold']}: precision {chosen['precision']}, "
          f"recall {chosen['recall']}, false-hit rate {chosen['false_hit_rate']}{flag}")
    print(f"-> set SEMANTIC_CANDIDATE_THRESHOLD={args.candidate} and "
          f"SEMANTIC_VERIFY_THRESHOLD={chosen['verify_threshold']} in .env")


if __name__ == "__main__":
    main()
