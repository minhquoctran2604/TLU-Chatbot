"""Recall@2/@5 of supporting passages per (run tag, mode), by hop count, with paired tests vs a reference.

Recall@k = |top-k passages ∩ gold| / |gold|   (HippoRAG's retrieval metric)
Bridge rank = position of the first bridge entity in the returned entity list (Hit@5, Hit@10, MRR):
this is what PPR re-scores, so it is the most direct read-out of PPR's contribution.
Paired test: Wilcoxon signed-rank per question, plus a 95% bootstrap CI of the mean difference.

Usage: python eval/multihop/report.py eval/multihop/musique_n300 [--ref full:graph]
"""
import argparse
import json
import random
import re
from pathlib import Path

from scipy.stats import wilcoxon


def recall(keys, gold, k):
    return len(set(keys[:k]) & set(gold)) / len(gold)


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def bridge_rr(entities, bridge):
    """Reciprocal rank of the first returned entity that names a bridge entity (0 if none)."""
    bs = [norm(b) for b in bridge]
    for i, e in enumerate(norm(x) for x in entities):
        if any(e == b or (len(b) >= 4 and b in e) or (len(e) >= 4 and e in b) for b in bs):
            return 1 / (i + 1)
    return 0.0


def paired(diff):
    if not any(diff):
        return "identical"
    lo, hi = boot_ci(diff)
    return f"Δ={sum(diff) / len(diff):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  Wilcoxon p={wilcoxon(diff).pvalue:.3g}"


def boot_ci(diff, n=10000, seed=0):
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choices(diff, k=len(diff))) / len(diff) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--ref", default="full:graph", help="tag:mode used as the reference in paired tests")
    a = ap.parse_args()

    d = Path(a.bench_dir)
    qs = json.load(open(d / "questions.json", encoding="utf-8"))
    runs = {p.stem: json.load(open(p)) for p in sorted((d / "runs").glob("*.json"))}
    systems = sorted({(t, k.split("|")[1]) for t, r in runs.items() for k in r})
    hops = sorted({q["hops"] for q in qs})

    def scores(t, m, k, qset=qs):
        return [recall(runs[t][f"{q['id']}|{m}"]["keys"], q["gold"], k) for q in qset if f"{q['id']}|{m}" in runs[t]]

    head = f"{'system':24} {'n':>4} {'R@2':>6} {'R@5':>6}" + "".join(f" {f'R@5 {h}hop':>9}" for h in hops)
    print(head)
    for t, m in systems:
        r5 = scores(t, m, 5)
        line = f"{t + ':' + m:24} {len(r5):4d} {sum(scores(t, m, 2)) / len(r5):6.3f} {sum(r5) / len(r5):6.3f}"
        for h in hops:
            s = scores(t, m, 5, [q for q in qs if q["hops"] == h])
            line += f" {sum(s) / len(s):9.3f}" if s else f" {'-':>9}"
        print(line)

    rt, rm = a.ref.split(":")
    print(f"\npaired vs {a.ref} on Recall@5 (questions answered by both)")
    for t, m in systems:
        if (t, m) == (rt, rm):
            continue
        both = [q for q in qs if f"{q['id']}|{m}" in runs[t] and f"{q['id']}|{rm}" in runs[rt]]
        diff = [x - y for x, y in zip(scores(t, m, 5, both), scores(rt, rm, 5, both))]
        print(f"{t + ':' + m:24} {paired(diff)}")

    def rr(t, m, qset):
        return [bridge_rr(runs[t][f"{q['id']}|{m}"].get("entities", []), q["bridge"]) for q in qset]

    bq = [q for q in qs if q.get("bridge")]
    print(f"\nbridge entity rank ({len(bq)} questions with a bridge entity; graph-based modes)")
    print(f"{'system':24} {'n':>4} {'Hit@5':>6} {'Hit@10':>6} {'MRR':>6}")
    kg = [(t, m) for t, m in systems if m != "naive"]
    for t, m in kg:
        got = [q for q in bq if f"{q['id']}|{m}" in runs[t]]
        r = rr(t, m, got)
        if r:
            print(f"{t + ':' + m:24} {len(r):4d} {sum(x >= 1 / 5 for x in r) / len(r):6.3f} "
                  f"{sum(x >= 1 / 10 for x in r) / len(r):6.3f} {sum(r) / len(r):6.3f}")
    print(f"\npaired vs {a.ref} on bridge reciprocal rank")
    for t, m in kg:
        if (t, m) == (rt, rm):
            continue
        both = [q for q in bq if f"{q['id']}|{m}" in runs[t] and f"{q['id']}|{rm}" in runs[rt]]
        diff = [x - y for x, y in zip(rr(t, m, both), rr(rt, rm, both))]
        print(f"{t + ':' + m:24} {paired(diff)}")


if __name__ == "__main__":
    main()
