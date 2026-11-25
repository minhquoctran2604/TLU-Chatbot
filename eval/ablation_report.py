"""Graph-mode ablation helper.

  merge  : eval/ablation/{variant}.json -> merged.json, variants mapped onto the
           5 mode slots evaluate_pairwise.py expects.
  report : judged.json -> Table 5 rows (mean rank, rank-1 wins) + Friedman/Nemenyi.
"""
import json
import sys
from math import sqrt
from pathlib import Path

from scipy import stats

D = Path(__file__).parent / "ablation"
SLOTS = {"full": "bm25", "bfs_only": "naive", "no_flow_rank": "hybrid",
         "no_structure": "mix", "depth1": "graph"}
Q_NEMENYI = 2.728  # k=5, alpha=0.05 (Demsar 2006)


def merge():
    out = []
    for v, slot in SLOTS.items():
        for r in json.load(open(D / f"{v}.json", encoding="utf-8"))["results"]:
            out.append({**r, "mode": slot, "variant": v})
    json.dump({"results": out}, open(D / "merged.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"merged {len(out)} answers")


def report():
    back = {s: v for v, s in SLOTS.items()}
    res = [r for r in json.load(open(D / "judged.json", encoding="utf-8"))["results"] if "ranks" in r]
    n, k = len(res), len(SLOTS)
    ranks = {v: [r["ranks"][s] for r in res] for s, v in back.items()}
    mean = {v: sum(x) / n for v, x in ranks.items()}
    wins = {v: sum(x == 1 for x in xs) for v, xs in ranks.items()}
    assert abs(sum(mean.values()) - 15) < 1e-6 and sum(wins.values()) == n  # sanity: strict 5-way ranks

    chi2, p = stats.friedmanchisquare(*ranks.values())
    cd = Q_NEMENYI * sqrt(k * (k + 1) / (6 * n))
    best = min(mean.values())
    print(f"n={n} judged ({sum('error' in r for r in json.load(open(D / 'judged.json'))['results'])} failed)")
    print(f"Friedman chi2(4)={chi2:.2f} p={p:.3g} | Nemenyi CD={cd:.3f}\n")
    print(f"{'Variant':<14}{'Mean rank':>10}{'Wins':>6}  within CD of best")
    for v in SLOTS:
        print(f"{v:<14}{mean[v]:>10.2f}{wins[v]:>6}  {'yes' if mean[v] - best < cd else 'no'}")


if __name__ == "__main__":
    {"merge": merge, "report": report}[sys.argv[1]]()
