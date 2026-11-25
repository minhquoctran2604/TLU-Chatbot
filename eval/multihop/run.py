"""Retrieve passages for every question and mode; store the ranked passage keys (no answer generation).

Keywords are extracted once per question (first KG mode call) and passed to every later call,
so all modes and ablation variants see identical ll/hl keywords.

Usage:
  python eval/multihop/run.py eval/multihop/musique_n300 --tag full --modes naive,hybrid,mix,graph
  python eval/multihop/run.py eval/multihop/musique_n300 --tag bfs_only --modes graph   # server started with GRAPH_FLOW_MAX_ITER=0
Output: <bench_dir>/runs/<tag>.json   (resumable)
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def ranked_keys(chunks):
    out = []
    for c in chunks:
        k = c.get("file_path", "")
        if k.startswith("mh-") and k not in out:
            out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--modes", default="naive,hybrid,mix,graph")
    ap.add_argument("--server-url", default="http://localhost:9641")
    ap.add_argument("--chunk-top-k", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="questions queried in parallel")
    a = ap.parse_args()

    d = Path(a.bench_dir)
    qs = json.load(open(d / "questions.json", encoding="utf-8"))[: a.limit or None]
    kw_file, out_file = d / "keywords.json", d / "runs" / f"{a.tag}.json"
    out_file.parent.mkdir(exist_ok=True)
    kws = json.load(open(kw_file)) if kw_file.exists() else {}
    res = json.load(open(out_file)) if out_file.exists() else {}

    lock, done = threading.Lock(), [0]

    def one(q):
        for mode in a.modes.split(","):
            if f"{q['id']}|{mode}" in res:
                continue
            body = {"query": q["question"], "mode": mode, "chunk_top_k": a.chunk_top_k, "enable_rerank": False}
            if q["id"] in kws:
                body.update(kws[q["id"]])
            t0 = time.time()
            # an empty result is almost always a transient backend error (e.g. embedding OOM): retry, never store it
            for attempt in range(4):
                try:
                    r = requests.post(f"{a.server_url}/query/data", json=body, timeout=600)
                    r.raise_for_status()  # 500 = LLM worker timeout on a slow free model: worth a retry
                    j = r.json()
                    if j.get("data", {}).get("chunks"):
                        break
                except requests.RequestException as e:
                    print(f"{q['id']}|{mode}: {e} (retry {attempt + 1})", flush=True)
                time.sleep(30)
            else:
                sys.exit(f"empty result for {q['id']}|{mode} (backend down?) — re-run later to resume")
            if mode != "naive" and q["id"] not in kws:
                k = j.get("metadata", {}).get("keywords", {})
                if not (k.get("high_level") or k.get("low_level")):
                    sys.exit(f"keyword extraction failed for {q['id']} (models exhausted?) — re-run later to resume")
                with lock:  # dicts are only mutated under the lock, so json.dump never sees them change
                    kws[q["id"]] = {"hl_keywords": k.get("high_level", []), "ll_keywords": k.get("low_level", [])}
                    json.dump(kws, open(kw_file, "w"), ensure_ascii=False)
            data = j.get("data", {})
            row = {"keys": ranked_keys(data.get("chunks", [])),
                   "entities": [e.get("entity_name", "") for e in data.get("entities", [])],
                   "latency": round(time.time() - t0, 2)}  # inflated under --workers > 1
            with lock:
                res[f"{q['id']}|{mode}"] = row
        with lock:
            json.dump(res, open(out_file, "w"), ensure_ascii=False)
            done[0] += 1
            print(f"[{a.tag}] {done[0]}/{len(qs)}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, qs))  # re-raises the first failure (SystemExit) after in-flight questions finish


if __name__ == "__main__":
    main()
