"""Insert a prepared corpus into a running LightRAG server (one passage = one document).

Run against an isolated server (see run.sh), never the TLU workspace.
Resumable: progress lives in the server's storage. Re-running re-queues nothing that is already indexed,
and LightRAG picks up pending / processing / failed documents again. Failed documents (e.g. every model
rate-limited) are retried --retries times, --wait seconds apart; if some still fail the script exits 2
so run.sh stops before querying an incomplete graph — re-run later to finish.

Usage: python eval/multihop/index.py eval/multihop/musique_n300 --server-url http://localhost:9641
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir")
    ap.add_argument("--server-url", default="http://localhost:9641")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--wait", type=int, default=300)
    a = ap.parse_args()

    corpus = json.load(open(Path(a.bench_dir) / "corpus.json", encoding="utf-8"))
    for i in range(0, len(corpus), a.batch):
        part = corpus[i:i + a.batch]
        r = requests.post(f"{a.server_url}/documents/texts", timeout=120, json={
            "texts": [f"{p['title']}\n{p['text']}" for p in part],
            "file_sources": [p["key"] for p in part],
        })
        r.raise_for_status()
        print(f"queued {i + len(part)}/{len(corpus)}")
    # On a resumed run every text is a duplicate, so queuing starts no pipeline; this call (re)starts it
    # for pending, interrupted (processing) and failed documents alike.
    requests.post(f"{a.server_url}/documents/reprocess_failed", timeout=60).raise_for_status()

    for attempt in range(a.retries + 1):
        c = wait_idle(a.server_url, len(corpus))
        if not c.get("failed"):
            print("index done")
            return
        if attempt < a.retries:
            print(f"{c['failed']} documents failed; retrying in {a.wait}s ({attempt + 1}/{a.retries})")
            time.sleep(a.wait)
            requests.post(f"{a.server_url}/documents/reprocess_failed", timeout=60).raise_for_status()
    sys.exit(f"{c['failed']} documents still failed (models exhausted?) — re-run later to resume")


def wait_idle(url, total):
    """Wait until no document is pending or processing (docs are registered asynchronously after queuing)."""
    while True:
        c = requests.get(f"{url}/documents/status_counts", timeout=30).json()["status_counts"]
        left = c.get("pending", 0) + c.get("processing", 0) + c.get("preprocessed", 0)
        print(f"indexing… processed {c.get('processed', 0)}/{c.get('all', 0)}, failed {c.get('failed', 0)}")
        if c.get("all", 0) >= total and not left:
            return c
        time.sleep(20)

if __name__ == "__main__":
    main()
