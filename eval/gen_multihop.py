"""Generate multi-hop cross-subject benchmark queries.

Strategy:
  1. Audit the KG graphml to find "bridge entities" — entities whose chunks
     span 2+ subjects. These are points where graph traversal is theoretically
     more valuable than pure vector search.
  2. For each of the chosen subject pairs, pick top-K bridge entities ranked
     by chunk presence + description quality.
  3. For each (pair, entity), sample 1 chunk from subject A + 1 chunk from
     subject B + the entity description. Prompt the LLM to compose a query
     that requires understanding BOTH chunks (and the bridge entity) to answer.

The resulting queries should expose pure graph-mode's strength: multi-hop
reasoning that single-chunk retrieval cannot satisfy.

Usage:
  python gen_multihop.py --count 5 --pairs 12 --out eval/queries_multihop.json
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time as _time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from gen_queries import SUBJECTS, detect_subject, _strip_images  # reuse

ROUTER_HOST = os.getenv("ROUTER_HOST", "http://localhost:20128/v1")
ROUTER_KEY = os.getenv("ROUTER_API_KEY", "dummy")
GEN_MODEL = os.getenv("GEN_QUERY_MODEL", "nvidia/minimaxai/minimax-m2.7")
_env_fb = os.getenv("GEN_FAILOVER_MODELS", "") or os.getenv("ROUTER_FAILOVER_MODELS", "")
GEN_FAILOVER = [m.strip() for m in _env_fb.split(",") if m.strip() and m.strip() != GEN_MODEL]
GEN_CHAIN = [GEN_MODEL] + GEN_FAILOVER
CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "3"))
MAX_RPM = int(os.getenv("GEN_MAX_RPM", "40"))
MIN_INTERVAL = 60.0 / MAX_RPM

WORKING_DIR = os.getenv("WORKING_DIR", "../tlu_workspace")
GRAPHML = Path(WORKING_DIR).expanduser().resolve() / "graph_chunk_entity_relation.graphml"
if not GRAPHML.exists():
    GRAPHML = Path("/home/tts/AI/aiQuoc/tlu_workspace/graph_chunk_entity_relation.graphml")

# Entity names too generic to gen meaningful cross-subject questions about.
GENERIC = {
    "data", "title", "information", "system", "content", "process", "value",
    "user", "users", "method", "type", "name", "result", "model", "table",
    "size", "format", "id", "key", "field", "object", "function", "page",
    "input", "output", "code", "file", "task", "test", "service", "tool",
    "step", "stage", "phase", "concept", "ai", "it", "image", "text",
    "understanding", "knowledge", "study", "research", "example", "case",
    "application", "form", "speed", "variable", "class", "element",
    "attribute", "label", "rule", "customer", "modeling", "maintenance",
}

MULTIHOP_PROMPT = """Sinh 1 câu hỏi MULTI-HOP CROSS-SUBJECT (bắc cầu giữa 2 lĩnh vực khác nhau qua một khái niệm chung).

Khái niệm cầu nối: **{entity}**
Mô tả: {entity_desc}

Lĩnh vực A: **{subject_a}**
Nội dung A:
{chunk_a}

Lĩnh vực B: **{subject_b}**
Nội dung B:
{chunk_b}

Yêu cầu:
- Câu hỏi tiếng Việt, BẮT BUỘC đòi hỏi hiểu CẢ HAI lĩnh vực để trả lời ("Vai trò của X trong môn A liên hệ thế nào với X trong môn B?", "So sánh cách X được dùng trong A và B?", "Khi X xuất hiện ở A, nó ảnh hưởng tới B ra sao?")
- KHÔNG được trả lời được nếu chỉ đọc 1 trong 2 lĩnh vực
- Reference (3-6 câu) phải bridge cả hai: giải thích vai trò của khái niệm trong từng lĩnh vực + mối liên hệ
- evidence_keywords gồm 4-6 từ khoá bao gồm cả 2 lĩnh vực + khái niệm cầu

Trả về JSON duy nhất (không markdown):
{{"query": "...", "reference": "...", "evidence_keywords": ["...", "..."], "expected_best_mode": "graph"}}
"""

_last_call_time = [0.0]
_rate_lock = None


def _connect_pg():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        dbname=os.environ['POSTGRES_DATABASE'],
    )


def load_chunk_contents(chunk_ids: set) -> dict:
    """Load chunk content from Postgres by chunk_id (id column)."""
    if not chunk_ids:
        return {}
    conn = _connect_pg()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, content FROM lightrag_doc_chunks WHERE id = ANY(%s)",
        (list(chunk_ids),),
    )
    out = {cid: content for cid, content in cur.fetchall() if content and len(content) >= 200}
    conn.close()
    return out


def find_bridges() -> dict:
    """Return {(subj_a, subj_b): [bridge_entity_dict, ...]} sorted by quality score."""
    print(f"[GRAPH] Loading {GRAPHML}")
    G = nx.read_graphml(GRAPHML)
    print(f"[GRAPH] Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}")

    bridge_by_pair = defaultdict(list)
    for node, attrs in G.nodes(data=True):
        fp = attrs.get('file_path', '')
        sids = attrs.get('source_id', '')
        desc = attrs.get('description', '')
        if not fp or not sids:
            continue
        name_lower = node.lower().strip()
        if name_lower in GENERIC or len(name_lower) < 4:
            continue
        if len(desc) < 100:
            continue

        files = fp.split('<SEP>')
        chunks = sids.split('<SEP>')
        subj_chunks = defaultdict(list)
        for f, c in zip(files, chunks):
            s = detect_subject(f)
            if s:
                subj_chunks[s].append(c)
        if len(subj_chunks) < 2:
            continue

        subjs = sorted(subj_chunks.keys())
        for i in range(len(subjs)):
            for j in range(i+1, len(subjs)):
                sa, sb = subjs[i], subjs[j]
                na, nb = len(subj_chunks[sa]), len(subj_chunks[sb])
                if na < 1 or nb < 1:
                    continue
                # Quality score: balanced presence + description richness
                score = min(na, nb) * 2 + (na + nb) + len(desc) / 200
                bridge_by_pair[(sa, sb)].append({
                    'entity': node,
                    'desc': desc[:500].replace('<SEP>', ' | '),
                    'n_a': na, 'n_b': nb,
                    'chunks_a': subj_chunks[sa],
                    'chunks_b': subj_chunks[sb],
                    'score': score,
                })

    for pair in bridge_by_pair:
        bridge_by_pair[pair].sort(key=lambda x: -x['score'])

    return bridge_by_pair


async def _rate_limit_wait():
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    async with _rate_lock:
        now = _time.time()
        wait = _last_call_time[0] + MIN_INTERVAL - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time[0] = _time.time()


async def gen_one(client, bridge: dict, sa: str, sb: str, chunk_a_text: str, chunk_b_text: str):
    """Gen 1 multihop query about bridge entity across subjects sa and sb."""
    prompt = MULTIHOP_PROMPT.format(
        entity=bridge['entity'],
        entity_desc=bridge['desc'][:400],
        subject_a=SUBJECTS[sa]['label'],
        subject_b=SUBJECTS[sb]['label'],
        chunk_a=_strip_images(chunk_a_text)[:6000],
        chunk_b=_strip_images(chunk_b_text)[:6000],
    )
    last_err = None
    for model in GEN_CHAIN:
        await _rate_limit_wait()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                timeout=60,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                last_err = f"{model}: no JSON"
                continue
            data = json.loads(m.group(0))
            required = {"query", "reference", "evidence_keywords"}
            if not required.issubset(data.keys()):
                last_err = f"{model}: missing keys"
                continue
            data.setdefault("expected_best_mode", "graph")
            return data, None
        except Exception as ex:
            last_err = f"{model}: {type(ex).__name__}: {str(ex)[:80]}"
            continue
    return None, last_err


async def main(count_per_pair: int, num_pairs: int, out_path: Path):
    from openai import AsyncOpenAI

    bridges = find_bridges()
    sorted_pairs = sorted(bridges.items(), key=lambda x: -len(x[1]))[:num_pairs]
    print(f"\n[PAIRS] Selected {len(sorted_pairs)} pairs:")
    for (sa, sb), lst in sorted_pairs:
        print(f"  {sa:>7} ↔ {sb:<7}: {len(lst):>3} bridges")

    # Gather all chunk_ids needed
    needed_chunks = set()
    for (sa, sb), lst in sorted_pairs:
        for b in lst[:count_per_pair * 2]:  # over-sample
            needed_chunks.update(b['chunks_a'][:3])
            needed_chunks.update(b['chunks_b'][:3])
    print(f"\n[PG] Fetching {len(needed_chunks)} chunk contents...")
    chunk_content = load_chunk_contents(needed_chunks)
    print(f"[PG] Got {len(chunk_content)} chunks with content")

    client = AsyncOpenAI(base_url=ROUTER_HOST, api_key=ROUTER_KEY, timeout=60)
    sem = asyncio.Semaphore(CONCURRENCY)
    rng = random.Random(42)

    tasks = []
    plan = []  # list of (pair_idx, seq, sa, sb, bridge, chunk_a, chunk_b)

    for pair_idx, ((sa, sb), lst) in enumerate(sorted_pairs):
        usable = []
        for b in lst:
            cas = [c for c in b['chunks_a'] if c in chunk_content]
            cbs = [c for c in b['chunks_b'] if c in chunk_content]
            if cas and cbs:
                usable.append((b, cas, cbs))
            if len(usable) >= count_per_pair:
                break
        if len(usable) < count_per_pair:
            print(f"  [WARN] {sa}↔{sb}: only {len(usable)} usable bridges (need {count_per_pair})")
        for seq, (b, cas, cbs) in enumerate(usable[:count_per_pair], 1):
            ca = chunk_content[rng.choice(cas)]
            cb = chunk_content[rng.choice(cbs)]
            plan.append((pair_idx, seq, sa, sb, b, ca, cb))

    print(f"\n[GEN] Generating {len(plan)} multihop queries...")

    async def gen_with_id(pair_idx, seq, sa, sb, b, ca, cb):
        async with sem:
            result, err = await gen_one(client, b, sa, sb, ca, cb)
        qid = f"mhop-{sa}-{sb}-{seq:02d}"
        if result is None:
            print(f"  FAIL {qid} ({b['entity'][:30]}): {err}")
            return None
        new_q = {
            "id": qid,
            "type": "multihop",
            "subject": f"{SUBJECTS[sa]['label']} ↔ {SUBJECTS[sb]['label']}",
            "bridge_entity": b['entity'],
            "pair": f"{sa}_{sb}",
            "source_file": "",  # multihop spans multiple
            "query": result["query"],
            "reference": result["reference"],
            "evidence_keywords": result["evidence_keywords"],
            "expected_best_mode": result.get("expected_best_mode", "graph"),
        }
        print(f"  OK   {qid} ({b['entity'][:25]}): {result['query'][:70]}")
        return new_q

    for pair_idx, seq, sa, sb, b, ca, cb in plan:
        tasks.append(gen_with_id(pair_idx, seq, sa, sb, b, ca, cb))

    results = await asyncio.gather(*tasks)
    queries = [r for r in results if r is not None]

    out = {"meta": {"total": len(queries), "type": "multihop", "pairs": [f"{p[0][0]}_{p[0][1]}" for p in sorted_pairs]}, "queries": queries}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out_path} | total={len(queries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5, help="Queries per subject pair")
    parser.add_argument("--pairs", type=int, default=12, help="Number of subject pairs (top by bridge count)")
    parser.add_argument("--out", default="eval/queries_multihop.json")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.pairs, Path(args.out)))
