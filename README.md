# TLU-Chatbot — LightRAG fork for university-domain QA

A specialized fork of [LightRAG](https://github.com/HKUDS/LightRAG) tuned for a
university course corpus (lecture PDFs, slides, course notes in Vietnamese),
backed by a Postgres + pgvector store and a graph layer that supports
multi-hop retrieval and bridge-entity discovery.

The upstream LightRAG code lives in `LightRAG/`. This fork adds the data
pipeline, benchmark harness, and graph-ego-walk retrieval mode on top.

---

## How the system flows

```
                 ┌──────────────┐
  PDF / MD ────► │   Ingest     │  chunk → entity/relation extract (LLM)
                 └──────┬───────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │  KV      │    │  Vector  │    │  Graph   │
  │  (cache) │    │  (pgvec) │    │ (graphml)│
  └────┬─────┘    └────┬─────┘    └────┬─────┘
       └───────────────┼───────────────┘
                       ▼
                 ┌──────────────┐
                 │   Query      │  embed → seed select → retrieve
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   LLM answer │
                 └──────────────┘
```

1. **Ingest** — PDFs and markdown from `PDF/`, `MD/` are chunked and fed to
   an extraction LLM that emits `(entity, relation, description)` triples.
2. **Storage** — chunks and LLM cache go to KV (JSON / Postgres), entity +
   relation embeddings go to a vector store, the `(entity, relation)` graph
   is persisted as a GraphML file and mirrored into Postgres.
3. **Query** — the user question is embedded, used to pick seed entities,
   then a retrieval mode (naive / hybrid / mix / graph / bm25) gathers
   context, and the LLM writes the final answer.

---

## Improvements over upstream LightRAG

This fork ships four additions aimed at long-form, cross-subject academic QA.

### 1. Graph ego-walk retrieval mode

A new `graph` mode that anchors retrieval on the topology of the knowledge
graph instead of relying purely on vector similarity.

- **Seed selection** — embed `ll_keywords` from the query → top-K
  entities by cosine similarity.
- **BFS + PathRAG-style flow propagation** — flow splits across neighbors
  with `S(child) = α · S(parent) / degree(parent)`, capping hub nodes at
  `GRAPH_LARGE_TOP_N_EDGES=20` to avoid hub domination.
- **Adaptive depth** — propagation stops when `child_flow < θ` (default
  `θ = 0.05`) so the walk doesn't bleed into peripheral nodes.
- **PPR refinement** — the BFS flow warm-starts Personalized PageRank;
  PPR redistributes mass until `δ = ‖v_new − v‖₁ < θ`. The column-stochastic
  property of `Wᵀ` guarantees `δ` decreases at least geometrically
  (`≤ cᵏ ‖Δ⁰‖₁`), so refinement converges in ~5–8 iterations.
- **Entity ranking** — `rank(seed) = cos(seed) + λ · flow(seed)` with
  `λ = 0.1`, blending vector similarity with graph-proximity signal.

```
Query ──► embed ──► top-K seeds ──► BFS flow ──► PPR refine ──► rank
                                              ▲
                                              │ warm-start
                                              └──────────────┘
```

Configurable via env vars: `GRAPH_SEED_TOP_K`, `GRAPH_FLOW_ALPHA`,
`GRAPH_PPR_C`, `GRAPH_FLOW_THETA`, `GRAPH_FLOW_MAX_DEPTH`,
`GRAPH_LARGE_TOP_N_EDGES`, `GRAPH_FLOW_ENTITY_LAMBDA`,
`GRAPH_HL_KEYWORD_MODE`.

### 2. Knowledge-graph audit and cleanup

Academic PDFs produce noisy entities: page markers (`slide 12`, `trang 5`),
`[IMG_xxx]` placeholders, single-letter fragments, and case-variant
duplicates (`HTTP` vs `http`). `eval/audit_graph.py` classifies nodes into
`delete / merge / keep`, and `eval/apply_cleanup.py` and
`eval/cleanup_graph.py` rewrite the GraphML and Postgres rows in one pass
(edge weights are summed during merges, descriptions are deduped).

### 3. Multi-mode benchmark harness

`eval/` is an end-to-end evaluation kit for comparing retrieval modes:

- **Query generation** — `gen_queries.py` builds 5 query types
  (factoid, relational, broad, aggregate) from a chunked corpus.
  `gen_2hop.py` mines paths A→B→C across subjects where A and C never
  share a source file (so vector-only retrieval should fail). `gen_multihop.py`
  builds bridge-entity queries whose chunks span 2+ subjects.
- **Run** — `run_benchmark.py` POSTs each query × mode to the LightRAG
  server, throttles to provider rate limits, records raw responses +
  latency. `run_local.py` adds BM25 as a baseline mode without a server
  roundtrip.
- **Evaluate** — `evaluate_benchmark.py` scores with BERTScore
  (`xlm-roberta-large`, `lang=vi`) and reports per-mode / per-type
  aggregates with bootstrap 95 % CI. `evaluate_pairwise.py` runs LLM-as-judge
  on five anonymized responses (A–E) with per-type rubrics and Borda
  scoring. `evaluate_ragas.py` covers answer relevancy, correctness,
  faithfulness, and context precision.
- **Dashboard** — `build_dashboard.py` emits a single self-contained
  `dashboard.html` with grouped bars for win count, mean rank, and
  p50/p95 latency per mode.

### 4. Resilient LLM extraction

A separate `extract_*_llm_func` decoupled from the query LLM lets a cheap
extractor (small model) feed the index while a stronger model serves
user queries. Failover chains (`ROUTER_FAILOVER_MODELS`) route around
quota or 5xx errors without dropping documents mid-ingest. The
`KeyError` on `graph.degree(node_id)` when a seed has no edges is
handled so a single missing node doesn't abort a query.

---

## Repository layout

```
.
├── LightRAG/        ← upstream library (core + WebUI + API server)
├── eval/            ← benchmark harness, query generation, KG audit
├── PDF/             ← source lecture PDFs
├── MD/              ← source markdown notes (extracted for ingest)
├── docs/            ← design notes (PPR convergence, Algorithm walkthrough)
└── README.md
```

---

## Quick start

### Server (local)

```bash
cd LightRAG
uv sync --extra api               # or: pip install -e ".[api]"
cp env.example .env               # fill LLM / embedding / Postgres creds
cd lightrag_webui && bun install --frozen-lockfile && bun run build && cd ..
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m lightrag.api.lightrag_server
# Server now listening on http://localhost:9621
```

### Ingest

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()           # required
    await rag.ainsert(["doc1 text", "doc2 text"])
    print(await rag.aquery("...", param=QueryParam(mode="graph")))

asyncio.run(main())
```

### Benchmark

```bash
cd eval
python run_benchmark.py                        # raw responses
python evaluate_benchmark.py --bootstrap 1000  # BERTScore + 95% CI
python evaluate_pairwise.py                    # LLM-as-judge, Borda
python build_dashboard.py                      # → dashboard.html
```

### Graph cleanup

```bash
cd eval
python audit_graph.py     # → cleanup_plan.json  (dry-run)
python apply_cleanup.py   # rewrite GraphML + Postgres rows
```

---

## Storage backends used

- **KV** — `JsonKVStorage` for cache, `PGKVStorage` for production runs.
- **Vector** — `PGVectorStorage` with `pgvector`, embedding model
  `microsoft/harrier-oss-v1-270m` (640-dim) or `BAAI/bge-m3` (1024-dim).
- **Graph** — `NetworkXStorage` (GraphML on disk) for development,
  `PGGraphStorage` for production.
- **Doc status** — `PGDocStatusStorage` tracks per-document processing
  state so partial ingests resume cleanly.

`workspace` parameter gives every LightRAG instance isolated KV /
vector / status namespaces; graph storage uses collection-name
prefixes.

---

## Operational notes

- **9router** is the local LLM proxy (`http://localhost:20128/v1`)
  used by both the server and `eval/llm_call.py` so BM25 and LightRAG
  go through the same provider path.
- **Rate limits** — `eval/run_benchmark.py` defaults to a 2 s throttle
  between calls to stay under NIM 40 RPM.
- **Eval results layout** — per-type artifacts land in
  `eval/results/{type}/` (`results_raw.json`, `results_eval.json`,
  `results_chunks.json`, `results_pairwise.json`).
- **Offline install** — see `LightRAG/docs/OfflineDeployment.md` for
  pre-caching model weights and Python wheels for air-gapped deploys.

---

## References

- Guo et al., *LightRAG: Simple and Fast Retrieval-Augmented Generation*
  (arXiv 2410.05779).
- Edge et al., *Personalized PageRank* — used for PPR refinement in
  `graph` mode.
- See `docs/ppr_convergence.md` for the formal proof that the L1-norm
  stop condition decreases geometrically under the column-stochastic
  transition matrix.
