#!/bin/bash
# =============================================================================
# LightRAG Benchmark Pipeline — 1-click runner
# Run all 5 modes × queries, eval with BERTScore + RAGAS, generate report.
#
# Usage:
#   cd ~/LightRAG
#   bash eval/run_all.sh                              # full pipeline (default queries)
#   bash eval/run_all.sh --queries my_queries.json    # custom query file
#   bash eval/run_all.sh --skip-bench                 # only re-eval existing results_raw.json
#   bash eval/run_all.sh --skip-ragas                 # skip slow RAGAS step
#   bash eval/run_all.sh --modes naive,hybrid         # subset of modes
#
# Prerequisites:
#   - .env configured (see eval/.env.template)
#   - LightRAG server running on port 9621 (background)
#   - tlu_workspace/ synced (contains graph_chunk_entity_relation.graphml)
#   - Python venv activated with deps installed
# =============================================================================

set -e  # exit on error
set -u  # unset var = error
set -o pipefail

# ---- Config ----
QUERIES_FILE="eval/benchmark_queries.json"
MODES="bm25,naive,hybrid,mix,graph"
SKIP_BENCH=false
SKIP_RAGAS=false
SERVER_URL="http://localhost:9621"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --queries)        QUERIES_FILE="$2"; shift 2 ;;
        --modes)          MODES="$2"; shift 2 ;;
        --skip-bench)     SKIP_BENCH=true; shift ;;
        --skip-ragas)     SKIP_RAGAS=true; shift ;;
        --server-url)     SERVER_URL="$2"; shift 2 ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Pre-flight checks ----
echo "============================================================"
echo "  LightRAG Benchmark Pipeline"
echo "============================================================"
echo "Queries:     $QUERIES_FILE"
echo "Modes:       $MODES"
echo "Server URL:  $SERVER_URL"
echo "Skip bench:  $SKIP_BENCH"
echo "Skip RAGAS:  $SKIP_RAGAS"
echo "Time:        $(date)"
echo "============================================================"
echo ""

# Check venv
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[WARN] No venv activated. Activate with: source venv/bin/activate"
fi

# Check queries file
if [ ! -f "$QUERIES_FILE" ]; then
    echo "[FATAL] Queries file not found: $QUERIES_FILE"
    exit 1
fi

# Check .env
if [ ! -f ".env" ]; then
    echo "[FATAL] .env not found. Copy from eval/.env.template and fill in keys."
    exit 1
fi

# Check server alive (unless skipping bench)
if [ "$SKIP_BENCH" = false ]; then
    if ! curl -s -o /dev/null --max-time 5 "$SERVER_URL/health"; then
        echo "[FATAL] LightRAG server not responding at $SERVER_URL"
        echo "  Start it with:"
        echo "    cd ~/LightRAG"
        echo "    source venv/bin/activate"
        echo "    nohup python -u -m lightrag.api.lightrag_server > server.log 2>&1 &"
        exit 1
    fi
    echo "[OK] Server alive at $SERVER_URL"
fi

# ---- Step 1: Run benchmark (queries × modes) ----
if [ "$SKIP_BENCH" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Step 1/4: Run benchmark — $MODES on $(jq '.queries | length' $QUERIES_FILE) queries"
    echo "============================================================"
    # NOTE: run_benchmark.py reads queries from eval/benchmark_queries.json (hardcoded)
    # and writes to eval/results_raw.json. Override QUERIES_FILE by copying first.
    if [ "$QUERIES_FILE" != "eval/benchmark_queries.json" ]; then
        echo "[INFO] Copying $QUERIES_FILE -> eval/benchmark_queries.json (override)"
        cp "$QUERIES_FILE" eval/benchmark_queries.json
    fi
    cd eval
    python run_benchmark.py --modes "$MODES" 2>&1 | tee run_benchmark.log
    cd ..
    echo "[OK] Step 1 done."
else
    echo "[SKIP] Step 1 (benchmark) skipped."
fi

# ---- Step 2: BERTScore eval ----
echo ""
echo "============================================================"
echo "  Step 2/4: BERTScore evaluation"
echo "============================================================"
python eval/evaluate_benchmark.py 2>&1 | tee eval/evaluate_benchmark.log
echo "[OK] Step 2 done. → eval/results_eval.json + eval/report.md"

# ---- Step 3: Fetch chunks for RAGAS ----
if [ "$SKIP_RAGAS" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Step 3a/4: Fetch chunks per (query, mode) for RAGAS"
    echo "============================================================"
    python eval/fetch_chunks.py 2>&1 | tee eval/fetch_chunks.log
    echo "[OK] Step 3a done. → eval/results_chunks.json"

    # ---- Step 4: RAGAS eval ----
    echo ""
    echo "============================================================"
    echo "  Step 3b/4: RAGAS 4 metrics (faithfulness, ans_relevancy, ans_correctness, context_precision)"
    echo "============================================================"
    python eval/evaluate_ragas.py 2>&1 | tee eval/evaluate_ragas.log
    echo "[OK] Step 3b done. → eval/results_ragas.json"

    # ---- Step 5: Retry failed RAGAS entries (faithfulness only — most failure-prone) ----
    echo ""
    echo "============================================================"
    echo "  Step 3c/4: Retry failed RAGAS entries (faithfulness)"
    echo "============================================================"
    python eval/retry_failed_ragas.py --metric faithfulness 2>&1 | tee eval/retry_faithfulness.log
    echo "[OK] Step 3c done."
else
    echo "[SKIP] Step 3 (RAGAS) skipped."
fi

# ---- Summary ----
echo ""
echo "============================================================"
echo "  Pipeline complete @ $(date)"
echo "============================================================"
echo ""
echo "Output files:"
ls -lh eval/results_*.json eval/report.md 2>/dev/null || true
echo ""
echo "Logs:"
ls -lh eval/*.log 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Review eval/report.md"
echo "  2. scp results back to local:"
echo "     scp tts@<server>:~/LightRAG/eval/results_*.json ./"
echo "     scp tts@<server>:~/LightRAG/eval/report.md ./"
