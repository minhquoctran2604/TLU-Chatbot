#!/bin/bash
# Graph-mode ablation on the 2-hop set: 5 variants -> one 5-way judging pass.
# Each variant runs on its own server (port 9631) with env overrides; .env stays untouched.
# Usage: bash eval/run_ablation.sh [queries_file]   (resumable: re-run skips finished variants)
set -euo pipefail
cd "$(dirname "$0")/.."

Q=${1:-eval/queries_2hop.json}
OUT=eval/ablation
PORT=9631
mkdir -p "$OUT"

# Same generation model for ALL variants, no failover (mixing models would confound the ablation).
# If Gemma is rate-limited, change GEN_MODEL here and re-run every variant.
GEN_MODEL=${GEN_MODEL:-nvidia/google/gemma-3n-e4b-it}
COMMON="ENABLE_LLM_CACHE=false LLM_MODEL=$GEN_MODEL LLM_FAILOVER_MODELS="

declare -A VARIANTS=(
  [full]=""
  [bfs_only]="GRAPH_FLOW_MAX_ITER=0"
  [no_flow_rank]="GRAPH_FLOW_ENTITY_LAMBDA=0"
  [no_structure]="GRAPH_FLOW_MAX_ITER=0 GRAPH_FLOW_ENTITY_LAMBDA=0"
  [depth1]="GRAPH_FLOW_MAX_DEPTH=1"
)

for v in full bfs_only no_flow_rank no_structure depth1; do
  [ -f "$OUT/$v.done" ] && { echo "[skip] $v"; continue; }
  echo "[run] $v: ${VARIANTS[$v]:-defaults}"
  env $COMMON ${VARIANTS[$v]} lightrag-server --port $PORT > "$OUT/$v.server.log" 2>&1 &
  pid=$!
  until curl -sf "http://localhost:$PORT/health" >/dev/null; do sleep 5; done
  python eval/run_benchmark.py --modes graph --queries "$Q" --out "$OUT/$v.json" \
    --server-url "http://localhost:$PORT" --resume
  kill $pid; wait $pid 2>/dev/null || true
  touch "$OUT/$v.done"
done

python eval/ablation_report.py merge
python eval/evaluate_pairwise.py --eval "$OUT/merged.json" --out "$OUT/judged.json" --resume
python eval/ablation_report.py report
