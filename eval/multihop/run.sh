#!/bin/bash
# Multi-hop retrieval benchmark on HippoRAG's MuSiQue / 2Wiki subsets: index once, then all modes + graph ablation.
# Runs its own server on an isolated file-based workspace (eval/multihop/rag_<bench>); the TLU workspace and .env stay untouched.
# Usage: bash eval/multihop/run.sh eval/multihop/musique_n300      (resumable: re-run skips finished steps)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -d venv ] && source venv/bin/activate
API_ENV=${API_ENV:-eval/multihop/api.env}
[ -f "$API_ENV" ] && { set -a; source "$API_ENV"; set +a; }

BENCH=${1:?bench dir from prepare.py}
NAME=$(basename "$BENCH")
# provenance: code diff and model config this run used
git diff HEAD -- lightrag/operate.py > "$BENCH/operate.diff"; cp "$API_ENV" "$BENCH/api.env.used" 2>/dev/null || true
PORT=9641
URL="http://localhost:$PORT"
ISOLATE="WORKING_DIR=$PWD/eval/multihop/rag_$NAME WORKSPACE=mh_$NAME PORT=$PORT ENABLE_LLM_CACHE=true \
  LIGHTRAG_KV_STORAGE=JsonKVStorage LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage \
  LIGHTRAG_GRAPH_STORAGE=NetworkXStorage LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage"

declare -A VARIANTS=(
  [full]=""
  [bfs_only]="GRAPH_FLOW_MAX_ITER=0"
  [no_flow_rank]="GRAPH_FLOW_ENTITY_LAMBDA=0"
  [no_structure]="GRAPH_FLOW_MAX_ITER=0 GRAPH_FLOW_ENTITY_LAMBDA=0"
  [depth1]="GRAPH_FLOW_MAX_DEPTH=1"
  [full_norm]="GRAPH_FLOW_NORM=1"                      # same as full/bfs_only but v rescaled to max 1,
  [bfs_only_norm]="GRAPH_FLOW_MAX_ITER=0 GRAPH_FLOW_NORM=1"  # removing PPR's scale effect on lambda
)

for v in full bfs_only no_flow_rank no_structure depth1 full_norm bfs_only_norm; do
  [ -f "$BENCH/runs/$v.done" ] && { echo "[skip] $v"; continue; }
  echo "[run] $v: ${VARIANTS[$v]:-defaults}"
  env $ISOLATE ${VARIANTS[$v]} lightrag-server > "$BENCH/$v.server.log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null' EXIT
  until curl -sf "$URL/health" >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "server died, see $BENCH/$v.server.log"; exit 1; }
    sleep 5
  done
  if [ "$v" = full ]; then
    [ -f "$BENCH/index.done" ] || { python eval/multihop/index.py "$BENCH" --server-url "$URL"; touch "$BENCH/index.done"; }
    MODES=naive,hybrid,mix,graph
  else
    MODES=graph
  fi
  python eval/multihop/run.py "$BENCH" --tag $v --modes $MODES --server-url "$URL"
  kill $pid; wait $pid 2>/dev/null || true
  mkdir -p "$BENCH/runs" && touch "$BENCH/runs/$v.done"
done

python eval/multihop/report.py "$BENCH" | tee "$BENCH/report.txt"
