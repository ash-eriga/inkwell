#!/usr/bin/env bash
set -u
BIN=${1:-/home/ash/.cache/inkwell-front4e/bin/inkwell-f4e-cand}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$ROOT/opt/kvq64/front4e/artifacts/quality
REFROOT=/home/ash/.cache/inkwell-front3a/artifacts/quality-f3actl-ctx4096
mkdir -p "$OUT"
CTX=4096
PROMPT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/prompts/p$CTX.ids
export GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=$((CTX+256)) INKWELL_PROMPT_IDS=$PROMPT INKWELL_GENS=64 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
runone() {
  local name=$1 scope=$2; shift 2
  [ -s "$OUT/$name.bin" ] && return 0
  /home/ash/inkwell/opt/tools/gpu-run -p low -t "f4e-q-$name" -- env INKWELL_KV=k8v4 INKWELL_KV_SCOPE=$scope INKWELL_KVQ_DP4A=2 INKWELL_FORCE_IDS=$REFROOT/teacher.ids INKWELL_LOGIT_DUMP=$OUT/$name.bin "$@" \
    "$BIN" "$HOME/models/inkling" --run >"$OUT/$name.log" 2>&1
  echo "$name rc=$?"
}
runone cand-all all
runone cand-full full
for f in cand-all cand-full; do
  printf '%s-vs-ref ' "$f"
  uv run --with numpy python3 "$ROOT/opt/kvq64/analysis/compare_quality.py" "$REFROOT/ref.bin" "$OUT/$f.bin" "$REFROOT/teacher.ids" || true
done
