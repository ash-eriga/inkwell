#!/usr/bin/env bash
set -u
BIN=$1
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$ROOT/opt/kvq64/front3d/artifacts/quality
mkdir -p "$OUT"
CTX=4096
PROMPT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/prompts/p$CTX.ids
export GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=$((CTX+256)) INKWELL_PROMPT_IDS=$PROMPT INKWELL_GENS=64 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
runone() {
  local name=$1 scope=$2; shift 2
  [ -s "$OUT/$name.bin" ] && return 0
  env INKWELL_KV=k8v4 INKWELL_KV_SCOPE=$scope INKWELL_KVQ_DP4A=2 INKWELL_FORCE_IDS=$OUT/teacher.ids INKWELL_LOGIT_DUMP=$OUT/$name.bin "$@" \
    "$BIN" "$HOME/models/inkling" --run >"$OUT/$name.log" 2>&1
  echo "$name rc=$?"
}
if [ ! -s "$OUT/ref.bin" ]; then
  env INKWELL_KV=f16 INKWELL_LOGIT_DUMP=$OUT/ref.bin INKWELL_GEN_IDS_DUMP=$OUT/teacher.ids \
    "$BIN" "$HOME/models/inkling" --run >"$OUT/ref.log" 2>&1
  echo "ref rc=$?"
fi
runone ctl-all all INKWELL_NO_KVQ_GQA3=1
runone cand-all all
runone cand-full full
for f in ctl-all cand-all cand-full; do
  printf '%s-vs-ref ' "$f"
  systemd-run --user --scope -q -p MemoryMax=4G -- uv run --with numpy python3 "$ROOT/opt/kvq64/analysis/compare_quality.py" "$OUT/ref.bin" "$OUT/$f.bin" "$OUT/teacher.ids" || true
done
printf 'cand-all-vs-ctl-all '
python3 "$ROOT/opt/kvq64/analysis/compare_quality.py" "$OUT/ctl-all.bin" "$OUT/cand-all.bin" "$OUT/teacher.ids" || true
