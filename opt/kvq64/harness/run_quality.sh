#!/usr/bin/env bash
# usage inside gpu-run: run_quality.sh <binary> <ctx> <tag> [formats...]
set -u
BIN=$1; CTX=$2; TAG=$3; shift 3
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
OUT=$ROOT/opt/kvq64/artifacts/quality-$TAG-ctx$CTX
mkdir -p "$OUT"
PROMPT=$ROOT/opt/kvq64/prompts/p$CTX.ids
export GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=$((CTX+256)) INKWELL_PROMPT_IDS=$PROMPT INKWELL_GENS=64 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
runone() { # name fmt extraenv...
  local name=$1 fmt=$2; shift 2
  [ -s "$OUT/$name.bin" ] && return 0
  env INKWELL_KV=$fmt INKWELL_FORCE_IDS=$OUT/teacher.ids INKWELL_LOGIT_DUMP=$OUT/$name.bin "$@" \
    "$BIN" "$HOME/models/inkling" --run >"$OUT/$name.log" 2>&1
  echo "$name rc=$?"
}
if [ ! -s "$OUT/ref.bin" ]; then
  env INKWELL_KV=f16 INKWELL_LOGIT_DUMP=$OUT/ref.bin INKWELL_GEN_IDS_DUMP=$OUT/teacher.ids \
    "$BIN" "$HOME/models/inkling" --run >"$OUT/ref.log" 2>&1
  echo "ref rc=$?"
fi
runone repeat f16
runone noise64 f16 INKWELL_ATTNSPLIT_CHUNK=64
for fmt in "$@"; do runone $fmt $fmt; done
for f in "$OUT"/*.bin; do
  n=$(basename "$f" .bin)
  [ "$n" = ref ] && continue
  printf '%s ' "$n"
  systemd-run --user --scope -q -p MemoryMax=4G -- uv run --with numpy python3 "$ROOT/opt/kvq64/analysis/compare_quality.py" "$OUT/ref.bin" "$f" "$OUT/teacher.ids" || true
done
