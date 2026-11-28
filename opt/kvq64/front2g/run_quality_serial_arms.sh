#!/usr/bin/env bash
# usage inside gpu-run: run_quality_serial_arms.sh <binary> <ctx> <tag> <control-tag> [fmt:scope ...]
set -u
BIN=$1; CTX=$2; TAG=$3; CONTROLS=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$ROOT/opt/kvq64/front2g/artifacts/quality-serial-$TAG-ctx$CTX
CTL=$ROOT/opt/kvq64/front2g/artifacts/quality-serial-$CONTROLS-ctx$CTX
mkdir -p "$OUT"
for f in ref.bin ref.log repeat.bin repeat.log noise64.bin noise64.log teacher.ids; do
  [ -e "$OUT/$f" ] || cp "$CTL/$f" "$OUT/$f"
done
PROMPT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/prompts/p$CTX.ids
IDSZ=$(stat -c%s "$PROMPT")
N_PROMPT=$((IDSZ/4))
export GGML_PASCAL_GRAPHS=1 INKWELL_NO_PREFILL=1 INKWELL_MAX_CTX=$((CTX+256)) INKWELL_PROMPT_IDS=$PROMPT \
  INKWELL_GENS=64 INKWELL_FORCE_IDS=$OUT/teacher.ids INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 INKWELL_CACHE_MARGIN_MB=10 INKWELL_KVQ_DP4A=2
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
runone() {
  local name=$1 fmt=$2 scope=$3; shift 3
  INKWELL_KV=$fmt INKWELL_KV_SCOPE=$scope INKWELL_LOGIT_DUMP=$OUT/$name.bin "$@" \
    "$BIN" "$HOME/models/inkling" --run > "$OUT/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc"
  systemd-run --user --scope -q -p MemoryMax=4G -- uv run --with numpy python3 "$ROOT/opt/kvq64/analysis/compare_quality.py" "$OUT/ref.bin" "$OUT/$name.bin" "$OUT/teacher.ids" || true
}
echo "prompt_ids=$PROMPT n_prompt=$N_PROMPT"
echo "ref(existing)"
echo "repeat(existing)"
echo "noise64(existing)"
for spec in "$@"; do
  fmt=${spec%%:*}; scope=${spec##*:}; name=$(echo "$spec" | tr ':/' '-')
  runone "$name" "$fmt" "$scope"
done
