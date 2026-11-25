#!/usr/bin/env bash
# usage: run_case.sh <bin> <ctx> <prompt.ids> <force.ids> <outdir> <serial|prefill> [chunk]
set -euo pipefail
BIN=$1; CTX=$2; PROMPT=$3; FORCE=$4; OUT=$5; MODE=$6; CHUNK=${7:-128}
mkdir -p "$OUT"
EXTRA=()
if [[ "$MODE" == serial ]]; then EXTRA+=(INKWELL_NO_PREFILL=1); elif [[ "$MODE" == prefill ]]; then EXTRA+=(INKWELL_PREFILL_CHUNK=$CHUNK); else echo 'mode must be serial|prefill' >&2; exit 2; fi
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -lms 200 >"$OUT/mem" 2>/dev/null & sp=$!
set +e
env GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=$((CTX+256)) INKWELL_CACHE_MARGIN_MB=10 \
  INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
  INKWELL_PROMPT_IDS="$PROMPT" INKWELL_FORCE_IDS="$FORCE" INKWELL_GENS=32 \
  INKWELL_LOGIT_DUMP="$OUT/logits.bin" INKWELL_GEN_IDS_DUMP="$OUT/gen.ids" \
  "${EXTRA[@]}" LD_LIBRARY_PATH="$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin" \
  "$BIN" "$HOME/models/inkling" --run >"$OUT/log" 2>&1
rc=$?
set -e
kill $sp 2>/dev/null || true
peak=$(grep -v '^$' "$OUT/mem" 2>/dev/null | awk -F', ' '{v=$2; sub(/ MiB/,"",v); v+=0; if(v>m)m=v} END{print m+0}')
echo "$rc" >"$OUT/rc"; echo "$peak" >"$OUT/peak_mib"
echo "mode=$MODE ctx=$CTX rc=$rc peak=$peak $(grep -m1 'encode .* tok in' "$OUT" || true)"
exit "$rc"
