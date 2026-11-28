#!/usr/bin/env bash
# One 512-token INKWELL_PF_INT8=1 encode under nsys. Run through gpu-run.
set -uo pipefail
BIN=$1; OUT=$2
rm -rf "$OUT"; mkdir -p "$OUT"
P=/home/ash/.cache/inkwell-prefill-bench/p513.txt.ids
F=/home/ash/.cache/inkwell-prefill-bench/f513.ids
NSYS=/home/ash/lfm-gtx1060-bench/cuda-12.6/bin/nsys
"$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
  -o "$OUT/report" --force-overwrite true \
  env GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=768 INKWELL_CACHE_MARGIN_MB=10 \
    INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
    INKWELL_PROMPT_IDS="$P" INKWELL_FORCE_IDS="$F" INKWELL_GENS=0 \
    INKWELL_PREFILL_CHUNK=512 INKWELL_PF_INT8=1 \
    LD_LIBRARY_PATH="$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin" \
    "$BIN" "$HOME/models/inkling" --run >"$OUT/log" 2>&1
rc=$?
if [ -f "$OUT/report.nsys-rep" ]; then
  "$NSYS" export --type sqlite -o "$OUT/report.sqlite" --force-overwrite true "$OUT/report.nsys-rep" >>"$OUT/log" 2>&1
  python3 opt/tools/prefill/nsys_prefill_phase.py "$OUT/report.sqlite" | tee "$OUT/phase.txt"
fi
exit "$rc"
