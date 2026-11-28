#!/usr/bin/env bash
set -u
CTX=$1
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
BIN=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v8-warmsext
RES=$ROOT/opt/kvq64/artifacts/matrix-warm$CTX-v8.txt
: >"$RES"
for round in 1 2 3; do
  for fmt in f16 int8 fp8 k8v4 k4v2; do
    printf 'round=%s ctx=%s fmt=%s ' "$round" "$CTX" "$fmt" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" "$CTX" "mv2-$fmt" INKWELL_KV=$fmt INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
