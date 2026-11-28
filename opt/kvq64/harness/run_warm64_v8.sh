#!/usr/bin/env bash
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-warm64-v8.txt
: >"$RES"
BIN=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v8-warmsext
for round in 1 2 3; do
  for fmt in int8 k8v4; do
    printf 'round=%s bin=v8warm fmt=%s ' "$round" "$fmt" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 65536 "mv2-$fmt" INKWELL_KV=$fmt INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
