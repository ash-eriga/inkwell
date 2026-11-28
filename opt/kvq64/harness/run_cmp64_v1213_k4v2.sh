#!/usr/bin/env bash
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64-v1213-k4v2.txt
: >"$RES"
for round in 1 2 3; do
  for spec in v12:inkwell-kvq-v12-hppipe v13:inkwell-kvq-v13-ldg; do
    tag=${spec%%:*}; bin=/home/ash/.cache/inkwell-kv64k/bin/${spec#*:}
    printf 'round=%s bin=%s fmt=k4v2 ' "$round" "$tag" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$bin" 65536 "mv2-k4v2" INKWELL_KV=k4v2 INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
