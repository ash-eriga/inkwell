#!/usr/bin/env bash
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64-v1011.txt
: >"$RES"
for round in 1 2 3; do
  for spec in v10:inkwell-kvq-v10-qsplit v11:inkwell-kvq-v11-qpair; do
    tag=${spec%%:*}; bin=/home/ash/.cache/inkwell-kv64k/bin/${spec#*:}
    printf 'round=%s bin=%s fmt=k8v4 ' "$round" "$tag" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$bin" 65536 "mv2-k8v4" INKWELL_KV=k8v4 INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
