#!/usr/bin/env bash
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64-v67.txt
: >"$RES"
for round in 1 2 3; do
  for spec in v6:inkwell-kvq-v6-pipelined v7:inkwell-kvq-v7-sext; do
    tag=${spec%%:*}; bin=/home/ash/.cache/inkwell-kv64k/bin/${spec#*:}
    for fmt in int8 k8v4 k4v2; do
      printf 'round=%s bin=%s fmt=%s ' "$round" "$tag" "$fmt" | tee -a "$RES"
      "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$bin" 65536 "mv2-$fmt" INKWELL_KV=$fmt 2>&1 | tail -n 1 | tee -a "$RES"
    done
  done
done
