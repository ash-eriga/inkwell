#!/usr/bin/env bash
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64-v24.txt
: >"$RES"
for round in 1 2 3; do
  for spec in v2:inkwell-kvq-v2 v4:inkwell-kvq-v4-compact; do
    tag=${spec%%:*}; bin=/home/ash/.cache/inkwell-kv64k/bin/${spec#*:}
    for fmt in "$@"; do
      printf 'round=%s bin=%s fmt=%s ' "$round" "$tag" "$fmt" | tee -a "$RES"
      "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$bin" 65536 "mv2-$fmt" INKWELL_KV=$fmt 2>&1 | tail -n 1 | tee -a "$RES"
    done
  done
done
