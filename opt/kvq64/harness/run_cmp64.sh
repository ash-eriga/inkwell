#!/usr/bin/env bash
# usage inside gpu-run: compare v2/v3/v4 at 64K for selected formats, three repetitions each
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64.txt
: >"$RES"
for spec in v2:inkwell-kvq-v2 v3:inkwell-kvq-v3-maskzero v4:inkwell-kvq-v4-compact; do
  tag=${spec%%:*}; bin=/home/ash/.cache/inkwell-kv64k/bin/${spec#*:}
  for fmt in "$@"; do
    for rep in 1 2 3; do
      printf 'bin=%s fmt=%s rep=%s ' "$tag" "$fmt" "$rep" | tee -a "$RES"
      "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$bin" 65536 "mv2-$fmt" INKWELL_KV=$fmt 2>&1 | tail -n 1 | tee -a "$RES"
    done
  done
done
