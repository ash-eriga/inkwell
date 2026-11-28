#!/usr/bin/env bash
# usage inside gpu-run: run_select.sh <binary> <tag> <ctx> <fmt...>; three repetitions per format
set -u
BIN=$1; TAG=$2; CTX=$3; shift 3
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/matrix-$TAG.txt
mkdir -p "$(dirname "$RES")"
for fmt in "$@"; do
  for rep in 1 2 3; do
    printf 'ctx=%s fmt=%s rep=%s ' "$CTX" "$fmt" "$rep" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" "$CTX" "m$TAG-$fmt" INKWELL_KV=$fmt 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
