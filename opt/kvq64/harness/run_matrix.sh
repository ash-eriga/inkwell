#!/usr/bin/env bash
# usage inside gpu-run: run_matrix.sh <binary> <tag> [ctxs...] ; formats fixed
set -u
BIN=$1; TAG=$2; shift 2
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
CTXS=${@:-"4096 16384 32768 65536"}
RES=$ROOT/opt/kvq64/artifacts/matrix-$TAG.txt
mkdir -p "$(dirname "$RES")"
: >"$RES"
for ctx in $CTXS; do
  for fmt in f16 int8 fp8 k8v4 k4v2; do
    for rep in 1 2 3; do
      printf 'ctx=%s fmt=%s rep=%s ' "$ctx" "$fmt" "$rep" | tee -a "$RES"
      "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" "$ctx" "m$TAG-$fmt" INKWELL_KV=$fmt 2>&1 | tail -n 1 | tee -a "$RES"
    done
  done
done
