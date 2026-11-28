#!/usr/bin/env bash
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
RES=$ROOT/opt/kvq64/artifacts/chunk64-k8v4.txt
: >"$RES"
BIN=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v4-compact
for round in 1 2 3; do
  for chunk in 128 256 512; do
    printf 'round=%s chunk=%s ' "$round" "$chunk" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 65536 chunk64-k8v4 INKWELL_KV=k8v4 INKWELL_ATTNSPLIT_CHUNK=$chunk 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
