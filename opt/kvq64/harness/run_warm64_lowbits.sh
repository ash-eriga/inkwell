#!/usr/bin/env bash
# 64K warm rows for int8/fp8 on v13 (k8v4/k4v2 already validated interleaved).
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
BIN=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v13-ldg
CTX=65536
RES=$ROOT/opt/kvq64/artifacts/matrix-warm64-v13-lowbits.txt
[ -s "$RES" ] || : >"$RES"
for round in 1 2 3; do
  for fmt in int8 fp8 k8v4; do
    grep -q "round=$round ctx=$CTX fmt=$fmt decode_rc=0" "$RES" 2>/dev/null && continue
    printf 'round=%s ctx=%s fmt=%s ' "$round" "$CTX" "$fmt" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" "$CTX" "mv2-$fmt" INKWELL_KV=$fmt INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
