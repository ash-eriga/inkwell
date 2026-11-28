#!/usr/bin/env bash
# warm speed matrix for the final v13 binary; contexts passed as args.
set -u
ROOT=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38
BIN=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v13-ldg
for CTX in "$@"; do
RES=$ROOT/opt/kvq64/artifacts/matrix-warm$CTX-v13.txt
[ -s "$RES" ] || : >"$RES"
for round in 1 2 3; do
  for fmt in f16 int8 fp8 k8v4 k4v2; do
    grep -q "round=$round ctx=$CTX fmt=$fmt decode_rc=0" "$RES" 2>/dev/null && continue
    printf 'round=%s ctx=%s fmt=%s ' "$round" "$CTX" "$fmt" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" "$CTX" "mv2-$fmt" INKWELL_KV=$fmt INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
done
