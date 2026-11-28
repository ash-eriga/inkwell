#!/usr/bin/env bash
# interleaved warm 64K control/rowptr comparison for K8V4
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
H=$ROOT/opt/kvq64/harness/measure_ctx_state.sh
CTL=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v13-ldg
CAND=/home/ash/.cache/inkwell-kv64k2/bin/inkwell-kvq-v14-rowptr
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64-rowptr.txt
[ -s "$RES" ] || : >"$RES"
for round in 1 2 3; do
  for arm in ctl cand; do
    grep -q "round=$round arm=$arm " "$RES" 2>/dev/null && continue
    bin=$CTL; tag=cmp-v13-k8v4
    [ "$arm" = cand ] && bin=$CAND && tag=cmp-v14rowptr-k8v4
    printf 'round=%s arm=%s ' "$round" "$arm" | tee -a "$RES"
    "$H" "$bin" 65536 k8v4 "$tag-r$round" INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
