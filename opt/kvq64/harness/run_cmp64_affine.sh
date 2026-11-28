#!/usr/bin/env bash
# interleaved warm 64K K8V4 comparison: v13 control, v19 kernel, v20 kernel+cache tables
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
H=$ROOT/opt/kvq64/harness/measure_ctx_state.sh
RES=$ROOT/opt/kvq64/artifacts/matrix-cmp64-affine.txt
[ -s "$RES" ] || : >"$RES"
for round in 1 2 3; do
  for arm in ctl v19 v20; do
    grep -q "round=$round arm=$arm " "$RES" 2>/dev/null && continue
    case "$arm" in
      ctl) bin=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v13-ldg; tag=cmp-v13-k8v4 ;;
      v19) bin=/home/ash/.cache/inkwell-kv64k2/bin/inkwell-kvq-v19-rescale; tag=cmp-v19rescale-k8v4 ;;
      v20) bin=/home/ash/.cache/inkwell-kv64k2/bin/inkwell-kvq-v20-cachetab; tag=cmp-v20cachetab-k8v4 ;;
    esac
    printf 'round=%s arm=%s ' "$round" "$arm" | tee -a "$RES"
    "$H" "$bin" 65536 k8v4 "$tag-r$round" INKWELL_WARM_GENS=16 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
