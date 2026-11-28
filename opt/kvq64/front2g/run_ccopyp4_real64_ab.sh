#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/front2g/ccopyp4-real64-ab.txt
CTL=/home/ash/.cache/inkwell-front2d/bin/inkwell-f2d-final
CAND=/home/ash/.cache/inkwell-front2g/bin/inkwell-f2g-ccopy
: > "$RES"
for round in 1 2; do
  for kind in ctl cand cand ctl; do
    bin=$CTL; extra=; [ "$kind" = cand ] && { bin=$CAND; extra=INKWELL_CCOPY_PARTS=4; }
    tag="f2g-ccopyp4-64-r${round}-${kind}"
    printf 'round=%s kind=%s ' "$round" "$kind" | tee -a "$RES"
    /home/ash/inkwell/opt/tools/gpu-run -p normal -t "$tag" -- \
      "$ROOT/opt/kvq64/front2g/measure_real64.sh" "$bin" "$tag" $extra 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
