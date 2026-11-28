#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/front2g/final-ctx512-ab.txt
CTL=/home/ash/.cache/inkwell-front2d/bin/inkwell-f2d-final
CAND=/home/ash/.cache/inkwell-front2g/bin/inkwell-f2g-final-candidate
: > "$RES"
for round in 1 2 3 4 5 6 7 8; do
  for kind in ctl cand cand ctl; do
    [ "$kind" = ctl ] && bin=$CTL || bin=$CAND
    tag="f2g-final512-r${round}-${kind}"
    printf 'round=%s kind=%s ' "$round" "$kind" | tee -a "$RES"
    /home/ash/inkwell/opt/tools/gpu-run -p normal -t "$tag" -- \
      "$ROOT/opt/tools/benchq.sh" "$tag" "$bin" 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
