#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/front2g/ccopy-ctx512-ab.txt
CTL=${CTL:-/home/ash/.cache/inkwell-front2d/bin/inkwell-f2d-final}
CAND=${CAND:-/home/ash/.cache/inkwell-front2g/bin/inkwell-f2g-ccopy}
: > "$RES"
for round in 1 2 3 4; do
  for kind in ctl cand cand ctl; do
    bin=$CTL; [ "$kind" = cand ] && bin=$CAND
    tag="f2g-ccopy512-r${round}-${kind}"
    printf 'round=%s kind=%s ' "$round" "$kind" | tee -a "$RES"
    /home/ash/inkwell/opt/tools/gpu-run -p normal -t "$tag" -- \
      "$ROOT/opt/tools/benchq.sh" "$tag" "$bin" 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
