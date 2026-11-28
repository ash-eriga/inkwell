#!/usr/bin/env bash
set -u
BIN=$1
OUT=/home/ash/.cache/inkwell-front3d/artifacts/real64-ab.txt
: > "$OUT"
for round in 1 2; do
  for kind in ctl cand cand ctl; do
    tag=f3d-r${round}-${kind}
    if [ "$kind" = ctl ]; then extra=INKWELL_NO_KVQ_GQA3=1; else extra=INKWELL_NO_KVQ_GQA3_X=1; fi
    printf '%s ' "$tag" | tee -a "$OUT"
    /home/ash/inkwell/opt/tools/gpu-run -p low -t "$tag" -- \
      opt/kvq64/front2g/measure_real64.sh "$BIN" "$tag" "$extra" 2>&1 | tail -n 1 | tee -a "$OUT"
  done
done
