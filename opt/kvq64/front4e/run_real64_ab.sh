#!/usr/bin/env bash
set -u
CTLBIN=/home/ash/.cache/inkwell-front3d/bin/inkwell-f3d-cand
CANDBIN=/home/ash/.cache/inkwell-front4e/bin/inkwell-f4e-cand
OUT=/home/ash/.cache/inkwell-front4e/artifacts/real64-ab.txt
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
for round in 1 2; do
  for kind in ctl cand cand ctl; do
    tag=f4e-r${round}-${kind}
    if [ "$kind" = ctl ]; then
      bin=$CTLBIN; state=/home/ash/.cache/inkwell-real-states/real-k8v4-ctx65536/state.bin
    else
      bin=$CANDBIN; state=/home/ash/.cache/inkwell-real-states/real-k8v4-hm-ctx65536/state.bin
    fi
    printf '%s ' "$tag" | tee -a "$OUT"
    /home/ash/inkwell/opt/tools/gpu-run -p low -t "$tag" -- \
      opt/kvq64/front2g/measure_real64.sh "$bin" "$tag" INKWELL_STATE_LOAD="$state" 2>&1 | tail -n 1 | tee -a "$OUT"
  done
done
