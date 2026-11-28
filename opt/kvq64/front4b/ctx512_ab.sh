#!/usr/bin/env bash
# Interleaved ctx512 A/B. usage: ctx512_ab.sh <rounds> <tag> <ctlbin> <candbin>
set -u
R=${1:-2}; TAG=$2; CTL=$3; CAND=$4
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=/home/ash/.cache/inkwell-front4b/logs/$TAG-ctx512-ab.txt
: > "$OUT"
for r in $(seq 1 "$R"); do
  for kind in ctl cand cand ctl; do
    bin=$CTL; [ "$kind" = cand ] && bin=$CAND
    tag="$TAG-r$r-$kind"
    (cd "$ROOT" && opt/tools/gpu-run -p normal -t "$tag" -- opt/tools/benchq.sh "$tag" "$bin") \
      | sed "s/^/round=$r kind=$kind /" >> "$OUT"
  done
done
cat "$OUT"
