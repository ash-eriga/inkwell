#!/usr/bin/env bash
# Interleaved real64 A/B. usage: real64_ab.sh <rounds> <tag> <ctlbin> <candbin> [ctlenv...] -- [candenv...]
set -u
R=${1:-2}; TAG=$2; CTL=$3; CAND=$4; shift 4
CTLE=(); CANDE=(); iscand=0
for x in "$@"; do
  if [ "$x" = "--" ]; then iscand=1; continue; fi
  if [ "$iscand" = 0 ]; then CTLE+=("$x"); else CANDE+=("$x"); fi
done
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=/home/ash/.cache/inkwell-front4b/logs/$TAG-ab.txt
: > "$OUT"
for r in $(seq 1 "$R"); do
  for kind in ctl cand cand ctl; do
    if [ "$kind" = ctl ]; then bin=$CTL; envs=("${CTLE[@]}"); else bin=$CAND; envs=("${CANDE[@]}"); fi
    echo "run round=$r kind=$kind ${envs[*]}" >> "$OUT"
    (cd "$ROOT" && opt/tools/gpu-run -p low -t "$TAG-$r-$kind" -- opt/kvq64/front2g/measure_real64.sh "$bin" "$TAG-r$r-$kind" "${envs[@]}") \
      | sed "s/^/round=$r kind=$kind /" >> "$OUT"
  done
done
cat "$OUT"
