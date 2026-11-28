#!/usr/bin/env bash
set -u
BIN=/home/ash/.cache/inkwell-front4a/bin/inkwell-f4a-cand
OUT=opt/kvq64/front4a/real64-ab.txt
: > "$OUT"
KILL='INKWELL_NO_MIX_ZEROCAT=1 INKWELL_NO_QKVR_FUSE=1 INKWELL_NO_CONV_ADD=1'
# shellcheck disable=SC2086
for pair in 1 2; do
  for leg in ctl cand cand ctl; do
    tag="p${pair}-${leg}"
    if [ "$leg" = ctl ]; then
      opt/tools/gpu-run -p low -t "f4a64-$tag" -- opt/kvq64/front2g/measure_real64.sh "$BIN" "f4a64-$tag" $KILL >> "$OUT" 2>&1
    else
      opt/tools/gpu-run -p low -t "f4a64-$tag" -- opt/kvq64/front2g/measure_real64.sh "$BIN" "f4a64-$tag" >> "$OUT" 2>&1
    fi
  done
done
