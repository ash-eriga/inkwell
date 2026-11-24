#!/bin/bash
# Interleaved A/B: N pairs of (ctl,cand), run order ctl,cand,cand,ctl,...
# ctl = INKWELL_NO_CACHEL=1 (uniform ck), cand = default per-layer table.
# Results: /tmp/inkwell-ab-<tag>.out with one RUN line per run.
set -u
N=${1:-8}; TAG=${2:-ab}
OUT=/tmp/inkwell-ab-$TAG.out
: > "$OUT"
order=""
for i in $(seq 1 "$N"); do
  if [ $((i % 2)) -eq 1 ]; then order="$order C A"; else order="$order A C"; fi
done
i=0
for k in $order; do
  i=$((i+1))
  if [ "$k" = C ]; then
    opt/tools/bench1.sh "${TAG}-c$i" INKWELL_NO_CACHEL=1 >> "$OUT" 2>&1
  else
    opt/tools/bench1.sh "${TAG}-a$i" >> "$OUT" 2>&1
  fi
  tail -1 "$OUT"
done
echo "AB DONE"
