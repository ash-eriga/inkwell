#!/bin/bash
# Full-vocabulary KLD/top-1 comparisons for quality runs.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
OUT="$ART/results/logit_divergence.txt"
: > "$OUT"
for ctx in 4096 8192; do
  for pair in "old sdef" "sdef c64"; do
    set -- $pair
    a=$1; b=$2
    echo "== ctx=$ctx $a-vs-$b ==" | tee -a "$OUT"
    "$ROOT/opt/splitk-gate/bin/compare_logits" \
      "$ART/logits/q$ctx-$a.bin" "$ART/logits/q$ctx-$b.bin" \
      "$ART/results/q$ctx-$a-vs-$b-rows.tsv" 2>&1 | tee -a "$OUT"
  done
done
