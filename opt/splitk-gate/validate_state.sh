#!/bin/bash
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
P=$(cat "$ART/prompts/p1024.txt")
STATE="$ART/state1024.bin"
"$ROOT/opt/splitk-gate/run_infer.sh" state1024-save \
  INKWELL_MAX_CTX=1152 INKWELL_GENS=16 INKWELL_PROMPT="$P" \
  INKWELL_STATE_SAVE="$STATE"
"$ROOT/opt/splitk-gate/run_infer.sh" state1024-load \
  INKWELL_MAX_CTX=1152 INKWELL_GENS=16 INKWELL_PROMPT="$P" \
  INKWELL_STATE_LOAD="$STATE"
grep 'generated text' "$ART/logs/state1024-save.log" | md5sum | cut -c1-8 > "$ART/results/state1024-save.md5"
grep 'generated text' "$ART/logs/state1024-load.log" | md5sum | cut -c1-8 > "$ART/results/state1024-load.md5"
