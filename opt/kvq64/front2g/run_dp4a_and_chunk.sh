#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/front2g/dp4a-chunk-real64.txt
BIN=/home/ash/.cache/inkwell-front2d/bin/inkwell-f2d-final
: > "$RES"
for spec in dp4a1:INKWELL_KVQ_DP4A=1 dp4a2:INKWELL_KVQ_DP4A=2 chunk128:INKWELL_ATTNSPLIT_CHUNK=128 chunk256:INKWELL_ATTNSPLIT_CHUNK=256 chunk512:INKWELL_ATTNSPLIT_CHUNK=512; do
  name=${spec%%:*}; envv=${spec#*:}; tag="f2g-${name}-real64"
  printf '%s ' "$name" | tee -a "$RES"
  /home/ash/inkwell/opt/tools/gpu-run -p normal -t "$tag" -- \
    "$ROOT/opt/kvq64/front2g/measure_real64.sh" "$BIN" "$tag" "$envv" 2>&1 | tail -n 1 | tee -a "$RES"
done
