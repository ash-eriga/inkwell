#!/usr/bin/env bash
# Interleaved splitw/control vs tile16 64K K8V4+DP4A2 warm-state A/B.
# usage inside gpu-run: run_tile16_ab.sh <binary> [rounds]
set -u
BIN=$1
ROUNDS=${2:-1}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/artifacts/tile16-ab.txt
mkdir -p "$(dirname "$RES")"
: > "$RES"
runone() {
  local kind=$1 round=$2 rep=$3
  local tag="tile16ab-${kind}r${round}x${rep}"
  if [ "$kind" = ctl ]; then
    "$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 "$tag" \
      INKWELL_KVQ_DP4A=2 INKWELL_WARM_GENS=16 INKWELL_VRAM_CAP_MB=5180 2>&1 | tail -n 1
  else
    "$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 "$tag" \
      INKWELL_KVQ_DP4A=2 INKWELL_WARM_GENS=16 INKWELL_VRAM_CAP_MB=5180 \
      INKWELL_KVQ_TILE16=1 2>&1 | tail -n 1
  fi
}
for round in $(seq 1 "$ROUNDS"); do
  rep=0
  for kind in ctl cand cand ctl; do
    rep=$((rep+1))
    printf 'round=%s kind=%s ' "$round" "$kind" | tee -a "$RES"
    runone "$kind" "$round" "$rep" | tee -a "$RES"
  done
done
