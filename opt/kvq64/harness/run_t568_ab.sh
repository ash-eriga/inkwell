#!/usr/bin/env bash
# Interleaved T532/control vs T568/cache-reserve48 64K K8V4+DP4A2 warm-state A/B.
# Run inside opt/tools/gpu-run. Results append to opt/kvq64/artifacts/t568-ab.txt.
set -u
BIN=${1:-/home/ash/.cache/inkwell-front2/bin/inkwell-f2-swa256}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/artifacts/t568-ab.txt
mkdir -p "$(dirname "$RES")"
: > "$RES"
OV='0:8,1:48,2:40,4:32,5:12,9:64,13:56,15:84,18:32,19:36,20:32,21:40,22:44,23:40'
runone() {
  local kind=$1 round=$2
  local tag="t568ab-${kind}r${round}"
  if [ "$kind" = ctl ]; then
    "$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 "$tag" \
      INKWELL_KVQ_DP4A=2 INKWELL_WARM_GENS=16 INKWELL_VRAM_CAP_MB=5180 2>&1 | tail -n 1
  else
    "$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 "$tag" \
      INKWELL_KVQ_DP4A=2 INKWELL_WARM_GENS=16 INKWELL_VRAM_CAP_MB=5180 \
      INKWELL_RUNTIME_RESERVE_MB=48 "INKWELL_CACHE_MB_L=$OV" 2>&1 | tail -n 1
  fi
}
for round in 1 2; do
  for kind in ctl cand cand ctl; do
    printf 'round=%s kind=%s ' "$round" "$kind" | tee -a "$RES"
    runone "$kind" "$round" | tee -a "$RES"
  done
done
