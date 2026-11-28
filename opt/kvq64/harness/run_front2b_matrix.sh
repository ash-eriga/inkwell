#!/usr/bin/env bash
# Front-2B champion speed matrix: K8V4+DP4A2, scope=all/full, warm states.
# usage inside gpu-run: run_front2b_matrix.sh <binary>
set -u
BIN=$1
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RES=$ROOT/opt/kvq64/artifacts/front2b-matrix.txt
mkdir -p "$(dirname "$RES")"
: > "$RES"
for ctx in 4096 16384 32768 65536; do
  for scope in all full; do
    printf 'ctx=%s fmt=k8v4 dp4a=2 scope=%s ' "$ctx" "$scope" | tee -a "$RES"
    "$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" "$ctx" k8v4 "f2bm-${scope}-${ctx}" \
      INKWELL_KVQ_DP4A=2 INKWELL_WARM_GENS=16 INKWELL_VRAM_CAP_MB=5180 \
      INKWELL_KV_SCOPE=$scope 2>&1 | tail -n 1 | tee -a "$RES"
  done
done
