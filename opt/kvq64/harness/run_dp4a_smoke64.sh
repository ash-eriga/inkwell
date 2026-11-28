#!/usr/bin/env bash
# One-state smoke for dp4a decode attention modes. Run inside one gpu-run job.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-dp4a1}
echo '== off =='
"$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 dp4aoff-smoke INKWELL_WARM_GENS=16 || echo off_rc=$?
echo '== dp4a1 =='
"$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 dp4a1-smoke INKWELL_WARM_GENS=16 INKWELL_KVQ_DP4A=1 || echo dp4a1_rc=$?
echo '== dp4a2 =='
"$ROOT/opt/kvq64/harness/measure_ctx_state.sh" "$BIN" 65536 k8v4 dp4a2-smoke INKWELL_WARM_GENS=16 INKWELL_KVQ_DP4A=2 || echo dp4a2_rc=$?
