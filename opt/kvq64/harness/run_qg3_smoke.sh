#!/usr/bin/env bash
# Smoke the opt-in qg3+dp4a kernel at ctx512/cap768: fp32-q control, then dp4a2.
# Uses separate state tags because qg3 output may differ numerically.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-qg3}
echo '== qg3 dp4a0 fp32-q control =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 qg3ctrl768 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 INKWELL_KVQ_QG3=1 || { echo ctrl_rc=$?; exit 1; }
echo '== qg3 dp4a2 residual-q smoke =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 qg3dp2-768 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 INKWELL_KVQ_QG3=1 INKWELL_KVQ_DP4A=2 || { echo dp2_rc=$?; exit 1; }
echo '== qg3 dp4a4 grouped-q smoke =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 qg3dp4-768 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 INKWELL_KVQ_QG3=1 INKWELL_KVQ_DP4A=4 || { echo dp4_rc=$?; exit 1; }
