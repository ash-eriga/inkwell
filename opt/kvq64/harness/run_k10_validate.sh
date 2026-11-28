#!/usr/bin/env bash
# K10V6 correctness smoke plus full 4K teacher-forced quality.
# The smoke state is dp4a-independent; dp4a modes only change decode attention arithmetic.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-k10}
echo '== k10v6 ctx512 smoke, dp4a off =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 k10smoke768 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 || { echo smoke_off_rc=$?; exit 1; }
echo '== k10v6 ctx512 smoke, grouped dp4a =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 k10smoke768 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 INKWELL_KVQ_DP4A=3 || { echo smoke_qg_rc=$?; exit 1; }
echo '== k10v6 4k quality, grouped dp4a =='
export INKWELL_KVQ_DP4A=3
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 k10qg k10v6
