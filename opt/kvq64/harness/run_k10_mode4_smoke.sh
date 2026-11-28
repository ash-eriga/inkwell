#!/usr/bin/env bash
# Short K10V6 smoke for mode4 using a fresh cap768 state, then same-state mode4 decode.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4}
echo '== k10 mode4 ctx512/cap768 combined smoke =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 k10m4smoke768 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 INKWELL_KVQ_DP4A=4
