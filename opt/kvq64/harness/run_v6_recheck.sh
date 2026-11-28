#!/usr/bin/env bash
# Recheck K8V6 with the completed K10-era binary; v6b may have been contaminated by incomplete K10 scaffolding.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-k10
echo '== k8v6 ctx512 smoke on k10 binary =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 k8v6k10bin768 INKWELL_KV=k8v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 || { echo smoke_rc=$?; exit 1; }
echo '== k8v6 4k quality on k10 binary =='
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 v6k10bin k8v6
