#!/usr/bin/env bash
# Short-context smoke then full teacher-forced 4K quality for K8V6.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-v6b}
echo '== k8v6 ctx512 smoke =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 k8v6smoke768 INKWELL_KV=k8v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 || { echo smoke_rc=$?; exit 1; }
echo '== k8v6 4k quality =='
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 v6b k8v6
