#!/usr/bin/env bash
# K10V6 grouped-dp4a teacher-forced quality at 16K. Run inside gpu-run.
set -u
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-k10}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
export INKWELL_KVQ_DP4A=3
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 16384 k10qg k10v6
