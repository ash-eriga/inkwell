#!/usr/bin/env bash
# Recheck final-binary K8V6 correctness while reusing the F16 controls already
# generated for the K10 quality run.  Controls are format-independent.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-k10}
echo '== k8v6 ctx512 smoke on final binary =='
"$ROOT/opt/kvq64/harness/measure_ctx.sh" "$BIN" 512 k8v6final768 INKWELL_KV=k8v6 INKWELL_WARM_GENS=16 INKWELL_MAX_CTX=768 || { echo smoke_rc=$?; exit 1; }
echo '== k8v6 4k quality on final binary =='
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-v6final-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 v6final k8v6
