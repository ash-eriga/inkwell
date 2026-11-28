#!/usr/bin/env bash
# K10V6 4K quality with query dp4a disabled, reusing the complete F16 controls
# from the k10qg run.  Controls are format-independent.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-k10}
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-k10off-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
exec env -u INKWELL_KVQ_DP4A "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 k10off k10v6
