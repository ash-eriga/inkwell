#!/usr/bin/env bash
# K10V6 4K quality with four-channel grouped dp4a (mode4), reusing controls.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4}
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-k10m4-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
exec env INKWELL_KVQ_DP4A=4 "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 k10m4 k10v6
