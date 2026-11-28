#!/usr/bin/env bash
# Real 4K teacher-forced quality for K10V6 query-dp4a modes.  Controls are
# format-independent hard links from the completed K10 mode3 run.  Mode2 is the
# two-term residual query path; mode4 is the four-channel grouped path.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4}
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
setup() {
  local dst=$1
  mkdir -p "$dst"
  for f in ref.bin repeat.bin noise64.bin teacher.ids; do
    if [ -s "$SRC/$f" ] && [ ! -e "$dst/$f" ]; then ln "$SRC/$f" "$dst/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
  done
}
setup "$ART/quality-k10m2-ctx4096"
echo '== K10V6 4K quality dp4a mode2 residual query =='
env INKWELL_KVQ_DP4A=2 "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 k10m2 k10v6 || exit $?
setup "$ART/quality-k10m4-ctx4096"
echo '== K10V6 4K quality dp4a mode4 grouped4 query =='
env INKWELL_KVQ_DP4A=4 "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 k10m4 k10v6 || exit $?
