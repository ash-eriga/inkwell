#!/usr/bin/env bash
# K8V6 real 4K quality with high-precision sink+local tokens (INKWELL_KV_HP).
# Tests whether preserving attention-sink and recent tokens in fp16 closes
# the real-KLD gap.  HP forces the fp32 attention path (dp4a incompatible).
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
setup() {
  local dst=$1
  mkdir -p "$dst"
  for f in ref.bin repeat.bin noise64.bin teacher.ids; do
    if [ -s "$SRC/$f" ] && [ ! -e "$dst/$f" ]; then ln "$SRC/$f" "$dst/$f" 2>/dev/null || cp "$SRC/$f" "$dst/$f"; fi
  done
}
setup "$ART/quality-v6hp4_64-ctx4096"
echo '== K8V6 HP4:64 4K quality =='
env INKWELL_KV_HP=4:64 "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 v6hp4_64 k8v6 || exit $?
