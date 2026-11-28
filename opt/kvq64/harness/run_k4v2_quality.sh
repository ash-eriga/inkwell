#!/usr/bin/env bash
# K4V2 real 4K quality with current harness. K4V2 has the worst quantization
# (K4/V2) of any format; if it measures near INT8/K8V6 (~0.002-0.003), the
# real-KLD floor is kernel-path numerics, not quantization noise.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-k4v2now-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
echo '== K4V2 4K quality =='
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 k4v2now k4v2 || exit $?
