#!/usr/bin/env bash
# K8V6 4K quality with INKWELL_KV_HP=1024:1024 — half the 4K context fp16.
# Decisive floor test: if real KLD stays ~0.002, the kvq-kernel-vs-F16-kernel
# numerics floor dominates and NO quantized format can reach 1e-3; if it drops
# well below, quantization error dominates and higher-precision formats can win.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-v6hp1024-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
echo '== K8V6 HP1024:1024 4K quality =='
env INKWELL_KV_HP=1024:1024 "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 v6hp1024 k8v6 || exit $?
