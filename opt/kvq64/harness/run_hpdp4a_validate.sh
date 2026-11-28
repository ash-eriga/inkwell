#!/usr/bin/env bash
# Validate dp4a+HP integration end-to-end with the hpdp4a binary:
# 1) ctx512/cap768 smoke: K8V6+HP0:1024 with dp4a2 — correctness/coherence.
# 2) Real 4K quality K8V6+HP0:1024+dp4a2 must reproduce the fp32-path result
#    (~0.0009 mean, top1 1.0) — proves the mixed integer/fp32 kernel correct.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-hpdp4a
M=$ROOT/opt/kvq64/harness/measure_ctx.sh
HP=0:1024
echo '== hpdp4a ctx512 smoke K8V6 HP0:1024 dp4a2 =='
env INKWELL_KV_HP=$HP INKWELL_KVQ_DP4A=2 "$M" "$BIN" 512 v6hpdp4a-smoke INKWELL_KV=k8v6 || exit $?
echo '== hpdp4a ctx512 smoke K8V6 HP0:1024 dp4a off (control) =='
env INKWELL_KV_HP=$HP "$M" "$BIN" 512 v6hpoff-smoke INKWELL_KV=k8v6 || exit $?
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-v6hpdp4a-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
echo '== hpdp4a K8V6 HP0:1024 dp4a2 4K quality =='
env INKWELL_KV_HP=$HP INKWELL_KVQ_DP4A=2 "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 4096 v6hpdp4a k8v6 || exit $?
