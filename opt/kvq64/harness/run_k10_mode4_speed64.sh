#!/usr/bin/env bash
# K10V6 64K interleaved decode A/B: dp4a off, grouped8 mode3, grouped4 mode4.
# Reuses or creates the same K10 state; three rounds interleave all arms.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4}
M=$ROOT/opt/kvq64/harness/measure_ctx.sh
MS=$ROOT/opt/kvq64/harness/measure_ctx_state.sh
if [ ! -s /home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts/mv2-k10v6-ctx65536/state.bin ]; then
  echo '== create k10v6 64K state =='
  "$M" "$BIN" 65536 mv2-k10v6 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 || exit $?
fi
for r in 1 2 3; do
  echo "== round$r k10v6 dp4a off =="
  "$MS" "$BIN" 65536 k10v6 "k10v6-m4cmp-off64-r$r" INKWELL_WARM_GENS=16 || exit $?
  echo "== round$r k10v6 dp4a mode3 =="
  "$MS" "$BIN" 65536 k10v6 "k10v6-m4cmp-m3-64-r$r" INKWELL_WARM_GENS=16 INKWELL_KVQ_DP4A=3 || exit $?
  echo "== round$r k10v6 dp4a mode4 =="
  "$MS" "$BIN" 65536 k10v6 "k10v6-m4cmp-m4-64-r$r" INKWELL_WARM_GENS=16 INKWELL_KVQ_DP4A=4 || exit $?
done
