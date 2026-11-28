#!/usr/bin/env bash
# INT8 64K interleaved decode A/B: dp4a off vs dp4a2, three rounds, warm-corrected.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4}
M=$ROOT/opt/kvq64/harness/measure_ctx.sh
MS=$ROOT/opt/kvq64/harness/measure_ctx_state.sh
if [ ! -s /home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts/mv2-int8-ctx65536/state.bin ]; then
  echo '== create int8 64K state =='
  "$M" "$BIN" 65536 mv2-int8 INKWELL_KV=int8 INKWELL_WARM_GENS=16 || exit $?
fi
for r in 1 2 3; do
  echo "== round$r int8 dp4a off =="
  "$MS" "$BIN" 65536 int8 "int8-spd-off64-r$r" INKWELL_WARM_GENS=16 || exit $?
  echo "== round$r int8 dp4a mode2 =="
  "$MS" "$BIN" 65536 int8 "int8-spd-m2-64-r$r" INKWELL_WARM_GENS=16 INKWELL_KVQ_DP4A=2 || exit $?
done
