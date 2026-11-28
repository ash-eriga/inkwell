#!/usr/bin/env bash
# K10V6 64K state creation plus interleaved decode A/B for grouped dp4a.
# Run inside one gpu-run job. The first arm creates the expensive state; later arms reuse it.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=${1:-/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-k10}
M=$ROOT/opt/kvq64/harness/measure_ctx.sh
MS=$ROOT/opt/kvq64/harness/measure_ctx_state.sh
if [ ! -s /home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts/mv2-k10v6-ctx65536/state.bin ]; then
  echo '== create k10v6 64K state =='
  "$M" "$BIN" 65536 mv2-k10v6 INKWELL_KV=k10v6 INKWELL_WARM_GENS=16 || exit $?
fi
for r in 1 2 3; do
  echo "== round$r k10v6 dp4a off =="
  "$MS" "$BIN" 65536 k10v6 "k10v6-off64-r$r" INKWELL_WARM_GENS=16 || exit $?
  echo "== round$r k10v6 grouped dp4a =="
  "$MS" "$BIN" 65536 k10v6 "k10v6-qg64-r$r" INKWELL_WARM_GENS=16 INKWELL_KVQ_DP4A=3 || exit $?
done
