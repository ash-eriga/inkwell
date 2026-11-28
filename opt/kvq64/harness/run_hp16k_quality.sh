#!/usr/bin/env bash
# K8V6+HP 16K real quality AND INT8 16K reference, one job. 16K must be no
# worse than INT8. At 16K the fp16 window is only ~6% of tokens (vs 25-50% at
# 4K), so this is the honest test of the hybrid mechanism.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BIN=/home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
HP=${HP:-0:1024}
setup() {
  local dst=$1
  mkdir -p "$dst"
  # 16K controls: generate fresh if missing (ref/repeat/noise are format-independent)
  :
}
echo '== INT8 16K quality reference =='
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 16384 int8-16k int8 || exit $?
echo "== K8V6 HP$HP 16K quality =="
env INKWELL_KV_HP=$HP "$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 16384 "v6hp${HP//:/_}-16k" k8v6 || exit $?
echo '== K8V6 no-HP 16K quality =='
"$ROOT/opt/kvq64/harness/run_quality.sh" "$BIN" 16384 v6nohp-16k k8v6 || exit $?
