#!/usr/bin/env bash
# Interleaved 64K K8V4 comparison of the v13 control, v22 rowptr/old-cache,
# and v26 rowptr/joint-cache binaries.  Run inside one gpu-run job.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
M=$ROOT/opt/kvq64/harness/measure_ctx_state.sh
B13=/home/ash/.cache/inkwell-kv64k/bin/inkwell-kvq-v13-ldg
B22=/home/ash/.cache/inkwell-kv64k2/bin/inkwell-kvq-v22-exactpipe
B26=/home/ash/.cache/inkwell-kv64k2/bin/inkwell-kvq-v26-rowptr-jcache
for r in 1 2 3; do
  echo "== round$r v13 =="
  "$M" "$B13" 65536 k8v4 "cmpv13-r$r" INKWELL_WARM_GENS=16 || exit $?
  echo "== round$r v22 =="
  "$M" "$B22" 65536 k8v4 "cmpv22-r$r" INKWELL_WARM_GENS=16 || exit $?
  echo "== round$r v26 =="
  "$M" "$B26" 65536 k8v4 "cmpv26-r$r" INKWELL_WARM_GENS=16 || exit $?
done
