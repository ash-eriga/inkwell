#!/usr/bin/env bash
# Run the three validation jobs whose original gpu-run wrappers were terminated
# before reaching the GPU.  Keep them in one queue slot because the shared GPU
# queue is long; record all results even if one step fails.
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
rc_all=0
echo '== controlled K8V6 attention consistency =='
/home/ash/.cache/inkwell-intkv/kvq-attn-test
r=$?; echo "attntest_rc=$r"; [ "$r" -eq 0 ] || rc_all=1
echo '== K10V6 4K quality, dp4a off =='
"$ROOT/opt/kvq64/harness/run_k10_off_quality.sh"
r=$?; echo "k10off_rc=$r"; [ "$r" -eq 0 ] || rc_all=1
echo '== K8V6 final-binary smoke and 4K quality =='
"$ROOT/opt/kvq64/harness/run_v6_final_recheck.sh"
r=$?; echo "v6final_rc=$r"; [ "$r" -eq 0 ] || rc_all=1
exit "$rc_all"
