#!/usr/bin/env bash
# 1) K10V6 controlled write-roundtrip + attention-consistency test.
# 2) INT8 (K8V8) real 4K quality, reusing F16 controls, completing the
#    real-KLD vs V-bits curve: V4=0.0072, V6=0.00262, V8=?
set -u
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
rc_all=0
echo '== K10V6 controlled roundtrip+attention =='
/home/ash/.cache/inkwell-intkv/kvq-attn-test-k10
r=$?; echo "k10test_rc=$r"; [ "$r" -eq 0 ] || rc_all=1
echo '== INT8 4K quality =='
ART=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/artifacts
SRC=$ART/quality-k10qg-ctx4096
DST=$ART/quality-int8now-ctx4096
mkdir -p "$DST"
for f in ref.bin repeat.bin noise64.bin teacher.ids; do
  if [ -s "$SRC/$f" ] && [ ! -e "$DST/$f" ]; then ln "$SRC/$f" "$DST/$f" 2>/dev/null || cp "$SRC/$f" "$DST/$f"; fi
done
"$ROOT/opt/kvq64/harness/run_quality.sh" /home/ash/.cache/inkwell-intkv/bin/inkwell-intkv-mode4 4096 int8now int8
r=$?; echo "int8q_rc=$r"; [ "$r" -eq 0 ] || rc_all=1
exit "$rc_all"
