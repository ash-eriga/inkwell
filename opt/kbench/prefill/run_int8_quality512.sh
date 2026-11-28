#!/usr/bin/env bash
# 512-token q8-prefill quality against the archived serial FP32 reference.
set -uo pipefail
BIN=$1; OUT=$2
rm -rf "$OUT"; mkdir -p "$OUT"
INKWELL_PF_INT8=1 opt/tools/prefill/run_case.sh "$BIN" 512 \
  /home/ash/.cache/inkwell-prefill-bench/p513.txt.ids \
  /home/ash/.cache/inkwell-prefill-bench/f513.ids \
  "$OUT/prefill" prefill 512 >"$OUT/runner" 2>&1
rc=$?
python3 opt/kbench/prefill/quality_metrics.py \
  /home/ash/.cache/inkwell-prefill/out/relfix/512/serial/logits.bin \
  "$OUT/prefill/logits.bin" \
  /home/ash/.cache/inkwell-prefill/out/relfix/512/serial/gen.ids \
  "$OUT/prefill/gen.ids" "$OUT/compare.json" \
  /home/ash/.cache/inkwell-prefill-bench/f513.ids >"$OUT/compare.stdout" 2>&1 || true
cat "$OUT/runner"; cat "$OUT/compare.json" 2>/dev/null || true
exit "$rc"
