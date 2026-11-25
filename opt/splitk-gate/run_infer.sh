#!/bin/bash
# Run one inkwell inference under the shared GPU lock and record maximum per-process VRAM peak.
set -u
TAG=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
LOG="$ART/logs/$TAG.log"
mkdir -p "$ART/logs" "$ART/results"
(
  nice -n 19 flock /tmp/inkwell-gpu.lock \
  env GGML_PASCAL_GRAPHS=1 INKWELL_CACHE_MARGIN_MB=10 \
      INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 "$@" \
      LD_LIBRARY_PATH="$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin" \
      "$ROOT/target/release/inkwell" "$HOME/models/inkling" --run
) >"$LOG" 2>&1 &
PID=$!
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  M=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{if($2>m)m=$2} END{print m+0}')
  [ "$M" -gt "$PEAK" ] && PEAK=$M
  sleep 0.5
done
wait "$PID"
RC=$?
MD5=$(grep 'generated text' "$LOG" | md5sum | cut -c1-8)
printf 'RUN %s rc=%s md5=%s vram_peak_mib=%s log=%s\n' "$TAG" "$RC" "$MD5" "$PEAK" "$LOG" | tee "$ART/results/$TAG.run"
exit "$RC"
