#!/bin/bash
# One bench run under the GPU lock. Bench env = the A/B baseline env.
# Usage: GRAPH_OFF=1 INKWELL_PROMPT='...' bench1.sh <tag> [extra env KEY=V ...]
# Prints "RUN <tag> rc=.. TIME_MS_TOK=.. MD5=.. VRAM_PEAK_MIB=.. VRAM_TOT_MIB=.."
# Full stderr log: /tmp/inkwell-bench-<tag>.log
set -u
TAG=$1; shift
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="/tmp/inkwell-bench-$TAG.log"
GRAPHS=(GGML_PASCAL_GRAPHS=1)
if [ "${GRAPH_OFF:-0}" = "1" ]; then GRAPHS=(-u GGML_PASCAL_GRAPHS); fi
(
  flock /tmp/inkwell-gpu.lock \
  env "${GRAPHS[@]}" INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 \
      INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 "$@" \
      LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin \
      "$ROOT/target/release/inkwell" $HOME/models/inkling --run
) > "$LOG" 2>&1 &
BPID=$!
PEAK=0; PEAKT=0
while kill -0 $BPID 2>/dev/null; do
  OUT=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  M=$(echo "$OUT" | awk '{if($2>m)m=$2} END{print m+0}')
  S=$(echo "$OUT" | awk '{s+=$2} END{print s+0}')
  [ "$M" -gt "$PEAK" ] && PEAK=$M
  [ "$S" -gt "$PEAKT" ] && PEAKT=$S
  sleep 0.4
done
wait $BPID
RC=$?
DECT=$(grep "decode 8 tok in" "$LOG" | sed 's/.*in //')
MD5=$(grep "generated text" "$LOG" | md5sum | cut -c1-8)
MS=$(echo "$DECT" | python3 -c "import sys,re; s=sys.stdin.read(); m=re.search(r'([\d.]+)s', s); print(f'{float(m.group(1))*1000/64:.2f}' if m else 'NA')")
echo "RUN $TAG rc=$RC TIME_MS_TOK=$MS MD5=$MD5 VRAM_PEAK_MIB=$PEAK VRAM_TOT_MIB=$PEAKT"
