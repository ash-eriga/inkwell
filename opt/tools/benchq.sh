#!/usr/bin/env bash
# Canonical ctx512 bench for use inside gpu-run (unlike bench1.sh, no nested flock).
# usage: benchq.sh <tag> <binary> [extra-env KEY=V ...]
set -u
TAG=$1; BIN=$2; shift 2
LOG=/tmp/inkwell-benchq-$TAG.log
(
  env GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 \
      INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 "$@" \
      LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin \
      "$BIN" "$HOME/models/inkling" --run
) >"$LOG" 2>&1 &
BPID=$!
PEAK=0
while kill -0 $BPID 2>/dev/null; do
  OUT=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  M=$(echo "$OUT" | awk -v pids="$(pgrep -P $BPID | tr '\n' ' ')$BPID" '{for(i=1;i<=length(pids);i++){}} $1==pids {if($2>m)m=$2} END{print m+0}')
  # nvidia-smi reports the inference child, not the env wrapper; fall back to maximum if the child is not visible yet.
  if [ "$M" -eq 0 ]; then M=$(echo "$OUT" | awk '{if($2>m)m=$2} END{print m+0}'); fi
  [ "$M" -gt "$PEAK" ] && PEAK=$M
  sleep 0.25
done
wait $BPID; RC=$?
python3 - "$LOG" "$TAG" "$RC" "$PEAK" <<'PY'
import sys,re,hashlib,pathlib
log=pathlib.Path(sys.argv[1]).read_text(errors='replace')
m=re.search(r'decode 8 tok in ([0-9.]+)s',log)
g=64
ms=float(m.group(1))*1000/g if m else -1
text='\n'.join(re.findall(r'^generated text: .*$',log,re.M))+'\n'
md5=hashlib.md5(text.encode()).hexdigest()[:8]
print(f"RUN {sys.argv[2]} rc={sys.argv[3]} TIME_MS_TOK={ms:.3f} MD5={md5} VRAM_PEAK_MIB={sys.argv[4]}")
PY
exit $RC
