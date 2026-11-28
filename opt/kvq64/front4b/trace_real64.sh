#!/usr/bin/env bash
# Capture the ordered real64 routing trace. usage inside gpu-run: trace_real64.sh <binary> <tag> <out.idseq>
set -u
BIN=$1; TAG=$2; IDS=$3
OUT=/home/ash/.cache/inkwell-front4b/logs/$TAG
mkdir -p "$OUT"
rm -f "$IDS"
export INKWELL_MAX_CTX=65536 \
  INKWELL_PROMPT_IDS=opt/kvq64/prompts/p65536.ids \
  INKWELL_GENS=64 INKWELL_WARM_GENS=16 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
  INKWELL_CACHE_MARGIN_MB=10 INKWELL_KV=k8v4 INKWELL_KVQ_DP4A=2 \
  INKWELL_VRAM_CAP_MB=5180 INKWELL_STATE_LOAD=/home/ash/.cache/inkwell-real-states/real-k8v4-ctx65536/state.bin \
  INKWELL_IDSEQ="$IDS"
unset GGML_PASCAL_GRAPHS
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
"$BIN" "$HOME/models/inkling" --run > "$OUT/decode.log" 2>&1
rc=$?
python3 - "$IDS" "$rc" <<'PY'
import sys,struct,os
p=sys.argv[1]; rc=int(sys.argv[2])
n=os.path.getsize(p)//36 if os.path.exists(p) else 0
c0=0
if n:
 d=open(p,'rb').read(); c0=sum(1 for o in range(0,len(d),36) if struct.unpack_from('<i',d,o)[0]==0)
print(f'trace_rc={rc} records={n} layer0_visits={c0} path={p}')
PY
exit $rc
