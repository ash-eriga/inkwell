#!/usr/bin/env bash
# usage inside gpu-run: measure_zero64.sh <binary> <tag> <fmt> <scope> [extra-env KEY=V ...]
set -u
BIN=$1; TAG=$2; FMT=$3; SCOPE=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
OUT=$ROOT/opt/kvq64/front2g/artifacts/$TAG
STATE=/home/ash/.cache/inkwell-front2g/zero-states/$FMT-$SCOPE-ctx65536.bin
mkdir -p "$OUT"
[ -s "$STATE" ] || { echo "missing_state=$STATE"; exit 2; }
export GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=65536 \
  INKWELL_PROMPT_IDS=/home/ash/.t3/worktrees/inkwell/t3-866aa1a8-c87531e8cbdf0e38/opt/kvq64/prompts/p65536.ids \
  INKWELL_GENS=64 INKWELL_WARM_GENS=16 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
  INKWELL_CACHE_MARGIN_MB=10 INKWELL_KV=$FMT INKWELL_KV_SCOPE=$SCOPE INKWELL_KVQ_DP4A=2 \
  INKWELL_VRAM_CAP_MB=5180 INKWELL_STATE_LOAD=$STATE
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
for e in "$@"; do export "$e"; done
"$BIN" "$HOME/models/inkling" --run > "$OUT/decode.log" 2>&1 & rp=$!
PEAK=0
while kill -0 $rp 2>/dev/null; do
  M=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -v p=$rp '$1==p{if($2>m)m=$2} END{print m+0}')
  if [ "$M" -eq 0 ]; then M=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{if($2>m)m=$2} END{print m+0}'); fi
  [ "$M" -gt "$PEAK" ] && PEAK=$M
  sleep 0.2
done
wait $rp; rc=$?
python3 - "$OUT" "$rc" "$PEAK" <<'PY'
import re,sys,hashlib,pathlib
out=pathlib.Path(sys.argv[1]); rc=int(sys.argv[2]); peak=sys.argv[3]
log=(out/'decode.log').read_text(errors='replace')
m=re.search(r'decode_timed (\d+) tok in ([0-9.]+)s',log)
ms=float(m.group(2))*1000/int(m.group(1)) if m else -1
text=''.join(re.findall(r'^generated text: .*$',log,re.M))
md5=hashlib.md5(text.encode()).hexdigest()[:8] if text else 'none'
kv=re.search(r'kvq: format=\S+ quant=(\d+)MiB hp=(\d+)MiB',log)
kvb=re.search(r'bufs: dev=\d+MiB kv=(\d+)MiB',log)
vc=re.search(r'used_pre_cache=(\d+)MiB budget=(\d+)MiB table=(\d+)MiB',log)
print(f"decode_rc={rc} ms={ms:.3f} peak={peak} md5={md5} kvq={kv.group(1) if kv else '-'} hp={kv.group(2) if kv else '-'} kvtotal={kvb.group(1) if kvb else '?'} precache={vc.group(1) if vc else '?'} budget={vc.group(2) if vc else '?'} table={vc.group(3) if vc else '?'}")
PY
exit $rc
