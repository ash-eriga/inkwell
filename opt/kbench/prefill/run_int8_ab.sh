#!/usr/bin/env bash
# Interleaved chunk512 encode A/B for INKWELL_PF_INT8.  Run through gpu-run.
# usage: run_int8_ab.sh <inkwell-bin> <outdir> [rounds]
set -uo pipefail
BIN=$1; OUT=$2; ROUNDS=${3:-1}
P=/home/ash/.cache/inkwell-prefill-bench/p513.txt.ids
F=/home/ash/.cache/inkwell-prefill-bench/f513.ids
rm -rf "$OUT"; mkdir -p "$OUT"
runone(){
  local arm=$1 val=$2 leg=$3
  local od=$OUT/$leg
  mkdir -p "$od"
  env GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=768 INKWELL_CACHE_MARGIN_MB=10 \
    INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
    INKWELL_PROMPT_IDS="$P" INKWELL_FORCE_IDS="$F" INKWELL_GENS=0 \
    INKWELL_PREFILL_CHUNK=512 INKWELL_PF_INT8=$val \
    LD_LIBRARY_PATH="$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin" \
    "$BIN" "$HOME/models/inkling" --run >"$od/log" 2>&1
  echo $? >"$od/rc"
}
for r in $(seq 1 "$ROUNDS"); do
  runone ctl 0 "r${r}-ctl-a"
  runone cand 1 "r${r}-cand-a"
  runone cand 1 "r${r}-cand-b"
  runone ctl 0 "r${r}-ctl-b"
done
python3 - "$OUT" "$ROUNDS" <<'PY'
import re,sys,statistics,pathlib
out=pathlib.Path(sys.argv[1]); rounds=int(sys.argv[2])
res={'ctl':[], 'cand':[]}; rcs=[]
for p in sorted(out.iterdir()):
    txt=(p/'log').read_text(errors='replace')
    m=re.search(r'encode (\d+) tok in ([0-9.]+)(ms|s) \(',txt)
    rcs.append((p.name,int((p/'rc').read_text())))
    if m:
        sec=float(m.group(2))*(.001 if m.group(3)=='ms' else 1)
        res['ctl' if '-ctl-' in p.name else 'cand'].append(int(m.group(1))/sec)
print('rcs',rcs)
for arm in ('ctl','cand'):
    print(arm,','.join(f'{x:.2f}' for x in res[arm]),'median',f'{statistics.median(res[arm]):.2f}' if res[arm] else '-')
if res['ctl'] and res['cand']:
    print('delta_pct',f"{100*(statistics.median(res['cand'])/statistics.median(res['ctl'])-1):+.2f}")
PY
