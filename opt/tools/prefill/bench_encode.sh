#!/usr/bin/env bash
# Serial-vs-chunked encode matrix. Run inside one gpu-run invocation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="${1:-$ROOT/target/release/inkwell}"
OUT="${2:-/tmp/inkwell-pf-encode-bench}"
CHUNK="${3:-128}"
CORPUS=/home/ash/.cache/inkwell-prefill-bench
mkdir -p "$OUT"

labels=(127 512 2048 4096 8192)
prompts=(/tmp/inkwell-pf-bench/p128.txt.ids "$CORPUS/p513.txt.ids" "$CORPUS/p2049.txt.ids" "$CORPUS/p4097.txt.ids" "$CORPUS/p8193.txt.ids")
forces=(/tmp/inkwell-pf-bench/f128.ids "$CORPUS/f513.ids" "$CORPUS/f2049.ids" "$CORPUS/f4097.ids" "$CORPUS/f8193.ids")

for i in "${!labels[@]}"; do
  n="${labels[$i]}"
  mkdir -p "$OUT/$n"
  for mode in serial prefill; do
    od="$OUT/$n/$mode"
    rm -rf "$od"
    "$ROOT/opt/tools/prefill/run_case.sh" "$BIN" "$n" "${prompts[$i]}" "${forces[$i]}" "$od" "$mode" "$CHUNK" >"$od.runner" 2>&1 || true
  done
  python3 "$ROOT/opt/tools/prefill/compare_logits.py" \
    "$OUT/$n/serial/logits.bin" "$OUT/$n/prefill/logits.bin" \
    "$OUT/$n/serial/gen.ids" "$OUT/$n/prefill/gen.ids" \
    "$OUT/$n/compare.json" >/dev/null 2>&1 || true
done

python3 - "$OUT" <<'PY'
import json,re,sys
from pathlib import Path
out=Path(sys.argv[1])
rows=[]
for nd in sorted((p for p in out.iterdir() if p.is_dir()), key=lambda p:int(p.name)):
    row={'label':int(nd.name)}
    for mode in ('serial','prefill'):
        log=nd/mode/'log'
        row[mode]={}
        if log.exists():
            txt=log.read_text(errors='replace')
            m=re.search(r'encode (\d+) tok in ([0-9.]+)(ms|s) \(',txt)
            if m:
                sec=float(m.group(2))*(0.001 if m.group(3)=='ms' else 1.0)
                tok=int(m.group(1))
                row[mode]={'tok':tok,'sec':sec,'tok_s':tok/sec}
        rc=nd/mode/'rc'
        peak=nd/mode/'peak_mib'
        if rc.exists(): row[mode]['rc']=int(rc.read_text().strip())
        if peak.exists(): row[mode]['peak_mib']=int(peak.read_text().strip())
    cj=nd/'compare.json'
    if cj.exists(): row['quality']=json.loads(cj.read_text())
    rows.append(row)
print('label mode encode_tok seconds tok_s speedup peak_mib rc mean_kld max_kld top1 max_abs_logit_diff')
for r in rows:
    q=r.get('quality',{})
    for mode in ('serial','prefill'):
        m=r.get(mode,{})
        s=r.get('serial',{}); p=r.get('prefill',{})
        speed=(s.get('sec',0)/p.get('sec',1)) if mode=='prefill' and s.get('sec') and p.get('sec') else 1.0
        print(r['label'],mode,m.get('tok','-'),f"{m.get('sec',0):.6f}",f"{m.get('tok_s',0):.3f}",f"{speed:.3f}",m.get('peak_mib','-'),m.get('rc','-'),
              f"{q.get('mean_kld',float('nan')):.6g}",f"{q.get('max_kld',float('nan')):.6g}",f"{q.get('top1_agree',float('nan')):.6g}",f"{q.get('max_abs_logit_diff',float('nan')):.6g}")
(out/'summary.json').write_text(json.dumps(rows,indent=2))
PY
