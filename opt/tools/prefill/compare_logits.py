#!/usr/bin/env python3
"""Compare record-prefixed INKWELL_LOGIT_DUMP files and generated-id files."""
import json, math, struct, sys
from pathlib import Path

def rows(path):
    data=Path(path).read_bytes(); out=[]; off=0
    while off+4<=len(data):
        n=struct.unpack_from('<I',data,off)[0]; off+=4
        vals=list(struct.unpack_from('<%df'%n,data,off)); off+=4*n
        out.append(vals)
    return out

def ids(path):
    data=Path(path).read_bytes()
    return list(struct.unpack('<%dI'%(len(data)//4),data))

def softmax(x):
    m=max(x); e=[math.exp(v-m) for v in x]; s=sum(e); return [v/s for v in e]

def kld(p,q):
    return sum(pi*math.log(max(pi,1e-300)/max(qi,1e-300)) for pi,qi in zip(p,q))

def main():
    if len(sys.argv)<5:
        print('usage: compare_logits.py serial.logits cand.logits serial.ids cand.ids [out.json]',file=sys.stderr); return 2
    arows,brows=rows(sys.argv[1]),rows(sys.argv[2])
    n=min(len(arows),len(brows),32)
    ks=[]; agree=0; maxabs=0.0; first_greedy_diff=None
    for i in range(n):
        a,b=arows[i],brows[i]
        maxabs=max(maxabs,max(abs(x-y) for x,y in zip(a,b)))
        pa,pb=softmax(a),softmax(b)
        ks.append(kld(pa,pb))
        ha=max(range(len(a)),key=lambda j:a[j])
        hb=max(range(len(b)),key=lambda j:b[j])
        same=ha==hb
        agree += int(same)
        if not same and first_greedy_diff is None: first_greedy_diff=i
    ai,bi=ids(sys.argv[3]),ids(sys.argv[4])
    m=min(len(ai),len(bi),32)
    # gen.ids is the fed stream: both arms are teacher-forced from the same
    # INKWELL_FORCE_IDS, so equality there proves nothing about quality. The
    # greedy gate must come from the logits argmax.
    forced=sum(x==y for x,y in zip(ai[:m],bi[:m]))
    forced_firstdiff=next((i for i,(x,y) in enumerate(zip(ai[:m],bi[:m])) if x!=y),None)
    res={'rows':n,'row0_kld':ks[0] if ks else None,'mean_kld':sum(ks)/len(ks) if ks else None,
         'max_kld':max(ks) if ks else None,'top1_agree':agree/n if n else None,
         'max_abs_logit_diff':maxabs,
         'greedy32_matches':agree,'greedy32_first_diff':first_greedy_diff,
         'greedy32_exact':agree==n,
         'forced32_matches':forced,'forced32_first_diff':forced_firstdiff,
         'forced32_exact':forced==m}
    text=json.dumps(res,indent=2,sort_keys=True)
    if len(sys.argv)>5: Path(sys.argv[5]).write_text(text+'\n')
    print(text)
    return 0
if __name__=='__main__': sys.exit(main())
