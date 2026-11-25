#!/usr/bin/env python3
import struct, sys
import numpy as np

def rows(path):
    out=[]
    with open(path,'rb') as f:
        while True:
            b=f.read(4)
            if not b: return out
            (n,)=struct.unpack('<I',b)
            out.append(np.frombuffer(f.read(4*n),dtype='<f4').astype(np.float64))

def kld(a,b):
    x=a-a.max(); y=b-b.max()
    ex=np.exp(x); ey=np.exp(y)
    pa=ex/ex.sum(); pb=ey/ey.sum()
    return float(np.sum(pa*(np.log(pa)-np.log(pb))))

def compare(pa,pb):
    a,b=rows(pa),rows(pb); n=min(len(a),len(b))
    vals=[]; agree=0; maxabs=0.0
    for i in range(n):
        vals.append(kld(a[i],b[i])); agree += int(np.argmax(a[i])==np.argmax(b[i])); maxabs=max(maxabs,float(np.max(np.abs(a[i]-b[i]))))
    v=np.array(vals)
    print(f'rows={n} top1_agree={agree}/{n} ({agree/n:.4f})')
    print(f'kld_mean={v.mean():.9g} kld_median={np.median(v):.9g} kld_p95={np.quantile(v,.95):.9g} kld_max={v.max():.9g}')
    print(f'row0_kld={v[0]:.9g} row0_logit_max_abs={float(np.max(np.abs(a[0]-b[0]))):.9g} row0_top1={int(np.argmax(a[0])==np.argmax(b[0]))}')
    print(f'logit_max_abs={maxabs:.9g}')
    print('first16_kld=' + ','.join(f'{x:.4g}' for x in v[:16]))
    return 0 if len(a)==len(b) else 2
if __name__=='__main__': sys.exit(compare(sys.argv[1],sys.argv[2]))
