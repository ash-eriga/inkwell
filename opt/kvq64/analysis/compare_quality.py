#!/usr/bin/env python3
import struct,sys
import numpy as np

def rows(path):
    out=[]
    with open(path,'rb') as f:
        while True:
            b=f.read(4)
            if not b: return out
            (n,)=struct.unpack('<I',b)
            out.append(np.frombuffer(f.read(4*n),dtype='<f4').astype(np.float64))

def ids(path):
    b=open(path,'rb').read(); return np.frombuffer(b,dtype='<u4')

def prob(row):
    x=row-row.max(); e=np.exp(x); return e/e.sum()

def kld(a,b):
    pa,pb=prob(a),prob(b)
    return float(np.sum(pa*(np.log(pa)-np.log(pb))))

def compare(ref,cand,teacher=None):
    a,b=rows(ref),rows(cand); n=min(len(a),len(b))
    v=np.array([kld(a[i],b[i]) for i in range(n)])
    agree=sum(int(np.argmax(a[i])==np.argmax(b[i])) for i in range(n))
    out=dict(rows=n,top1=agree/n if n else 0,mean=float(v.mean()) if n else 0,p95=float(np.quantile(v,.95)) if n else 0,max=float(v.max()) if n else 0)
    if teacher:
        t=ids(teacher)[:n]
        def ppl(x):
            nll=[]
            for i in range(n):
                p=prob(x[i]); nll.append(-float(np.log(max(p[int(t[i])],1e-300))))
            return float(np.exp(np.mean(nll))) if nll else 0.0
        out['ppl']=ppl(b); out['refppl']=ppl(a)
    return out
if __name__=='__main__':
    r=compare(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
    print(' '.join(f'{k}={v:.9g}' if isinstance(v,float) else f'{k}={v}' for k,v in r.items()))
