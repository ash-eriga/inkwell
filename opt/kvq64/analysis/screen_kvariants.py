#!/usr/bin/env python3
"""Targeted K-side screening on real attention dumps."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import simulate_formats as sf

def sym_quant_weighted(x, bits, group, w):
    # Per-row/group adaptive scale times a fixed per-channel weight vector w [h,1,d].
    h,n,d=x.shape
    xx=x.reshape(h,n,d//group,group)
    ww=w.reshape(h,1,d//group,group)
    qmax=(1<<(bits-1))-1
    z=xx/np.maximum(ww,1e-30)
    sc=(np.abs(z).max(axis=3,keepdims=True)/qmax).astype(np.float16).astype(np.float32)
    den=np.where(sc==0,1.0,sc)
    qi=np.clip(np.rint(z/den),-qmax,qmax)
    return (qi*sc*ww).reshape(h,n,d)

def eval_custom(case, kbits=8, vbits=6, rot=None, kmean=None, kw=None):
    _,_,k,v,q,bias,mask=case
    kk=k.copy(); vv=v.copy(); qq=q.copy()
    kmu=None
    if kmean is not None:
        kmu=kmean(kk); kk=kk-kmu
    if rot is not None:
        kk=kk@rot; qq=qq@rot
    if kw is None:
        ku=sf.sym_quant(kk,kbits,64)
    else:
        ku=sym_quant_weighted(kk,kbits,64,kw)
    vu=sf.affine_quant(vv,vbits,64)
    outs,probs=sf.attention_with(ku,vu,qq,bias,mask)
    return outs,probs

def summarize_mean(cases, rows):
    refs=[]
    print('tag mean_rel max_rel mean_attnkl max_attnkl')
    for tag,kw in rows:
        rels=[]; kls=[]
        for case in cases:
            outs,probs=eval_custom(case,**kw)
            r,rp,_=sf.ref_attention(case[2],case[3],case[4],case[5],case[6])
            err=outs-r
            rels.append(float(np.sqrt(np.mean(err*err))/np.sqrt(np.mean(r*r))))
            dk=[]
            for pr,qr in zip(rp,probs):
                dk.append(float(np.sum(pr*(np.log(np.maximum(pr,1e-300))-np.log(np.maximum(qr,1e-300))))))
            kls.append(float(np.mean(dk)))
        print(f'{tag:30s} {np.mean(rels):.6g} {np.max(rels):.6g} {np.mean(kls):.6g} {np.max(kls):.6g}')

def main():
    cases=sf.load_cases(Path(sys.argv[1] if len(sys.argv)>1 else '/home/ash/.cache/inkwell-intkv/attn4k-v26-nocache'))
    if len(sys.argv)>2: cases=cases[:int(sys.argv[2])]
    print('cases',len(cases))
    h=sf.hadamard(128,False)
    # Oracle and prefix channel weights: RMS magnitude per channel, normalized to mean one.
    def cw_all(k):
        w=np.sqrt((k*k).mean(axis=1,keepdims=True))
        return w/np.maximum(w.mean(axis=2,keepdims=True),1e-30)
    def cw_p128(k):
        w=np.sqrt((k[:,:128,:]*k[:,:128,:]).mean(axis=1,keepdims=True))
        return w/np.maximum(w.mean(axis=2,keepdims=True),1e-30)
    rows=[]
    rows.append(('k8v6-base',dict()))
    rows.append(('k10v6',dict(kbits=10)))
    rows.append(('k12v6',dict(kbits=12)))
    rows.append(('k8v6-hadK',dict(rot=h)))
    rows.append(('k10v6-hadK',dict(kbits=10,rot=h)))
    rows.append(('k8v6-kmean',dict(kmean=sf.mean_all)))
    rows.append(('k8v6-hadK-kmean',dict(rot=h,kmean=sf.mean_all)))
    # Compute oracle weights per case inside a custom loop by monkey-patch-like lambdas.
    for label,fn in [('cwall',cw_all),('cwp128',cw_p128)]:
        rels=[]; kls=[]
        for case in cases:
            kw=fn(case[2])
            outs,probs=eval_custom(case,kw=kw)
            r,rp,_=sf.ref_attention(case[2],case[3],case[4],case[5],case[6])
            err=outs-r; rels.append(float(np.sqrt(np.mean(err*err))/np.sqrt(np.mean(r*r))))
            dk=[float(np.sum(pr*(np.log(np.maximum(pr,1e-300))-np.log(np.maximum(qr,1e-300))))) for pr,qr in zip(rp,probs)]
            kls.append(float(np.mean(dk)))
        print(f'k8v6-{label:23s} {np.mean(rels):.6g} {np.max(rels):.6g} {np.mean(kls):.6g} {np.max(kls):.6g}')
    summarize_mean(cases,rows)
if __name__=='__main__': main()
