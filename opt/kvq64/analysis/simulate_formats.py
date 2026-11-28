#!/usr/bin/env python3
"""CPU screening of quantized-KV formats on attention dumps.

The dump layout is the production ggml contiguous layout: K/V are
[8 kv heads][token][128] fp16, Q is [24 query heads][128] fp32, bias is
[24][token] fp32 when materialized per head, and mask is token-major fp32.
"""
import argparse, sys
from pathlib import Path
import numpy as np

HD = 128
HKV = 8
NH = 24
SCALE = 1.0 / np.sqrt(HD, dtype=np.float64)

def load_cases(root: Path):
    cases = []
    for nd in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        nkv = int(nd.name)
        for lp in sorted(p for p in nd.iterdir() if p.is_dir()):
            try:
                k = np.fromfile(lp/'k.f16', dtype=np.float16).astype(np.float32)
                v = np.fromfile(lp/'v.f16', dtype=np.float16).astype(np.float32)
                q = np.fromfile(lp/'q.f32', dtype=np.float32).astype(np.float32)
                if k.size != HKV*nkv*HD or v.size != HKV*nkv*HD or q.size != NH*HD:
                    print(f'skip {lp}: bad sizes k={k.size} v={v.size} q={q.size}', file=sys.stderr)
                    continue
                k = k.reshape(HKV,nkv,HD)
                v = v.reshape(HKV,nkv,HD)
                q = q.reshape(NH,HD)
                bias = np.zeros((NH,nkv), dtype=np.float32)
                if (lp/'bias.f32').exists():
                    b = np.fromfile(lp/'bias.f32', dtype=np.float32).astype(np.float32)
                    if b.size >= NH*nkv: bias = b[:NH*nkv].reshape(NH,nkv)
                    elif b.size >= nkv: bias = np.broadcast_to(b[:nkv], (NH,nkv)).copy()
                mask = np.zeros(nkv, dtype=np.float32)
                if (lp/'mask.f32').exists():
                    m = np.fromfile(lp/'mask.f32', dtype=np.float32).astype(np.float32)
                    if m.size >= nkv: mask = m[:nkv].copy()
                cases.append((lp.name, nkv, k, v, q, bias, mask))
            except FileNotFoundError:
                pass
    return cases

def softmax(s):
    s = s - np.max(s)
    e = np.exp(s)
    return e / np.sum(e)

def ref_attention(k, v, q, bias, mask):
    outs=[]; probs=[]; scores=[]
    for hk in range(HKV):
        for g in range(3):
            h = hk*3+g
            sc = (k[hk].astype(np.float64) @ q[h].astype(np.float64)) * SCALE + bias[h] + mask
            p = softmax(sc)
            outs.append(p @ v[hk].astype(np.float64))
            probs.append(p); scores.append(sc)
    return np.array(outs), probs, scores

def hadamard(n=128, signs=False):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h,h],[h,-h]])
    h /= np.sqrt(n)
    if signs:
        rng = np.random.default_rng(0x51f15e)
        h = h * rng.choice([-1.0,1.0], size=(1,n))
    return h

def sym_quant(x, bits, group, shrink=1.0):
    # x [h,n,d], row/group max-abs signed integer quantization, fp16 scales.
    h,n,d = x.shape
    xx = x.reshape(h,n,d//group,group)
    qmax = (1 << (bits-1)) - 1
    sc = (np.abs(xx).max(axis=3, keepdims=True) / qmax) * shrink
    sc = sc.astype(np.float16).astype(np.float32)
    den = np.where(sc == 0, 1.0, sc)
    qi = np.clip(np.rint(xx / den), -qmax, qmax)
    return (qi * sc).reshape(h,n,d)

def affine_quant(x, bits, group, shrink=1.0):
    h,n,d = x.shape
    xx = x.reshape(h,n,d//group,group)
    levels = (1 << bits) - 1
    mn = xx.min(axis=3, keepdims=True).astype(np.float16).astype(np.float32)
    mx = xx.max(axis=3, keepdims=True)
    sc = ((mx - mn) / levels * shrink).astype(np.float16).astype(np.float32)
    den = np.where(sc == 0, 1.0, sc)
    qi = np.clip(np.rint((xx - mn) / den), 0, levels)
    return (qi * sc + mn).reshape(h,n,d)

def qquant(q):
    # Per-head signed int8 query quantization, fp16 scale.  Returns integer q and scale.
    sc = (np.abs(q).max(axis=1, keepdims=True) / 127.0).astype(np.float16).astype(np.float32)
    den = np.where(sc == 0, 1.0, sc)
    qi = np.clip(np.rint(q / den), -127, 127).astype(np.int32)
    return qi, sc

def qquant2(q):
    # Two-term int8 decomposition: q ~= qs*hi + (qs/128)*lo.
    hi, qs = qquant(q)
    rem = q - hi.astype(np.float32) * qs
    lsc = qs / 128.0
    den = np.where(lsc == 0, 1.0, lsc)
    lo = np.clip(np.rint(rem / den), -127, 127).astype(np.int32)
    return hi, lo, qs

def intk_dequant(x, bits, group):
    """Signed K quantization but return integer codes and fp16 scales."""
    h,n,d = x.shape
    xx = x.reshape(h,n,d//group,group)
    qmax = (1 << (bits-1)) - 1
    sc = (np.abs(xx).max(axis=3, keepdims=True) / qmax).astype(np.float16).astype(np.float32)
    den = np.where(sc == 0, 1.0, sc)
    qi = np.clip(np.rint(xx / den), -qmax, qmax).astype(np.int32)
    return qi.reshape(h,n,d), sc.reshape(h,n,d//group)

def attention_with(kuse, vuse, q, bias, mask, vadd=None, qi=None, qlo=None, qsc=None, kqi=None, ksc=None):
    outs=[]; probs=[]
    for hk in range(HKV):
        for g in range(3):
            h = hk*3+g
            if qi is not None:
                # Exact integer q·k, with a possibly different fp16 scale per 64-channel K group.
                dots = np.zeros(kuse.shape[1], dtype=np.float64)
                ng = ksc.shape[2]
                grp = HD // ng
                for z in range(ng):
                    sl = slice(z*grp,(z+1)*grp)
                    dhi = kqi[hk,:,sl].astype(np.int32) @ qi[h,sl].astype(np.int32)
                    if qlo is not None:
                        dlo = kqi[hk,:,sl].astype(np.int32) @ qlo[h,sl].astype(np.int32)
                        dots += (dhi.astype(np.float64) + dlo.astype(np.float64) / 128.0) * (ksc[hk,:,z] * qsc[h,0])
                    else:
                        dots += dhi.astype(np.float64) * (ksc[hk,:,z] * qsc[h,0])
                sc = dots * SCALE + bias[h] + mask
            else:
                sc = (kuse[hk].astype(np.float64) @ q[h].astype(np.float64)) * SCALE + bias[h] + mask
            p = softmax(sc)
            vv = p @ vuse[hk].astype(np.float64)
            if vadd is not None: vv = vv + vadd[hk,0,:]
            outs.append(vv); probs.append(p)
    return np.array(outs), probs

def kl_rows(ps, qs):
    out=[]
    for p,q in zip(ps,qs):
        qq=np.maximum(q,1e-300); pp=np.maximum(p,1e-300)
        out.append(float(np.sum(p*(np.log(pp)-np.log(qq)))))
    return np.array(out)

def eval_config(name, case, kbits=8, vbits=4, vgroup=64, kmean=None, vmean=None, rot=None, q8=False, kshrink=1.0, vshrink=1.0, rotv=True):
    _, nkv, k, v, q, bias, mask = case
    kk = k.copy(); vv = v.copy(); qq = q.copy()
    kmu = vmu = None
    if kmean is not None:
        kmu = kmean(k); kk = kk - kmu
    if vmean is not None:
        vmu = vmean(v); vv = vv - vmu
    if rot is not None:
        kk = kk @ rot; qq = qq @ rot
        if rotv: vv = vv @ rot
    kuse = sym_quant(kk, kbits, 64, kshrink)
    vuse = sym_quant(vv, vbits, 64, vshrink) if vbits == 8 else affine_quant(vv, vbits, vgroup, vshrink)
    vadd = None
    if vmu is not None or rot is not None:
        # Inverse-rotate after the attention-weighted sum, then add back the fixed V mean.
        # Algebraically both operations are per-output-head/channel and outside the token sum.
        pass
    if q8:
        if q8 == 2:
            qi, qlo, qscc = qquant2(qq)
        else:
            qi, qscc = qquant(qq); qlo = None
        kqi, ksc = intk_dequant(kk, kbits, 64)
        # Account for inverse rotation by rotating the output back after accumulation.
        outs, probs = attention_with(kuse, vuse, qq, bias, mask, qi=qi, qlo=qlo, qsc=qscc, kqi=kqi, ksc=ksc)
    else:
        outs, probs = attention_with(kuse, vuse, qq, bias, mask)
    if rot is not None and rotv:
        outs = outs @ rot.T
    if vmu is not None:
        outs = outs + np.repeat(vmu[:,0,:], 3, axis=0).reshape(HKV,3,HD).reshape(NH,HD)
    return outs, probs

def mean_all(x): return x.mean(axis=1, keepdims=True)
def mean_prefix(n):
    return lambda x: x[:,:min(n,x.shape[1]),:].mean(axis=1, keepdims=True)

def summarize(tag, outs, probs, refs, refprobs):
    err = outs - refs
    rms = float(np.sqrt(np.mean(err*err)))
    ref_rms = float(np.sqrt(np.mean(refs*refs)))
    maxe = float(np.max(np.abs(err)))
    kls = []
    for pq in zip(refprobs, probs): kls.extend(kl_rows([pq[0]],[pq[1]]))
    # The loop above is not right for full rows; compute direct row KLs.
    direct=[]
    for pr,qr in zip(refprobs, probs):
        direct.append(float(np.sum(pr*(np.log(np.maximum(pr,1e-300))-np.log(np.maximum(qr,1e-300))))))
    direct=np.array(direct)
    print(f'{tag:34s} rel={rms/ref_rms:.6g} rms={rms:.6g} max={maxe:.6g} attnkl_mean={direct.mean():.6g} p95={np.quantile(direct,.95):.6g} maxkl={direct.max():.6g}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--case-limit', type=int, default=0)
    a = ap.parse_args()
    cases = load_cases(Path(a.dump))
    if not cases:
        print('no attention dump cases found', file=sys.stderr); return 2
    if a.case_limit: cases = cases[:a.case_limit]
    print(f'cases={len(cases)} ' + ','.join(f'{c[0]}@{c[1]}' for c in cases))
    h128 = hadamard(128, False)
    h128s = hadamard(128, True)
    for case in cases:
        lp,nkv,k,v,q,bias,mask = case
        refs, refprobs, _ = ref_attention(k,v,q,bias,mask)
        print(f'== {lp} nkv={nkv} ==')
        rows=[]
        def ev(tag, **kw): rows.append((tag, eval_config(tag,case,**kw)))
        ev('int8-base', vbits=8)
        ev('k8v4-base')
        ev('k8v4-v32', vgroup=32)
        ev('k8v5-v32', vbits=5, vgroup=32)
        ev('k8v6-v32', vbits=6, vgroup=32)
        ev('k8v6-v64', vbits=6, vgroup=64)
        ev('k8v6-hadK-kvmean-v32', vbits=6, rot=h128, rotv=False, kmean=mean_all, vmean=mean_all, vgroup=32)
        ev('k8v4-v32-vshrink90', vgroup=32, vshrink=0.9)
        ev('k8v4-v32-vshrink80', vgroup=32, vshrink=0.8)
        ev('k8v4-v32-kshrink95v90', vgroup=32, kshrink=0.95, vshrink=0.9)
        ev('k8v4-kmean-all', kmean=mean_all)
        ev('k8v4-vmean-all', vmean=mean_all)
        ev('k8v4-kvmean-all', kmean=mean_all, vmean=mean_all)
        ev('k8v4-kvmean-all-v32', kmean=mean_all, vmean=mean_all, vgroup=32)
        ev('k8v4-kvmean-p128-v32', kmean=mean_prefix(128), vmean=mean_prefix(128), vgroup=32)
        ev('k8v4-hadK', rot=h128, rotv=False)
        ev('k8v4-hadK-v32', rot=h128, rotv=False, vgroup=32)
        ev('k8v4-hadK-kmean-v32', rot=h128, rotv=False, kmean=mean_all, vmean=mean_all, vgroup=32)
        ev('k8v4-hadamard', rot=h128)
        ev('k8v4-hadamard-sign', rot=h128s)
        ev('k8v4-hads-kvmean-v32', rot=h128s, kmean=mean_all, vmean=mean_all, vgroup=32)
        ev('int8-q8dp4a', vbits=8, q8=1)
        ev('int8-q8x2dp4a', vbits=8, q8=2)
        ev('k8v4-q8dp4a', q8=1)
        ev('k8v4-q8x2dp4a', q8=2)
        ev('k8v4-q8dp4a-bestguess', q8=True, rot=h128s, kmean=mean_all, vmean=mean_all, vgroup=32)
        for tag,(outs,probs) in rows:
            summarize(tag,outs,probs,refs,refprobs)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
