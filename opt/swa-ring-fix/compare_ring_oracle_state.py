#!/usr/bin/env python3
# Compare ring-slot contents against the linear oracle arena at the positions
# each ring slot should hold after encoding positions 0..4094.
import struct, sys
import numpy as np

MAXC=4352; SWA=1024; HD=128; HKV=8; ROW=HD
FULL=HD*MAXC*HKV; RING=HD*SWA*HKV

def visits():
    return list(range(24))+list(range(6,18))+list(range(6,18))
def is_full(il): return il%6==5

def read_state(path):
    b=open(path,'rb').read()
    assert b[:8]==b'INKWST02'
    n=struct.unpack('<Q',b[8:16])[0]; assert n==144
    off=16; ts=[]
    for _ in range(n):
        ln=struct.unpack('<Q',b[off:off+8])[0]; off+=8
        ts.append(b[off:off+ln]); off+=ln
    assert off==len(b)
    return ts

def f16(x, cap):
    a=np.frombuffer(x,dtype='<f2').reshape(HKV,cap,HD).astype(np.float32)
    return a

ring=read_state(sys.argv[1]); oracle=read_state(sys.argv[2])
vs=visits()
rows=0; exact=0; correct_better=0
maxes=[]; ratios=[]; worst=(0,None)
positions=range(3071,4095)
for group in range(2):
  for slot,il in enumerate(vs):
    if is_full(il): continue
    r=f16(ring[group*48+slot],SWA)
    o=f16(oracle[group*48+slot],MAXC)
    for p in positions:
      jr=p%SWA
      a=r[:,jr,:].reshape(-1); b=o[:,p,:].reshape(-1)
      dc=float(np.max(np.abs(a-b)))
      # adjacent positions are a sensitive shift-by-one check.
      dm=min(float(np.max(np.abs(a-o[:,p-1,:].reshape(-1)))) if p>3071 else 1e9,
             float(np.max(np.abs(a-o[:,p+1,:].reshape(-1)))) if p<4094 else 1e9)
      rows+=1; exact+=int(dc==0); correct_better+=int(dc<=dm)
      maxes.append(dc); ratios.append(dc/(dm+1e-9))
      if dc>worst[0]: worst=(dc,('kv'[group],slot,il,p,dm))
print(f'rows={rows} exact={exact} correct_le_adjacent={correct_better} ({correct_better/rows:.4f})')
print(f'maxdiff_mean={np.mean(maxes):.6g} median={np.median(maxes):.6g} p95={np.quantile(maxes,.95):.6g} max={np.max(maxes):.6g}')
print(f'worst={worst}')
print(f'ratio_correct_over_bestadjacent mean={np.mean(ratios):.6g} p95={np.quantile(ratios,.95):.6g}')
