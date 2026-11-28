#!/usr/bin/env python3
# Create a zero-filled INKWST02 state for decode-speed measurements.
# Values are semantically meaningless; allocation sizes, layout and attention lengths are exact.
import struct,sys,os

def rows(fmt):
    return {'int8':(1024,1024,96),'fp8':(1024,1024,0),'k8v4':(1024,512,96),'k8v4g32':(1024,512,192),'k8v6':(1024,768,96),'k10v6':(1280,768,96),'k4v2':(512,256,96)}[fmt]

def hp(fmt):
    return (8,56) if fmt=='k4v2' else (0,0)

def write(fmt,ctx,path,sinks=None,local=None):
    kr,vr,mr=rows(fmt)
    if sinks is None or local is None: sinks,local=hp(fmt)
    hp_cap=(sinks+local) if local else 0
    rec=[]
    conv=3*(2*128*8+2*2304)*4
    for _ in range(48): rec.append(conv)
    for slot in range(48):
        il=slot if slot<24 else 6+(slot-24)%12
        full=il%6==5
        cap=ctx if full else min(1024,ctx)
        q=cap*(kr+vr+mr)
        hpone=8*hp_cap*128*2
        rec.append(q+2*hpone+512)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'wb') as f:
        f.write(b'INKWST02'); f.write(struct.pack('<Q',len(rec)))
        z=bytearray(1024*1024)
        for n in rec:
            f.write(struct.pack('<Q',n))
            left=n
            while left:
                x=min(left,len(z)); f.write(z[:x]); left-=x
    print(path,sum(rec)//1048576,'MiB',len(rec),'records')
def write_f16(ctx,path):
    rec=[]
    for wantv in (0,1):
        for slot in range(48):
            il=slot if slot<24 else 6+(slot-24)%12
            full=il%6==5
            cap=ctx if full else min(1024,ctx)
            rec.append(128*cap*8*2)
    conv=3*(2*128*8+2*2304)*4
    for _ in range(48): rec.append(conv)
    import os
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'wb') as f:
        f.write(b'INKWST02'); f.write(struct.pack('<Q',len(rec)))
        z=bytearray(1024*1024)
        for n in rec:
            f.write(struct.pack('<Q',n))
            left=n
            while left:
                x=min(left,len(z)); f.write(z[:x]); left-=x
    print(path,sum(rec)//1048576,'MiB',len(rec),'records')
if __name__=='__main__':
    if sys.argv[1]=='f16': write_f16(int(sys.argv[2]),sys.argv[3])
    else: write(sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4]) if len(sys.argv)>4 else None,int(sys.argv[5]) if len(sys.argv)>5 else None)
