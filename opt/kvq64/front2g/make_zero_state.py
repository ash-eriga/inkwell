#!/usr/bin/env python3
# Front2G variant: create layout-exact zero states for K4V4 scope=all/full.
import struct,sys,os

def rows(fmt):
    return {'int8':(1024,1024,96),'fp8':(1024,1024,0),'k8v4':(1024,512,96),'k8v4g32':(1024,512,192),'k8v6':(1024,768,96),'k10v6':(1280,768,96),'k4v4':(512,512,96),'k4v2':(512,256,96)}[fmt]

def write(fmt,ctx,path,scope='all'):
    kr,vr,mr=rows(fmt)
    rec=[]
    # Physical-layer/slot mapping used by Runner: slots 0..47, full when il%6==5.
    full=[]
    for slot in range(48):
        il=slot if slot<24 else 6+(slot-24)%12
        full.append(il%6==5)
    quant=[scope=='all' or f for f in full]
    conv=3*(2*128*8+2*2304)*4
    # tensor order is all K arenas, then all V arenas, then conv states.
    for wantv in (0,1):
        for slot,f in enumerate(full):
            if not quant[slot]:
                cap=ctx if f else min(1024,ctx)
                rec.append(128*cap*8*2)
    for _ in range(48): rec.append(conv)
    for slot,f in enumerate(full):
        if quant[slot]:
            cap=ctx if f else min(1024,ctx)
            rec.append(cap*(kr+vr+mr)+512)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,'wb') as f:
        f.write(b'INKWST02'); f.write(struct.pack('<Q',len(rec)))
        z=bytearray(1024*1024)
        for n in rec:
            f.write(struct.pack('<Q',n))
            left=n
            while left:
                x=min(left,len(z)); f.write(z[:x]); left-=x
    print(path,sum(rec)//1048576,'MiB',len(rec),'records','scope',scope)
if __name__=='__main__':
    write(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4] if len(sys.argv)>4 else 'all')
