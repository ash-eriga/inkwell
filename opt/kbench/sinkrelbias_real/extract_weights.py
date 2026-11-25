#!/usr/bin/env python3
import json, struct, sys
from pathlib import Path

def tensor(repo,name):
    idx=json.loads((Path(repo)/'model.safetensors.index.json').read_text())['weight_map']
    shard=idx[name]
    with open(Path(repo)/shard,'rb') as f:
        hlen=struct.unpack('<Q',f.read(8))[0]
        hdr=json.loads(f.read(hlen))
        e=hdr[name]
        a,z=e['data_offsets']
        f.seek(8+hlen+a)
        return e['shape'],f.read(z-a)

def bf16_to_f32_bytes(d):
    out=bytearray(len(d)*2)
    for i in range(0,len(d),2):
        u=d[i]|(d[i+1]<<8)
        out[i*2:i*2+4]=struct.pack('<I',u<<16)
    return bytes(out)

def transpose_raw(d,r,c,esz):
    out=bytearray(len(d))
    for a in range(r):
        for b in range(c):
            out[(b*r+a)*esz:(b*r+a+1)*esz]=d[(a*c+b)*esz:(a*c+b+1)*esz]
    return bytes(out)

def main():
    repo,layer,outdir=sys.argv[1],int(sys.argv[2]),Path(sys.argv[3])
    outdir.mkdir(parents=True,exist_ok=True)
    pre=f'model.llm.layers.{layer}'
    shape,sink=tensor(repo,pre+'.attn.attn_sink_proj.weight')
    (outdir/'sink.bf16').write_bytes(sink)
    shape,rel=tensor(repo,pre+'.attn.rel_logits_proj.proj')
    rt=transpose_raw(rel,16,1024,2)
    pad=rt+bytes(32)
    (outdir/'rel.bf16').write_bytes(pad)
    (outdir/'rel.f32').write_bytes(bf16_to_f32_bytes(pad))
    print('wrote',outdir,'sink',len(sink),'rel',len(pad))
if __name__=='__main__': main()
