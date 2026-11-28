#!/usr/bin/env python3
import re,sys,statistics
p=sys.argv[1] if len(sys.argv)>1 else 'opt/kvq64/front2g/final-ctx512-ab.txt'
rows=[]
for line in open(p):
    m=re.match(r'round=(\d+) kind=(ctl|cand) RUN \S+ rc=(\d+) TIME_MS_TOK=([0-9.]+) MD5=([0-9a-f]+) VRAM_PEAK_MIB=(\d+)',line)
    if m: rows.append(dict(round=int(m.group(1)),kind=m.group(2),rc=int(m.group(3)),ms=float(m.group(4)),md5=m.group(5),peak=int(m.group(6))))
for kind in ('ctl','cand'):
    xs=[r['ms'] for r in rows if r['kind']==kind]
    print(kind,'n',len(xs),'median',statistics.median(xs) if xs else 0,'vals','/'.join(f'{x:.3f}' for x in xs))
print('pairs',len(rows)//4,'rounds',len(set(r['round'] for r in rows)) if rows else 0,'badmd5',sum(r['md5']!='5cbe8e06' for r in rows),'badpeak',sum(r['peak']>5080 for r in rows),'badrc',sum(r['rc']!=0 for r in rows))
