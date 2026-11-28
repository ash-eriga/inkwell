#!/usr/bin/env python3
import re,sys,statistics,collections
rows=[]
for line in open(sys.argv[1]):
    m=re.match(r'round=(\d+) bin=(\S+) fmt=(\S+) decode_rc=(\d+) ms=([0-9.]+) .*peak=(\d+) md5=(\S+) kvq=(\S+) hp=(\S+) kvtotal=(\S+) precache=(\S+) budget=(\S+) table=(\S+)',line)
    if m:
        rows.append(dict(round=int(m[1]),bin=m[2],fmt=m[3],rc=int(m[4]),ms=float(m[5]),peak=int(m[6]),md5=m[7],kvq=m[8],hp=m[9],kvtotal=m[10],precache=m[11],budget=m[12],table=m[13]))
g=collections.defaultdict(list)
for r in rows: g[(r['bin'],r['fmt'])].append(r)
for (b,f),rs in sorted(g.items()):
    ms=[r['ms'] for r in rs]
    r=rs[len(rs)//2]
    print(f"{b} {f} n={len(rs)} median_ms={statistics.median(ms):.3f} all_ms={','.join(f'{x:.3f}' for x in ms)} peak={max(x['peak'] for x in rs)} md5s={','.join(sorted(set(x['md5'] for x in rs)))} kvq={r['kvq']} hp={r['hp']} kvtotal={r['kvtotal']} table={r['table']}")
