#!/usr/bin/env python3
import re,sys,statistics,collections
rows=collections.defaultdict(list)
for line in open(sys.argv[1] if len(sys.argv)>1 else '/tmp/kvq-matrix.out',errors='replace'):
 m=re.search(r'ctx=(\d+) fmt=(\S+) rep=(\d+) decode_rc=(\d+) ms=([\d.-]+).*?peak=(\d+) md5=(\S+) kvq=(\S+) hp=(\S+)',line)
 if m:
  ctx,fmt,rep,rc,ms,peak,md5,kvq,hp=m.groups()
  rows[(fmt,int(ctx))].append((int(rc),float(ms),int(peak),md5,kvq,hp))
for (fmt,ctx),v in sorted(rows.items(),key=lambda x:(x[0][0],x[0][1])):
 ok=[x for x in v if x[0]==0 and x[1]>0]
 if ok:
  print(fmt,ctx,'n',len(ok),'ms_med',round(statistics.median(x[1] for x in ok),3),'ms_all',[round(x[1],2) for x in ok],'peak_max',max(x[2] for x in ok),'kvq',ok[0][4],'hp',ok[0][5],'md5',sorted(set(x[3] for x in ok)))
 else: print(fmt,ctx,'all_failed',len(v))
