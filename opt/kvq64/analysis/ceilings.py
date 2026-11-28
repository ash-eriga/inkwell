#!/usr/bin/env python3
# Payload bytes/token/visit for the implemented layouts, and ideal attention time.
rows={'f16':(4096,0),'int8':(2048+96,0),'fp8':(2048,0),'k8v4':(1536+96,0),'k4v2':(864,64*8*128*2*2)}
for bw in (150,217):
 print('bandwidth',bw,'GB/s')
 for f,(b,hp) in rows.items():
  print(f, end='')
  for c in (4096,16384,32768,65536):
   sw=min(c,1024)
   bytes_tok=8*c*b+40*sw*b+(8 if c>1024 else 0)*hp
   print(f' ctx{c}={bytes_tok/1e9/bw*1e3:.2f}ms/{(8*c*b+40*sw*b+hp)/1048576:.0f}MiBKV',end='')
  print()
