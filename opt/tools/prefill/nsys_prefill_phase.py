#!/usr/bin/env python3
"""Aggregate an nsys sqlite export into prefill phase timings.

usage: nsys_prefill_phase.py <report.sqlite>
The encode window is first..last prefill-specific kernel. GPU time is summed by
kernel class; host/launch-gap is window span minus GPU kernel busy time. H2D
copies inside the window are reported separately when CUPTI memcpy rows exist.
"""
import collections, sqlite3, sys

path = sys.argv[1]
db = sqlite3.connect(path)
names = dict(db.execute("select id, value from StringIds"))
rows = db.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()

def short(i):
    s = names.get(i, '?')
    return s.split('(')[0].split('<')[0].replace('void ', '')

ev = [(s, e, short(n)) for s, e, n in rows]
pref = lambda n: (n.startswith('inkwell_pf_') or n.startswith('inkwell_flash_prefill')
    or n.startswith('inkwell_exl3_gemm') or n.startswith('inkwell_exl3_had_in_prefill')
    or n.startswith('inkwell_exl3_had_out_prefill') or n.startswith('inkwell_exl3_moe')
    or n.startswith('inkwell_moe_') or n.startswith('inkwell_conv_prefill'))
idx = [i for i, x in enumerate(ev) if pref(x[2])]
if not idx:
    raise SystemExit('no prefill kernels found')
lo, hi = idx[0], idx[-1]
win = ev[lo:hi+1]
t0, t1 = win[0][0], win[-1][1]

def cls(n):
    if n.startswith('inkwell_flash_prefill'): return 'attention'
    if n.startswith('inkwell_exl3_moe_grouped'): return 'moe_expert_gemm'
    if n.startswith('inkwell_exl3_gemm'): return 'dense_exl3_gemm'
    if n.startswith('inkwell_pf_dense_gemm'): return 'dense_small_gemm'
    if n.startswith('inkwell_exl3_moe') or n.startswith('inkwell_moe_'): return 'moe_glue'
    if n.startswith('inkwell_conv_prefill'): return 'conv'
    return 'glue'

agg = collections.Counter(); cnt = collections.Counter(); kop = collections.Counter(); kcnt = collections.Counter()
for s, e, n in win:
    dt = (e-s)/1e6
    agg[cls(n)] += dt; cnt[cls(n)] += 1
    kop[n] += dt; kcnt[n] += 1

# Memcpy rows overlapping the kernel window. Schema is stable in current nsys,
# but keep this best-effort because older exports may lack the table.
h2d = d2d = othercopy = 0.0
try:
    cp = db.execute("select start,end,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=?", (t0,t1)).fetchall()
    for s,e,k in cp:
        ov=(min(e,t1)-max(s,t0))/1e6
        if ov<=0: continue
        if k==1: h2d+=ov
        elif k==8: d2d+=ov
        else: othercopy+=ov
except sqlite3.Error:
    pass

busy=sum(agg.values())
span=(t1-t0)/1e6
print(f"window_ms={span:.3f} kernel_busy_ms={busy:.3f} host_launch_gap_ms={span-busy:.3f} h2d_ms={h2d:.3f} d2d_ms={d2d:.3f} other_copy_ms={othercopy:.3f}")
print('phase ms pct_busy launches')
for k,t in agg.most_common():
    print(f"{k} {t:.3f} {100*t/busy if busy else 0:.1f} {cnt[k]}")
print(f"host_launch_gap {span-busy:.3f} - -")
print('\nkernels')
print('kernel ms launches avg_us')
for k,t in kop.most_common(30):
    print(f"{k} {t:.3f} {kcnt[k]} {1000*t/kcnt[k]:.1f}")
