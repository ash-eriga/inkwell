#!/usr/bin/env python3
"""Aggregate an nsys sqlite export into prefill phase timings.

usage: nsys_prefill_phase.py <report.sqlite>
The encode window is first..last prefill-specific kernel. Per-kernel phase time
is summed by kernel class. Wall-clock GPU busy uses interval unions across all
kernels and memcpys, so overlapped copy streams do not count twice. Host/launch
idle is the window span minus that union.
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
    if n.startswith('inkwell_flash_prefill') or n.startswith('inkwell_pf_exact_attn'): return 'attention'
    if n.startswith('inkwell_exl3_moe_grouped'): return 'moe_expert_gemm'
    if n.startswith('inkwell_exl3_moe_group_sort') or n.startswith('inkwell_pf_router8'): return 'moe_sort_router'
    if n.startswith('inkwell_moe_had') or n.startswith('inkwell_moe_situ') or n.startswith('inkwell_moe_weighted') or n.startswith('inkwell_moe_fill'): return 'moe_hadamard_scatter'
    if n.startswith('inkwell_exl3_gemm'): return 'dense_exl3_core'
    if n.startswith('inkwell_exl3_had_in_prefill') or n.startswith('inkwell_exl3_had_out_prefill'): return 'dense_exl3_hadamard'
    if n.startswith('inkwell_pf_dense_gemm') or n.startswith('inkwell_pf_dense_mmvf'): return 'dense_small_gemm'
    if n.startswith('inkwell_conv_prefill'): return 'conv'
    return 'glue'

agg = collections.Counter(); cnt = collections.Counter(); kop = collections.Counter(); kcnt = collections.Counter()
for s, e, n in win:
    dt = (e-s)/1e6
    agg[cls(n)] += dt; cnt[cls(n)] += 1
    kop[n] += dt; kcnt[n] += 1

cp = []
try:
    cp = db.execute("select start,end,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=?", (t0,t1)).fetchall()
except sqlite3.Error:
    pass

h2d = d2h = d2d = othercopy = 0.0
copy_iv = []
for s,e,k in cp:
    a, b = max(s,t0), min(e,t1)
    ov=(b-a)/1e6
    if ov<=0: continue
    copy_iv.append((a,b))
    if k==1: h2d+=ov
    elif k==2: d2h+=ov
    elif k==8: d2d+=ov
    else: othercopy+=ov

def union_ms(iv):
    total=0.0
    ce=None; cs=None
    for a,b in sorted(iv):
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else:
            total+=(ce-cs)/1e6; cs,ce=a,b
    if cs is not None: total+=(ce-cs)/1e6
    return total

kernel_iv=[(s,e) for s,e,_ in win]
kernel_union=union_ms(kernel_iv)
copy_union=union_ms(copy_iv)
gpu_union=union_ms(kernel_iv+copy_iv)
busy=sum(agg.values())
span=(t1-t0)/1e6
print(f"window_ms={span:.3f} kernel_sum_ms={busy:.3f} kernel_union_ms={kernel_union:.3f} copy_union_ms={copy_union:.3f} gpu_busy_union_ms={gpu_union:.3f} host_launch_gap_ms={span-gpu_union:.3f} h2d_ms={h2d:.3f} d2h_ms={d2h:.3f} d2d_ms={d2d:.3f} other_copy_ms={othercopy:.3f}")
print('phase ms pct_kernel_sum launches')
for k,t in agg.most_common():
    print(f"{k} {t:.3f} {100*t/busy if busy else 0:.1f} {cnt[k]}")
print(f"copies_h2d {h2d:.3f} - -")
print(f"copies_d2h {d2h:.3f} - -")
print(f"copies_d2d {d2d:.3f} - -")
print(f"host_launch_idle {span-gpu_union:.3f} - -")
print('\nkernels')
print('kernel ms launches avg_us')
for k,t in kop.most_common(30):
    print(f"{k} {t:.3f} {kcnt[k]} {1000*t/kcnt[k]:.1f}")
