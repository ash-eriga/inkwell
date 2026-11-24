#!/usr/bin/env python3
"""gemv_moe_ntok time per visit, split by layer residency. usage: nsys_bylayer.py <sqlite> [n_blocks]"""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
names = dict(db.execute("select id, value from StringIds"))
K = db.execute("select start, end, demangledName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
short = lambda i: names.get(i, '?').split('(')[0]
ev = [(s, e, short(n), gx, gy, gz) for (s, e, n, gx, gy, gz) in K]
U = [i for i, e in enumerate(ev) if e[2] == 'inkwell_exl3_union_expand']
# visit order: 0..5, (6..17)x3, 18..23
order = list(range(6)) + [ip for _ in range(3) for ip in range(6, 18)] + list(range(18, 24))
blocks = [U[b:b + 48] for b in range(0, len(U), 48)][-nb:]
gu = collections.defaultdict(list); dn = collections.defaultdict(list)
for blk in blocks:
    for vi, u in enumerate(blk):
        # next two gemv_moe_ntok launches after this union_expand
        j = u; got = 0
        while got < 2 and j < len(ev):
            if ev[j][2] == 'inkwell_exl3_gemv_moe_ntok':
                (gu if got == 0 else dn)[order[vi]].append((ev[j][1] - ev[j][0]) / 1e3); got += 1
            j += 1
print(f"{'visit':>5} {'layer':>5} {'res':>5} {'gate/up ms':>10} {'down ms':>9} {'sum ms':>8}")
tot = {'dev': [0, 0], 'host': [0, 0]}
for vi, il in enumerate(order):
    res = 'dev' if il < 13 else 'host'
    g = sum(gu[il]) / len(gu[il]); d = sum(dn[il]) / len(dn[il])
    tot[res][0] += g; tot[res][1] += d
    print(f"{vi:>5} {il:>5} {res:>5} {g:>10.3f} {d:>9.3f} {g+d:>8.3f}")
for r in ('dev', 'host'):
    print(f"TOTAL {r}: gate/up {tot[r][0]:.1f} ms, down {tot[r][1]:.1f} ms, sum {tot[r][0]+tot[r][1]:.1f} ms")
print(f"TOTAL gemv_moe_ntok per block: {tot['dev'][0]+tot['dev'][1]+tot['host'][0]+tot['host'][1]:.1f} ms")
