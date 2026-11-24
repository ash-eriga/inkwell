#!/usr/bin/env python3
"""Per-block kernel breakdown from an nsys sqlite export of a spec run.
usage: nsys_rounds.py <report.sqlite> [n_blocks] [topN]
A verify block = 48 `inkwell_exl3_union_expand` launches (one per MoE visit);
its window runs to the last non-draft kernel before the next draft kernel.
Everything up to the next block's first kernel is that round's "between" time
(draft propose + ctx_append + host argmax + logits readback + graph build).
"""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 0
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 40
names = dict(db.execute("select id, value from StringIds"))
K = db.execute("select start, end, demangledName, streamId, registersPerThread, gridX, gridY, gridZ, blockX, blockY from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
def short(i):
    s = names.get(i, '?')
    for p in ('void ', 'const ', '(bool)1', '(bool)0', '__half', 'unsigned ', 'long ', 'int ', 'float ', '(int)', ' '):
        s = s.replace(p, '')
    return s.split('<')[0].split('(')[0]
ev = [(s, e, short(n), st, reg, gx, gy, gz, bx, by) for (s, e, n, st, reg, gx, gy, gz, bx, by) in K]
t0 = ev[0][0]
isdraft = lambda k: k.startswith('inkwell_draft')
U = [i for i, e in enumerate(ev) if e[2] == 'inkwell_exl3_union_expand']
blocks = []
for b in range(0, len(U), 48):
    s_i = U[b]; e_i = U[min(b + 47, len(U) - 1)]
    while e_i + 1 < len(ev) and not isdraft(ev[e_i + 1][2]): e_i += 1
    blocks.append((s_i, e_i))
print(f"kernels={len(ev)} union_expand={len(U)} blocks={len(blocks)} span={(ev[-1][1]-t0)/1e6:.0f}ms")
sel = blocks[-nb:] if nb else blocks
agg = collections.Counter(); cnt = collections.Counter(); reg_seen = {}
print(f"{'blk':>3} {'t_start':>8} {'verify_win':>10} {'gpu_busy':>9} {'idle':>6} {'after':>7} {'gap>20us':>8} {'n_kern':>6}")
info = []
for bi in range(len(sel)):
    s_i, e_i = sel[bi]
    v = ev[s_i:e_i + 1]
    win = (v[-1][1] - v[0][0]) / 1e6
    busy = sum(e - s for s, e, *_ in v) / 1e6
    after = (sel[bi + 1][0] - v[-1][1]) / 1e6 if bi + 1 < len(sel) else 0
    gaps = sum(1 for a, b2 in zip(v, v[1:]) if b2[0] - a[1] > 20000)
    info.append((win, busy, after, gaps, len(v)))
    print(f"{bi+1:>3} {(v[0][0]-t0)/1e6:>8.0f} {win:>10.1f} {busy:>9.1f} {win-busy:>6.1f} {after:>7.1f} {gaps:>8d} {len(v):>6d}")
    if bi + 1 < len(sel):
        for s, e, n, st, reg, *_ in v: agg[n] += (e - s) / 1e6; cnt[n] += 1; reg_seen[n] = reg
n = len(sel) - 1 if nb else len(sel)
print(f"MEAN verify_win={sum(i[0] for i in info[:-1])/n:.1f} busy={sum(i[1] for i in info[:-1])/n:.1f} after={sum(i[2] for i in info[:-1])/n:.1f}")
# what runs "after" (between blocks)
bet = collections.Counter(); betc = collections.Counter()
for bi in range(len(sel) - 1):
    for s, e, nm, *_ in ev[sel[bi][1] + 1:sel[bi + 1][0]]: bet[nm] += (e - s) / 1e6; betc[nm] += 1
tb = sum(bet.values())
print(f"\n--- between blocks (draft+ctx+host): kernel busy {tb/n:.1f} ms/round of {sum(i[2] for i in info[:-1])/n:.1f} ms")
for k, t in bet.most_common(14):
    print(f"{k[:44]:<44} {t/n:>8.2f} ms  n={betc[k]/n:>6.1f}  avg={t/betc[k]*1e3:>7.1f}us")
print(f"\n--- inside verify: {'kernel':<40} {'ms/blk':>8} {'%busy':>6} {'n/blk':>7} {'avg us':>8} {'regs':>5}")
for k, t in agg.most_common(topn):
    print(f"{k[:40]:<40} {t/n:>8.1f} {100*t/tot_v if (tot_v:=sum(agg.values())) else 0:>6.1f} {cnt[k]/n:>7.1f} {t/cnt[k]*1e3:>8.1f} {str(reg_seen[k]):>5}")
