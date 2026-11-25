#!/usr/bin/env python3
"""Count candidate CH2 tile loops in the patched production SASS."""
import re
import sys
from collections import Counter

text = open(sys.argv[1]).read()
chunk = next(c for c in re.split(r"\n\s*Function : ", text)[1:] if "gemv_moe_s" in c.split("\n")[0])
ins = []
for a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", chunk):
    op = re.sub(r"^@!?P\d\s+", "", o.strip()).split()[0].split(".")[0]
    ins.append((int(a, 16), op, o.strip()))
rows = []
for addr, op, full in ins:
    if op.split(".")[0] != "BRA":
        continue
    m = re.search(r"0x([0-9a-f]+)", full)
    if not m:
        continue
    target = int(m.group(1), 16)
    if target >= addr:
        continue
    body = [(a, o, f) for a, o, f in ins if target <= a <= addr]
    psl = sum(1 for _, o, full in body if full.startswith("XMAD.PSL") and "0x83dc" in full)
    if psl not in (32, 64):
        continue
    cnt = Counter(o for _, o, _ in body)
    rows.append((target, addr, len(body), psl, cnt, body))

# Deduplicate nested/overlapping candidates by end address, keeping the longest.
rows.sort(key=lambda x: (x[1], -x[2]))
dedup = []
for r in rows:
    if dedup and dedup[-1][1] == r[1]:
        continue
    dedup.append(r)
order = ["K1.5", "K2", "K2.5", "K3", "K3.5", "K4", "K5"]
print(f"{'K':5} {'start':>6} {'end':>6} {'ins':>4} {'noNOP':>5} {'NOP':>3} {'PSL':>3} {'XMAD':>4} {'SHF':>3} {'IDP':>3} {'FADD':>4} {'FFMA':>4} {'LDS':>3} {'LDG':>3} {'ins/w':>6} {'noNOP/w':>7}")
for i, (lo, hi, n, psl, cnt, body) in enumerate(dedup[:7]):
    weights = 32 if psl == 32 else 64
    nonop = n - cnt["NOP"]
    print(f"{order[i]:5} {lo:6x} {hi:6x} {n:4d} {nonop:5d} {cnt['NOP']:3d} {psl:3d} {cnt['XMAD']:4d} {cnt['SHF']:3d} {cnt['IDP']:3d} {cnt['FADD']:4d} {cnt['FFMA']:4d} {cnt['LDS']:3d} {cnt['LDG']:3d} {n/weights:6.3f} {nonop/weights:7.3f}")
    # print(' '.join(f'{k}:{v}' for k,v in sorted(cnt.items())))
