#!/usr/bin/env python3
"""Detect the six patched CH2 main loops in gemv_moe_s SASS."""
import json
import re
import sys
from collections import Counter

SHIFT_SETS = {
    frozenset((2, 3, 5, 6, 8, 9, 11)): "K1.5",
    frozenset((2, 4, 6, 8, 10, 12, 14)): "K2",
    frozenset((3, 5, 8, 10, 13, 15, 18)): "K2.5",
    frozenset((4, 7, 11, 14, 18, 21, 25)): "K3",
    frozenset((3, 6, 9, 12, 15, 18, 21)): "K3.5",
    frozenset((4, 8, 12, 16, 20, 24, 28)): "K4",
}

text = open(sys.argv[1]).read()
chunk = next(c for c in re.split(r"\n\s*Function : ", text)[1:] if "gemv_moe_s" in c.split("\n")[0])
ins = []
for a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", chunk):
    full = o.strip()
    op = re.sub(r"^@!?P\d\s+", "", full).split()[0].split(".")[0]
    ins.append((int(a, 16), op, full))

rows = []
for addr, op, full in ins:
    if op != "BRA":
        continue
    m = re.search(r"0x([0-9a-f]+)", full)
    if not m:
        continue
    target = int(m.group(1), 16)
    if target >= addr:
        continue
    body = [x for x in ins if target <= x[0] <= addr]
    cnt = Counter(o for _, o, _ in body)
    psl = sum(1 for _, _, f in body if f.startswith("XMAD.PSL") and "0x83dc" in f)
    if cnt["NOP"] != 32 or psl != 32:
        continue
    shifts = []
    for _, o, f in body:
        if o != "SHF":
            continue
        im = re.findall(r",\s*0x([0-9a-f]+),", f)
        if im:
            shifts.append(int(im[0], 16))
    rows.append({
        "start": target,
        "end": addr,
        "k": SHIFT_SETS.get(frozenset(shifts), "unknown"),
        "instructions": len(body),
    })

if len(rows) != 6 or len({(r["start"], r["end"]) for r in rows}) != 6:
    raise SystemExit(f"expected six distinct patched CH2 loops, got {rows}")
json.dump(rows, open(sys.argv[2], "w"), indent=2)
print(" ".join(f"{r['k']}:{r['start']:x}-{r['end']:x}" for r in rows))
