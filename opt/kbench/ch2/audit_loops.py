#!/usr/bin/env python3
"""Classify instructions in the six CH2 main loops."""
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
CLASS = {
    "LDG": "load-global", "LDS": "load-shared", "SHF": "extract", "SHL": "extract",
    "PRMT": "extract", "LOP3": "extract", "XMAD": "product", "IDP": "bytesum",
    "I2F": "convert", "FADD": "convert", "FFMA": "accumulate",
    "IADD": "address/loop", "IADD32I": "address/loop", "ISETP": "address/loop",
    "BRA": "address/loop", "DEPBAR": "control", "NOP": "nop", "MOV": "other",
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
    psl = sum(1 for _, _, f in body if f.startswith("XMAD.PSL") and "0x83dc" in f)
    if psl != 32:
        continue
    shifts = []
    for _, o, f in body:
        if o == "SHF":
            im = re.findall(r",\s*0x([0-9a-f]+),", f)
            if im:
                shifts.append(int(im[0], 16))
    rows.append((target, addr, SHIFT_SETS.get(frozenset(shifts), "unknown"), body))
seen = set()
out = []
for row in rows:
    key = (row[0], row[1])
    if key not in seen:
        seen.add(key)
        out.append(row)
if len(out) != 6:
    raise SystemExit(f"expected six CH2 loops, got {[(hex(r[0]), hex(r[1])) for r in out]}")

print(f"{'K':5} {'total':>5} {'core':>4} {'extr':>4} {'ldg':>3} {'lds':>3} {'addr':>4} {'ctl':>3} {'oth':>3} {'ins/w':>6}")
for lo, hi, name, body in out:
    cls = Counter(CLASS.get(o, "other") for _, o, _ in body)
    core = cls["product"] + cls["bytesum"] + cls["convert"] + cls["accumulate"]
    total = len(body)
    print(f"{name:5} {total:5d} {core:4d} {cls['extract']:4d} {cls['load-global']:3d} {cls['load-shared']:3d} "
          f"{cls['address/loop']:4d} {cls['control']:3d} {cls['other'] + cls['nop']:3d} {total/32:6.3f}")
    if "--verbose" in sys.argv:
        ops = Counter(o for _, o, _ in body)
        print("  " + " ".join(f"{k}:{v}" for k, v in sorted(ops.items())))
