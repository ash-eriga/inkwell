#!/usr/bin/env python3
"""Classify shipped scalar dense gemv_sf main loops (audit_loops.py method)."""
import re
import sys
from collections import Counter

# In sgemv_win, windows at bit end D>16 use funnelshift_r(lo,hi,48-D).
# These immediate sets therefore identify the dense K3+ loops unambiguously.
SHIFT_K = {
    frozenset((27, 30)): "K3",
    frozenset((23, 27, 30)): "K3.5",
    frozenset((20, 24, 28)): "K4",
    frozenset((13, 18, 23, 28)): "K5",
}
CLASS = {
    "LDG": "load-global", "LDS": "load-shared", "SHF": "extract", "SHL": "extract",
    "BFE": "extract", "PRMT": "extract", "LOP3": "extract", "XMAD": "product",
    "IDP": "bytesum", "I2F": "convert", "FFMA": "accumulate", "SHFL": "shuffle",
    "IADD": "address/loop", "IADD32I": "address/loop", "ISETP": "address/loop",
    "BRA": "address/loop", "DEPBAR": "control", "NOP": "nop",
}

text = open(sys.argv[1]).read()
chunks = re.split(r"\n\s*Function : ", text)[1:]
hits = [c for c in chunks if "gemv_sf" in c.split("\n", 1)[0] and "gemv_sf_ch2" not in c.split("\n", 1)[0]]
chunk = hits[0] if hits else text
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
    lo = int(m.group(1), 16)
    if lo >= addr:
        continue
    body = [x for x in ins if lo <= x[0] <= addr]
    ops = Counter(x[1] for x in body)
    if len(body) < 200 or ops["IDP"] != 32 or ops["FFMA"] != 32 or ops["SHFL"]:
        continue
    shifts = []
    for _, o, f in body:
        if o != "SHF":
            continue
        im = re.findall(r",\s*0x([0-9a-f]+),", f)
        if im:
            shifts.append(int(im[0], 16))
    name = SHIFT_K.get(frozenset(shifts))
    if name is None:
        # The low-K extraction lowering has no unique funnel-immediate set.
        # Use the ring stride computation immediately before the loop instead:
        # 48/64/80/128 bytes for K1.5/K2/K2.5/K4 (the last as a shift by 7).
        pre = [f for a, _o, f in ins if lo - 0x200 <= a < lo]
        if any("SHF.L.U64" in f and "0x7" in f for f in pre):
            name = "K4"
        elif any("SHF.L.U64" in f and "0x6" in f for f in pre):
            name = "K2"
        elif any("0xa0" in f for f in pre):
            name = "K5"
        elif any("0x70" in f for f in pre):
            name = "K3.5"
        elif any("0x60" in f for f in pre):
            name = "K3"
        elif any("0x50" in f for f in pre):
            name = "K2.5"
        elif any("0x30" in f for f in pre):
            name = "K1.5"
        else:
            name = "unknown"
    rows.append((lo, addr, name, body))

print(f"{'K':6} {'total':>5} {'ins/w':>6} {'core':>4} {'extr':>4} {'ldg':>3} {'lds':>3} {'addr':>4} {'ctl':>3} {'range':>15}")
for lo, hi, name, body in rows:
    cls = Counter(CLASS.get(o, "other") for _, o, _ in body)
    core = cls["product"] + cls["bytesum"] + cls["convert"] + cls["accumulate"]
    print(f"{name:6} {len(body):5d} {len(body)/32:6.3f} {core:4d} {cls['extract']:4d} "
          f"{cls['load-global']:3d} {cls['load-shared']:3d} {cls['address/loop']:4d} {cls['control']:3d} {lo:#x}-{hi:#x}")
