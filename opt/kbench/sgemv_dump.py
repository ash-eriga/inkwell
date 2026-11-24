#!/usr/bin/env python3
# Dump selected real EXL3 expert trellis tensors (raw little-endian bytes) for
# opt/kbench/sgemv_gold.cu. Picks experts so every K in {1.5,2,2.5,3,3.5,4}
# appears for gate, up and down.
# usage: python3 -I opt/kbench/sgemv_dump.py [model_dir=~/models/inkling] [out=/tmp/sgemv_gold]
import json, struct, re, sys, os, collections

model = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/models/inkling")
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sgemv_gold"
os.makedirs(out, exist_ok=True)

FILES = ["rexl3-00001.safetensors", "rexl3-00002.safetensors"]
PAT = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.trellis")

# candidates[proj][W] = list of (file, name, off0, off1)
cand = collections.defaultdict(lambda: collections.defaultdict(list))
hdrs = {}
for f in FILES:
    p = os.path.join(model, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdrs[f] = n
    for name, v in hdr.items():
        m = PAT.match(name)
        if not m:
            continue
        proj = m.group(3)
        w = v["shape"][2]
        cand[proj][w].append((f, name, v["data_offsets"][0], v["data_offsets"][1]))

def K_of(w):
    return w // 16, (w % 16) // 8  # ka, hk

# per-slot K lists (W = shape[2]); each list covers all six geometries
want = {
    "gate": [24, 32, 40, 48, 56, 64, 32, 48],
    "up":   [32, 48, 24, 64, 40, 56, 48, 32],
    "down": [24, 32, 40, 48, 56, 64, 40, 56],
}
PROJ = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}

meta = []
for kind, ws in want.items():
    proj = PROJ[kind]
    for slot, w in enumerate(ws):
        lst = cand[proj][w]
        if len(lst) > slot:
            f, name, o0, o1 = lst[slot]
        else:
            f, name, o0, o1 = lst[0]  # reuse first when the pool is exhausted
        p = os.path.join(model, f)
        with open(p, "rb") as fh:
            fh.seek(8 + hdrs[f] + o0)
            data = fh.read(o1 - o0)
        ka, hk = K_of(w)
        fn = os.path.join(out, f"{kind}_s{slot}.bin")
        with open(fn, "wb") as fh:
            fh.write(data)
        meta.append((kind, slot, ka, hk, name, len(data)))
        print(f"{kind} s{slot}: W={w} K={ka + hk / 2} bytes={len(data)}  {name}")

with open(os.path.join(out, "meta.txt"), "w") as fh:
    for kind, slot, ka, hk, name, nb in meta:
        fh.write(f"{kind} {slot} {ka} {hk} {nb} {name}\n")
print("meta ->", os.path.join(out, "meta.txt"))
