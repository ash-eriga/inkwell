#!/usr/bin/env python3
# Dump the REAL dense EXL3 trellis tensors (raw little-endian bytes) used by
# ink_launch_exl3_gemv for opt/kbench/sgemv_dense_gold.cu.
# Covers every dense K in {3,3.5,4,5}: wq_du/gate_attn K=3, wk_dv/wv_dv/wo_ud
# K=3.5, wr_du K=5, unembed K=4 (full size, used for both ref and timing).
# usage: python3 -I opt/kbench/sgemv_dense_dump.py [model_dir=~/models/inkling] [out=/tmp/sgemv_dense_gold]
import json, struct, sys, os

model = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/models/inkling")
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/sgemv_dense_gold"
os.makedirs(out, exist_ok=True)

FILES = ["rexl3-00001.safetensors", "rexl3-00002.safetensors"]
WANT = {
    "model.llm.layers.0.attn.wq_du.trellis":     "wq_du",
    "model.llm.layers.0.attn.gate_attn.trellis": "gate_attn",
    "model.llm.layers.0.attn.wr_du.trellis":     "wr_du",
    "model.llm.layers.0.attn.wk_dv.trellis":     "wk_dv",
    "model.llm.layers.0.attn.wv_dv.trellis":     "wv_dv",
    "model.llm.layers.0.attn.wo_ud.trellis":     "wo_ud",
    "model.llm.unembed.trellis":                 "unembed",
}

meta = []
for f in FILES:
    p = os.path.join(model, f)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for name, v in hdr.items():
        if name not in WANT:
            continue
        tk, tn, w16 = v["shape"]
        ka, hk = w16 // 16, (w16 % 16) // 8
        o0, o1 = v["data_offsets"]
        with open(p, "rb") as fh:
            fh.seek(8 + n + o0)
            data = fh.read(o1 - o0)
        short = WANT[name]
        fn = os.path.join(out, f"{short}.bin")
        with open(fn, "wb") as fh:
            fh.write(data)
        meta.append((short, tk, tn, ka, hk, len(data), name))
        print(f"{short}: tk={tk} tn={tn} K={ka + hk / 2} bytes={len(data)}  {name}")

with open(os.path.join(out, "meta.txt"), "w") as fh:
    for short, tk, tn, ka, hk, nb, name in meta:
        fh.write(f"{short} {tk} {tn} {ka} {hk} {nb} {name}\n")
print("meta ->", os.path.join(out, "meta.txt"))
