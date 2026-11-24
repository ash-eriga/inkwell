#!/usr/bin/env python3
"""
Extract small weights (suh, svh, norms, biases, markov_w1) from safetensors
and parse the 12 collected real MTP prompts (meta.json, vtok, veri) into
/tmp/drafter_data/proxy_inputs/ for acceptance_proxy.cu.
"""

import glob
import json
import os
import re
import struct

MTP_ST = "/home/ash/models/inkling/mtp.safetensors"
REXL_2 = "/home/ash/models/inkling/rexl3-00002.safetensors"
MTP_DIR = "/tmp/drafter_data/mtp_samples"
OUT_DIR = "/tmp/drafter_data/proxy_inputs"

def read_st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    return hdr, base

def extract_raw(path, hdr, base, name, out_path):
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        data = f.read(e - s)
    with open(out_path, "wb") as out:
        out.write(data)
    return info["dtype"], info["shape"], len(data)

def bf16_bytes_to_f32_bytes(raw):
    # Each bf16 uint16 becomes upper 16 bits of f32
    n = len(raw) // 2
    vals = struct.unpack(f"<{n}H", raw)
    out = struct.pack(f"<{n}I", *(v << 16 for v in vals))
    return out

def extract_bf16_as_f32(path, hdr, base, name, out_path):
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + s)
        raw = f.read(e - s)
    f32 = bf16_bytes_to_f32_bytes(raw)
    with open(out_path, "wb") as out:
        out.write(f32)
    return len(f32) // 4

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Extract unembed suh, svh (fp16 raw uint16)
    h2, b2 = read_st_header(REXL_2)
    extract_raw(REXL_2, h2, b2, "model.llm.unembed.suh", f"{OUT_DIR}/unembed_suh.f16")
    extract_raw(REXL_2, h2, b2, "model.llm.unembed.svh", f"{OUT_DIR}/unembed_svh.f16")
    print("Extracted unembed suh/svh")

    # 2. Extract mtp small tensors (norms/biases as f32, markov_w1 as raw bf16)
    hm, bm = read_st_header(MTP_ST)
    extract_raw(MTP_ST, hm, bm, "markov_head.markov_w1.weight", f"{OUT_DIR}/markov_w1.bf16")
    extract_bf16_as_f32(MTP_ST, hm, bm, "markov_head.rnn_proj.bias", f"{OUT_DIR}/rnn_bias.f32")
    extract_bf16_as_f32(MTP_ST, hm, bm, "markov_head.hidden_out.bias", f"{OUT_DIR}/ho_bias.f32")
    extract_bf16_as_f32(MTP_ST, hm, bm, "norm.weight", f"{OUT_DIR}/norm.f32")
    for l in range(3):
        p = f"layers.{l}"
        extract_bf16_as_f32(MTP_ST, hm, bm, f"{p}.input_layernorm.weight", f"{OUT_DIR}/l{l}_in_ln.f32")
        extract_bf16_as_f32(MTP_ST, hm, bm, f"{p}.post_attention_layernorm.weight", f"{OUT_DIR}/l{l}_post_ln.f32")
        extract_bf16_as_f32(MTP_ST, hm, bm, f"{p}.self_attn.q_norm.weight", f"{OUT_DIR}/l{l}_qn.f32")
        extract_bf16_as_f32(MTP_ST, hm, bm, f"{p}.self_attn.k_norm.weight", f"{OUT_DIR}/l{l}_kn.f32")
    print("Extracted mtp norms, biases, markov_w1")

    # 3. Parse all collected prompts
    samples = []
    for err_path in sorted(glob.glob(f"{MTP_DIR}/*.err")):
        name = os.path.basename(err_path).replace(".err", "")
        sdir = f"{MTP_DIR}/{name}"
        meta_path = f"{sdir}/meta.json"
        if not os.path.exists(meta_path):
            print(f"SKIP {name}: missing meta.json")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        txt = open(err_path).read()
        # First spec r1 vtok and veri
        m_vtok = re.search(r"   vtok=\[([^\]]+)\]", txt)
        m_veri = re.search(r"   veri=\[([^\]]+)\]", txt)
        if not (m_vtok and m_veri):
            print(f"SKIP {name}: missing vtok/veri in {err_path}")
            continue
        vtok = [int(x.strip()) for x in m_vtok.group(1).split(",")]
        veri = [int(x.strip()) for x in m_veri.group(1).split(",")]
        samples.append({
            "name": name,
            "dir": sdir,
            "c": meta["c"],
            "lo": meta["lo"],
            "anchor": meta["anchor"],
            "frow": meta["frow"],
            "vtok": vtok,
            "veri": veri,
        })
    print(f"Parsed {len(samples)} MTP prompt samples:")
    acc1 = sum(1 for s in samples if s["vtok"][1] == s["veri"][0])
    emits = []
    for s in samples:
        n = 0
        while n < 7 and s["vtok"][n + 1] == s["veri"][n]:
            n += 1
        emits.append(n + 1)
        print(f"  {s['name']:14s}: c={s['c']:3d} anchor={s['anchor']:6d} accept={n} d1={s['vtok'][1]} vs v0={s['veri'][0]}")
    print(f"  Baseline FP32 DSpark on {len(samples)} prompts: P(d1)={acc1}/{len(samples)} ({100*acc1/max(1,len(samples)):.1f}%), mean_emit={sum(emits)/max(1,len(emits)):.2f}")

    # Write flat manifest file for C++/CUDA reader
    with open(f"{OUT_DIR}/manifest.txt", "w") as f:
        f.write(f"{len(samples)}\n")
        for s in samples:
            vtok_s = " ".join(str(x) for x in s["vtok"])
            veri_s = " ".join(str(x) for x in s["veri"])
            f.write(f"{s['name']} {s['dir']} {s['c']} {s['lo']} {s['anchor']} {s['frow']} {vtok_s} {veri_s}\n")
    with open(f"{OUT_DIR}/samples.json", "w") as f:
        json.dump(samples, f, indent=2)

if __name__ == "__main__":
    main()
