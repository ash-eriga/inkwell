#!/usr/bin/env python3
"""
Question 2 (Coverage): Measure how often Inkling-2-Mini's ACTUAL serial greedy
tokens (640 tokens across 5 varied prompts: prose, Python, Rust, technical,
factual) fall inside the top-V' reduced vocabulary set.

Compares:
  1. Independent corpus frequency ranking (vocab_rank_corpus.bin)
     -- built from local Python stdlib + C headers + /usr/share/doc + manpages
        with ZERO overlap with the 5 evaluation prompts.
  2. Naive BPE ID prefix (id < V') -- demonstrates why Inkling's vocab layout
     (where leading-space English words sit at 149860..199999) requires a
     remapped uint32_t vocab_ids[V'] table.
"""

import glob
import json
import os
import re
import struct

VOCAB = 200058
SERIAL_DIR = "/tmp/drafter_data/serial"
RANK_BIN = "opt/kbench/drafter/vocab_rank_corpus.bin"

def load_serial_tokens():
    prompts = {}
    pat = re.compile(r"^gen \d+: id=\d+ next=(\d+) ")
    for p in sorted(glob.glob(f"{SERIAL_DIR}/*.err")):
        name = os.path.basename(p).replace(".err", "")
        toks = []
        with open(p, "r") as f:
            for line in f:
                m = pat.match(line)
                if m:
                    toks.append(int(m.group(1)))
        prompts[name] = toks
    return prompts

def main():
    prompts = load_serial_tokens()
    with open(RANK_BIN, "rb") as f:
        order = list(struct.unpack(f"<{VOCAB}I", f.read(VOCAB * 4)))
    rank_of = [0] * VOCAB
    for r, tid in enumerate(order):
        rank_of[tid] = r

    all_toks = [t for ts in prompts.values() for t in ts]
    print(f"Loaded {len(all_toks)} real target greedy tokens across {len(prompts)} prompts:")
    for k, ts in prompts.items():
        print(f"  {k:10s}: {len(ts)} tokens ({len(set(ts))} unique)")

    v_list = [4096, 8192, 16384, 32768, 65536]
    print("\n=== Coverage by Independent Corpus Frequency Ranking (vocab_rank_corpus.bin) ===")
    hdr = f"{'V_prime':>8} | {'Overall':>8} | " + " | ".join(f"{k:>8}" for k in prompts)
    print(hdr)
    print("-" * len(hdr))
    cov_corpus = {}
    for vp in v_list:
        ov = sum(1 for t in all_toks if rank_of[t] < vp) / len(all_toks)
        per = {k: sum(1 for t in ts if rank_of[t] < vp) / len(ts) for k, ts in prompts.items()}
        cov_corpus[vp] = {"overall": ov, "per_domain": per}
        print(f"{vp:8d} | {100*ov:7.2f}% | " + " | ".join(f"{100*per[k]:7.2f}%" for k in prompts))

    print("\n=== Coverage by Naive Token ID Prefix (id < V') ===")
    print(hdr)
    print("-" * len(hdr))
    cov_naive = {}
    for vp in v_list:
        ov = sum(1 for t in all_toks if t < vp) / len(all_toks)
        per = {k: sum(1 for t in ts if t < vp) / len(ts) for k, ts in prompts.items()}
        cov_naive[vp] = {"overall": ov, "per_domain": per}
        print(f"{vp:8d} | {100*ov:7.2f}% | " + " | ".join(f"{100*per[k]:7.2f}%" for k in prompts))

    # Also check how many missed tokens at V'=16K/32K were simply unseen in the 2MB corpus
    with open("opt/kbench/drafter/vocab_rank_meta.json") as f:
        meta = json.load(f)
    uniq_corp = meta["unique_corpus_tokens"]
    print(f"\nNote: corpus had {meta['total_corpus_tokens']} tokens, {uniq_corp} unique tokens.")
    for vp in [8192, 16384, 32768]:
        missed = [t for t in all_toks if rank_of[t] >= vp]
        print(f"  V'={vp}: {len(missed)}/{len(all_toks)} missed ({100*len(missed)/len(all_toks):.2f}%), sample ids={missed[:12]}")

    out = {
        "n_tokens": len(all_toks),
        "prompts": {k: len(v) for k, v in prompts.items()},
        "corpus_ranked": cov_corpus,
        "naive_id_prefix": cov_naive,
    }
    with open("opt/kbench/drafter/vocab_coverage_out.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote opt/kbench/drafter/vocab_coverage_out.json")

if __name__ == "__main__":
    main()
