#!/usr/bin/env python3
"""
Build an independent token frequency ranking over Inkling-2-Mini's 200,058-token
vocabulary using ONLY local system corpora completely disjoint from our test prompts:
  1. System documentation (/usr/share/doc/**/*.{txt,md,README,NEWS,FAQ,ChangeLog})
  2. Linux man pages (/usr/share/man/man{1,2,3,5,7,8}/*.gz) -- rich English prose + tech
  3. Python 3.14 standard library (/usr/lib/python3.14/**/*.py) -- Python code
  4. C/C++ system headers (/usr/include/**/*.h) -- systems code

Tie-breaker for corpus-unseen tokens (also 100% independent of evaluation):
  ASCII/printable Latin BPE tokens in GGUF BPE merge order (preferring Inkling's
  English leading-space block 149860..199999 and ASCII tokens before rare CJK/emoji).
"""

import glob
import gzip
import json
import os
import re
import struct
from collections import Counter

GGUF_PATH = "/home/ash/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf"
VOCAB = 200058
EOS_ID = 200006

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")

def skip_or_read_val(f, t, want=False):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        b = f.read(sizes[t])
        return b if want else None
    if t == 8:
        s = read_str(f)
        return s if want else None
    if t == 9:
        et, cnt = struct.unpack("<IQ", f.read(12))
        if et in sizes:
            f.seek(sizes[et] * cnt, 1)
            return None
        if et == 8:
            if want:
                return [read_str(f) for _ in range(cnt)]
            for _ in range(cnt):
                n = struct.unpack("<Q", f.read(8))[0]
                f.seek(n, 1)
            return None
        for _ in range(cnt):
            skip_or_read_val(f, et, False)
        return None
    raise ValueError(f"Unknown GGUF type {t}")

def load_gguf_bpe(path):
    tokens = None
    merges = None
    with open(path, "rb") as f:
        magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
        for _ in range(nkv):
            k = read_str(f)
            vt = struct.unpack("<I", f.read(4))[0]
            want = k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
            v = skip_or_read_val(f, vt, want=want)
            if k == "tokenizer.ggml.tokens":
                tokens = v
            elif k == "tokenizer.ggml.merges":
                merges = v
            if tokens is not None and merges is not None:
                break
    return tokens, merges

def gpt2_byte_map():
    b2c = {}
    n = 0
    for b in range(256):
        if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
            b2c[b] = chr(b)
        else:
            b2c[b] = chr(0x100 + n)
            n += 1
    return b2c

PRE_RE = re.compile(
    r" ?[A-Za-z]+(?:'[sStTmMdD]|'[rR][eE]|'[vV][eE]|'[lL][lL])?"
    r"|[0-9]{1,3}"
    r"| ?[^\sA-Za-z0-9]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
MAN_CLEAN_RE = re.compile(r"^\.[A-Za-z]{1,4}\s*|\x1b\[[0-9;]*m|\\f[A-Z]|\\\(..")

class FastBPE:
    def __init__(self, tokens, merges):
        self.tok2id = {t: i for i, t in enumerate(tokens)}
        self.ranks = {}
        for i, m in enumerate(merges):
            sep = m.find(" ", 1)
            if sep > 0:
                self.ranks[(m[:sep], m[sep + 1 :])] = i
        self.b2c = gpt2_byte_map()
        self.cache = {}

    def encode_piece(self, piece_bytes):
        s = "".join(self.b2c[b] for b in piece_bytes)
        hit = self.cache.get(s)
        if hit is not None:
            return hit
        if s in self.tok2id:
            res = (self.tok2id[s],)
            self.cache[s] = res
            return res
        word = list(s)
        ranks = self.ranks
        while len(word) > 1:
            best_r = 1 << 30
            best_pair = None
            for i in range(len(word) - 1):
                r = ranks.get((word[i], word[i + 1]))
                if r is not None and r < best_r:
                    best_r = r
                    best_pair = (word[i], word[i + 1])
            if best_pair is None:
                break
            a, b = best_pair
            ab = a + b
            new_word = []
            i = 0
            n = len(word)
            while i < n:
                if i + 1 < n and word[i] == a and word[i + 1] == b:
                    new_word.append(ab)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        res = tuple(self.tok2id[w] for w in word if w in self.tok2id)
        if len(self.cache) < 500000:
            self.cache[s] = res
        return res

    def count_text(self, text, counter):
        for m in PRE_RE.finditer(text):
            pb = m.group(0).encode("utf-8")
            for tid in self.encode_piece(pb):
                counter[tid] += 1

def gather_corpus():
    domains = {}

    # 1. Docs & prose (/usr/share/doc) up to 3.5 MB
    doc_files = sorted(
        p for p in glob.glob("/usr/share/doc/**/*", recursive=True)
        if os.path.isfile(p) and (
            p.endswith((".txt", ".md", ".rst"))
            or os.path.basename(p).startswith(("README", "NEWS", "FAQ", "HISTORY", "ChangeLog"))
        )
    )
    buf, sz = [], 0
    for p in doc_files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(60_000)
            if len(data) < 200:
                continue
            buf.append(data); sz += len(data)
            if sz >= 3_500_000: break
        except OSError:
            continue
    domains["docs_prose"] = "\n".join(buf)

    # 2. Man pages (/usr/share/man/man{1,2,3,5,7}/*.gz) up to 4.0 MB
    man_files = sorted(
        glob.glob("/usr/share/man/man1/*.gz")
        + glob.glob("/usr/share/man/man2/*.gz")
        + glob.glob("/usr/share/man/man3/*.gz")
        + glob.glob("/usr/share/man/man7/*.gz")
    )
    buf, sz = [], 0
    for p in man_files:
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                data = f.read(25_000)
            data = MAN_CLEAN_RE.sub("", data)
            if len(data) < 200: continue
            buf.append(data); sz += len(data)
            if sz >= 4_000_000: break
        except OSError:
            continue
    domains["man_pages"] = "\n".join(buf)

    # 3. Python 3.14 stdlib up to 3.0 MB
    py_files = sorted(glob.glob("/usr/lib/python3.14/**/*.py", recursive=True))
    buf, sz = [], 0
    for p in py_files:
        if "test" in p: continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(40_000)
            buf.append(data); sz += len(data)
            if sz >= 3_000_000: break
        except OSError:
            continue
    domains["python_stdlib"] = "\n".join(buf)

    # 4. C/C++ headers up to 1.5 MB
    h_files = sorted(glob.glob("/usr/include/*.h") + glob.glob("/usr/include/sys/*.h") + glob.glob("/usr/include/linux/*.h"))
    buf, sz = [], 0
    for p in h_files:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(25_000)
            buf.append(data); sz += len(data)
            if sz >= 1_500_000: break
        except OSError:
            continue
    domains["c_headers"] = "\n".join(buf)

    return domains

def is_ascii_bpe_tok(s):
    # GPT-2 byte-level BPE: 'Ġ' (0x120) is space, 'Ċ' (0x10A) is newline, ASCII printable is 0x21..0x7E
    for ch in s:
        o = ord(ch)
        if not ((0x21 <= o <= 0x7E) or o in (0x120, 0x10A, 0x109, 0x10D)):
            return False
    return True

def main():
    print("Loading GGUF BPE vocabulary...")
    tokens, merges = load_gguf_bpe(GGUF_PATH)
    bpe = FastBPE(tokens, merges)

    print("Gathering expanded independent multi-domain corpus (~12 MB)...")
    domains = gather_corpus()
    # Balanced weighting across domains (equal weight per domain so C headers don't drown English prose!)
    dom_counters = {}
    dom_stats = {}
    for dname, text in domains.items():
        dc = Counter()
        bpe.count_text(text, dc)
        dom_counters[dname] = dc
        dom_stats[dname] = {"chars": len(text), "tokens": sum(dc.values()), "unique": len(dc)}
        print(f"  {dname:15s}: {len(text):7d} chars -> {sum(dc.values()):7d} tokens ({len(dc):6d} unique)")

    # Weighted frequency: 40% man_pages + 30% docs_prose + 20% python_stdlib + 10% c_headers
    weights = {"man_pages": 0.40, "docs_prose": 0.30, "python_stdlib": 0.20, "c_headers": 0.10}
    scores = {}
    for dname, w in weights.items():
        dc = dom_counters[dname]
        tot = max(1, sum(dc.values()))
        for tid, c in dc.items():
            scores[tid] = scores.get(tid, 0.0) + w * (c / tot)

    # Also compute merge-rank order for unseen ASCII English tokens (lower merge rank = higher pretraining freq!)
    tok_merge_rank = [10_000_000] * VOCAB
    for i, m in enumerate(merges):
        sep = m.find(" ", 1)
        if sep > 0:
            merged = m[:sep] + m[sep + 1 :]
            tid = bpe.tok2id.get(merged)
            if tid is not None and tid < VOCAB and i < tok_merge_rank[tid]:
                tok_merge_rank[tid] = i

    must_keep = set(range(256)) | {EOS_ID}
    ascii_flag = [1 if is_ascii_bpe_tok(tokens[tid]) else 0 for tid in range(VOCAB)]
    # Sort key:
    #   tier 0: byte tokens 0..255 + EOS
    #   tier 1: observed in independent corpus (by balanced weighted frequency desc)
    #   tier 2: unseen ASCII/English BPE tokens (by GGUF BPE merge rank asc — earlier merge = higher training freq!)
    #   tier 3: remaining non-ASCII tokens (by GGUF BPE merge rank asc)
    order = list(range(VOCAB))
    def sort_key(tid):
        if tid in must_keep:
            return (0, 0.0, tid)
        sc = scores.get(tid, 0.0)
        if sc > 0.0:
            return (1, -sc, tok_merge_rank[tid])
        if ascii_flag[tid]:
            return (2, 0.0, tok_merge_rank[tid])
        return (3, 0.0, tok_merge_rank[tid])

    order.sort(key=sort_key)

    out_bin = "opt/kbench/drafter/vocab_rank_corpus.bin"
    with open(out_bin, "wb") as f:
        f.write(struct.pack(f"<{VOCAB}I", *order))

    out_meta = "opt/kbench/drafter/vocab_rank_meta.json"
    with open(out_meta, "w") as f:
        json.dump({
            "source": "Independent local corpus (Linux manpages + /usr/share/doc + Python 3.14 stdlib + /usr/include C headers), domain-balanced, with GGUF BPE merge-rank tie-breaker for unseen ASCII tokens. Zero overlap with evaluation prompts.",
            "domains": dom_stats,
            "total_corpus_tokens": sum(s["tokens"] for s in dom_stats.values()),
            "unique_corpus_tokens": len(scores),
            "ascii_tokens_total": sum(ascii_flag),
            "top_nonbyte_20": [(tid, tokens[tid], round(scores.get(tid, 0.0), 6)) for tid in order if tid >= 256][:20],
        }, f, indent=2)
    print(f"Total corpus tokens: {sum(s['tokens'] for s in dom_stats.values())}, unique observed: {len(scores)}, total ASCII BPE: {sum(ascii_flag)}")
    print("Top 15 non-byte:", [(tid, tokens[tid]) for tid in order if tid >= 256][:15])

if __name__ == "__main__":
    main()
