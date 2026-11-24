#!/usr/bin/env python3
import collections
import re
import sys

OPS = ("XMAD", "IDP", "I2F", "FFMA", "SHR", "BFE", "SHF", "LOP32I", "LDG", "LDS")


def functions(text):
    for chunk in re.split(r"\n\s*Function : ", text)[1:]:
        name = chunk.split("\n")[0].strip()
        ins = []
        for line in chunk.split("\n"):
            m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
            if m:
                ins.append((int(m.group(1), 16), m.group(2).strip()))
        yield name, ins


def opcode(op):
    return re.sub(r"^@!?P\d\s+", "", op).split()[0].split(".")[0]


def loops(ins, min_len):
    for addr, op in ins:
        if opcode(op) != "BRA":
            continue
        t = re.search(r"0x([0-9a-f]+)", op)
        if not t or int(t.group(1), 16) >= addr:
            continue
        lo = int(t.group(1), 16)
        body = [opcode(o) for a, o in ins if lo <= a <= addr and opcode(o) != "NOP"]
        if len(body) >= min_len:
            yield lo, addr, collections.Counter(body)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: sass_loops.py <cuobjdump -sass output> <function substring>...")
    text = open(sys.argv[1]).read()
    for name, ins in functions(text):
        if not any(w in name for w in sys.argv[2:]):
            continue
        print(name)
        for lo, hi, c in loops(ins, 100):
            n = sum(c.values())
            iadd = c["IADD"] + c["IADD3"]
            cols = " ".join(f"{k}={c[k]}" for k in OPS)
            print(f"  {lo:#x}..{hi:#x} instr={n} {cols} IADD={iadd} instr/w={n / max(c['IDP'], 1):.2f}")


if __name__ == "__main__":
    main()
