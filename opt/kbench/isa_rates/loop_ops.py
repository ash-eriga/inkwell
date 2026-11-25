#!/usr/bin/env python3
import collections
import re
import sys


def opcode(op):
    return re.sub(r"^@!?P\d\s+", "", op).split()[0]


def main_loop(ins):
    best = []
    for addr, op in ins:
        if opcode(op).split(".")[0] != "BRA":
            continue
        target = re.search(r"0x([0-9a-f]+)", op)
        if not target or int(target.group(1), 16) >= addr:
            continue
        lo = int(target.group(1), 16)
        body = [opcode(o) for a, o in ins if lo <= a <= addr and opcode(o) != "NOP"]
        best = max(best, body, key=len)
    return best


def main():
    text = open(sys.argv[1]).read()
    for chunk in re.split(r"\n\s*Function : ", text)[1:]:
        name = chunk.split("\n")[0].strip()
        m = re.match(r"_Z\d+k_(\w+?)ILi8ELi(n?\d)E", name)
        if not m:
            continue
        ins = [(int(a, 16), o.strip()) for a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", chunk)]
        body = main_loop(ins)
        counts = collections.Counter(body).most_common(5)
        print(f"{m.group(1):<12} mode{m.group(2):<3} loop={len(body):<4} " + " ".join(f"{k}={v}" for k, v in counts))


if __name__ == "__main__":
    main()
