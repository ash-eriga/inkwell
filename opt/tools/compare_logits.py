#!/usr/bin/env python3
import struct
import sys


def rows(path):
    out = []
    with open(path, "rb") as f:
        while True:
            b = f.read(4)
            if not b:
                return out
            (n,) = struct.unpack("<I", b)
            v = struct.unpack(f"<{n}f", f.read(4 * n))
            out.append(v)


def main():
    a, b = rows(sys.argv[1]), rows(sys.argv[2])
    n = min(len(a), len(b))
    max_abs = 0.0
    agree = 0
    for i in range(n):
        xa, xb = a[i], b[i]
        max_abs = max(max_abs, max(abs(x - y) for x, y in zip(xa, xb)))
        agree += max(range(len(xa)), key=xa.__getitem__) == max(range(len(xb)), key=xb.__getitem__)
    print(f"rows={n} max_abs={max_abs:.6g} top1_agree={agree}/{n}")
    return 0 if len(a) == len(b) else 2


if __name__ == "__main__":
    sys.exit(main())
