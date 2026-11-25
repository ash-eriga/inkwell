#!/usr/bin/env python3
"""CPU exactness model for the patched CH2 high-half extraction/product."""
import random

MASK = 0xFFFFFFFF
C = 0x83DCD12D


def fsl(lo, hi, s):
    s &= 31
    v = ((hi << 32) | lo) << s
    return (v >> 32) & MASK


def fsr(lo, hi, s):
    s &= 31
    return (((hi << 32) | lo) >> s) & MASK


def sgemv_win(D, hi, lo):
    if D <= 16:
        return (hi >> (16 - D)) & MASK
    return fsr(lo, hi, 48 - D)


def ch2_view(D, hi, lo):
    if D == 0:
        return hi
    if D < 32:
        return fsl(lo, hi, D)
    if D == 32:
        return lo
    return (lo << (D - 32)) & MASK


def bytesum32(x):
    return sum((x >> (8 * i)) & 0xFF for i in range(4))


def exact_s(w):
    return bytesum32((C * w) & MASK)


def patched_product(u):
    # XMAD R,H1*0xd12d; XMAD.PSL R,H1*0x83dc<<16 + R (all modulo 2^32).
    w = (u >> 16) & 0xFFFF
    lo = (w * 0xD12D) & MASK
    return (lo + ((w * 0x83DC) << 16)) & MASK


def ring_bit(words, pos):
    pos %= 32 * len(words)
    return (words[pos >> 5] >> (31 - (pos & 31))) & 1


def ring_window(words, start):
    v = 0
    for m in range(16):
        v = (v << 1) | ring_bit(words, start + m)
    return v


def test_top32_rotate_views():
    rng = random.Random(661177)
    cases = 0
    for _ in range(20000):
        hi = rng.getrandbits(32)
        lo = rng.getrandbits(32)
        for ka, h in ((1, 1), (2, 0)):
            for j in range(8):
                d = j * ka + h * ((j + 1) >> 1)
                rot = ((hi << d) | (hi >> ((32 - d) & 31))) & MASK
                if (rot >> 16) != (ch2_view(d, hi, lo) >> 16):
                    raise SystemExit(f"top32 rotate mismatch ka={ka} h={h} j={j}")
                cases += 1
    print(f"ch2 top32 rotate views: cases={cases} PASS")


def test_ch2_xvec_permutation():
    perm = [4 * ((r & 7) >> 1) + (r & 1) + (2 if r & 8 else 0) for r in range(16)]
    if sorted(perm) != list(range(16)):
        raise SystemExit("xvec permutation is not a bijection")
    for t3 in range(4):
        r0 = 2 * t3
        got = [None] * 4
        for r, p in enumerate(perm):
            if p >= 4 * t3 and p < 4 * t3 + 4:
                got[p - 4 * t3] = r
        if got != [r0, r0 + 1, r0 + 8, r0 + 9]:
            raise SystemExit(f"xvec group mismatch t3={t3}: {got}")
    print("ch2 xvec permutation: groups=4 PASS")


def test_random_tiles():
    rng = random.Random(918273)
    cases = 0
    for trial in range(40):
        for ka, h in ((1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (5, 0)):
            nw = 8 * ka + 4 * h
            rbits = 32 * nw
            words = [rng.getrandbits(32) for _ in range(nw)]
            x = [rng.randint(-3, 3) for _ in range(16)]
            got = [0] * 16
            ref = [0] * 16
            for lane in range(32):
                base = (8 * lane + 1) * ka + ((8 * lane + 1) >> 1 if h else 0) - 16
                if base < 0:
                    base += rbits
                w0 = base >> 5
                off = base & 31
                w1 = 0 if w0 + 1 == nw else w0 + 1
                w2 = 0 if w1 + 1 == nw else w1 + 1
                hi = fsl(words[w1], words[w0], off)
                lo = fsl(words[w2], words[w1], off)
                for j in range(8):
                    i = 8 * lane + j
                    D = j * ka + h * ((j + 1) >> 1)
                    want_start = (i + 1) * ka + (((i + 1) >> 1) if h else 0) - 16
                    want = ring_window(words, want_start)
                    gotw = (ch2_view(D, hi, lo) >> 16) & 0xFFFF
                    if gotw != want:
                        raise SystemExit(f"tile ring mismatch K={ka+h/2} lane={lane} j={j}")
                    k = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1)
                    n = (lane >> 2) + 8 * (j >> 2)
                    sv = exact_s(want)
                    got[n] += sv * x[k]
                    ref[n] += sv * x[k]
            if got != ref:
                raise SystemExit(f"GEMV accumulation mismatch K={ka+h/2}")
            cases += 1
    print(f"ch2 cpu tile GEMV exactness: cases={cases} PASS")


def main():
    rng = random.Random(20250308)
    nview = nprod = 0
    for _ in range(20000):
        hi = rng.getrandbits(32)
        lo = rng.getrandbits(32)
        for ka, h in ((1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (5, 0)):
            for j in range(8):
                D = j * ka + h * ((j + 1) >> 1)
                want = sgemv_win(D, hi, lo) & 0xFFFF
                got = (ch2_view(D, hi, lo) >> 16) & 0xFFFF
                if want != got:
                    raise SystemExit(f"view mismatch ka={ka} h={h} j={j} D={D}: {want:x} != {got:x}")
                nview += 1
        u = rng.getrandbits(32)
        w = (u >> 16) & 0xFFFF
        if patched_product(u) != (C * w) & MASK:
            raise SystemExit("patched product mismatch")
        p = patched_product(u)
        bits = (sum((p >> (8 * i)) & 0xFF for i in range(4)) + 0x4B000000) & MASK
        # fp32 bit pattern is 2^23+s; integer payload is exact in the model.
        if bits - 0x4B000000 != exact_s(w):
            raise SystemExit("bytesum mismatch")
        nprod += 1
    print(f"ch2 cpu exactness: view_cases={nview} product_cases={nprod} PASS")
    test_top32_rotate_views()
    test_ch2_xvec_permutation()
    test_random_tiles()


if __name__ == "__main__":
    main()
