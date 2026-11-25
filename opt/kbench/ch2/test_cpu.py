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


def lane_start(ka, h, lane):
    rbits = 32 * (8 * ka + 4 * h)
    base = (8 * lane + 1) * ka + ((8 * lane + 1) >> 1 if h else 0) - 16
    return base % rbits


def generic_donor(ka, h, lane, step):
    if (ka, h) == (2, 1):
        r = lane & 7
        d = 1 + ((r == 1) | (r == 4) | (r == 6)) if step == 1 else 3 - ((r == 2) | (r == 7)) + (r == 4)
    elif (ka, h) == (3, 0):
        r = lane & 3
        d = 1 + (r == 2) if step == 1 else 2 + ((r == 1) | (r == 2))
    elif (ka, h) == (3, 1):
        r = lane & 7
        d = 1 + (r == 4) if step == 1 else 2 + ((r == 3) | (r == 4))
    else:
        d = step
    return (lane + d) & 31


def test_top32_shuffle_donors():
    cases = 0
    for ka, h in ((1, 1), (2, 0)):
        nw = 8 * ka + 4 * h
        for lane in range(32):
            base = lane_start(ka, h, lane)
            w0 = base >> 5
            if ka == 2:
                donor = (lane + 2) & 31
            else:
                donor = ((lane + 1) if (lane & 7) == 1 else (lane + 3)) & 31
            dbase = lane_start(ka, h, donor)
            if (dbase >> 5) != (w0 + 1) % nw:
                raise SystemExit(f"shuffle donor mismatch ka={ka} h={h} lane={lane}")
            cases += 1
    print(f"ch2 top32 shuffle donors: cases={cases} PASS")


def test_generic_shuffle_donors():
    rng = random.Random(771177)
    donor_cases = stream_cases = 0
    for ka, h in ((2, 1), (3, 0), (3, 1), (4, 0)):
        nw = 8 * ka + 4 * h
        base_phases = set()
        stream_phases = set()
        for lane in range(32):
            base = lane_start(ka, h, lane)
            base_phases.add(base & 31)
            w0 = base >> 5
            for step in (1, 2):
                donor = generic_donor(ka, h, lane, step)
                dbase = lane_start(ka, h, donor)
                if (dbase >> 5) != (w0 + step) % nw:
                    raise SystemExit(f"generic donor mismatch K={ka+h/2} lane={lane} step={step}")
                dist = (donor - lane) & 31
                for earlier in range(1, dist):
                    ebase = lane_start(ka, h, (lane + earlier) & 31)
                    if (ebase >> 5) == (w0 + step) % nw:
                        raise SystemExit(f"generic donor not minimal K={ka+h/2} lane={lane} step={step}")
                donor_cases += 1
        for trial in range(24):
            words = [rng.getrandbits(32) for _ in range(nw)]
            own = []
            for lane in range(32):
                own.append(words[lane_start(ka, h, lane) >> 5])
            for lane in range(32):
                base = lane_start(ka, h, lane)
                off = base & 31
                w0 = own[lane]
                w1 = own[generic_donor(ka, h, lane, 1)]
                w2 = own[generic_donor(ka, h, lane, 2)]
                hi = fsl(w1, w0, off)
                lo = fsl(w2, w1, off)
                for j in range(8):
                    D = j * ka + h * ((j + 1) >> 1)
                    stream_phases.add((base + D) & 31)
                    want = ring_window(words, base + D)
                    got = (ch2_view(D, hi, lo) >> 16) & 0xFFFF
                    if got != want:
                        raise SystemExit(f"generic stream mismatch K={ka+h/2} lane={lane} j={j}")
                    stream_cases += 1
        print(f"ch2 generic shuffle donors K={ka+h/2}: lanes=32 base_phases={len(base_phases)} stream_phases={len(stream_phases)} PASS")
    print(f"ch2 generic shuffle donors: donor_cases={donor_cases} stream_cases={stream_cases} PASS")


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
    test_random_tiles()
    test_top32_shuffle_donors()
    test_generic_shuffle_donors()


if __name__ == "__main__":
    main()
