#!/usr/bin/env python3
"""Rewrite the compiler's 3-XMAD CH2 product into the exact 2-XMAD form.

ptxas emits, for each high-half window U:
  LP    XMAD t, U.H1, 0xd12d, RZ
  LOW   XMAD p, U,    0x83dc, t     (low-half term plus t)
  HIGH  XMAD.PSL p, U.H1, 0x83dc, p

The exact core wants LOW to be the already-computed LP instead.  This script
rewrites LOW in place to `XMAD p, U.H1, 0xd12d, RZ` and retires LP to NOP:
  LOW'  XMAD p, U.H1, 0xd12d, RZ
  HIGH  XMAD.PSL p, U.H1, 0x83dc, p
which is exactly 0x83DCD12D*(U.H1) mod 2^32.  Register/immediate field layouts
follow the sm_61 XMAD encoding documented by maxas: r0, r8, immediate at bit
20, H1 at bit 53, and r39/RZ=0xff.
"""
import re
import subprocess
import sys
from pathlib import Path

NOP_WORD = 0x50B0000000070F00
KERNELS = ("gemv_moe_s", "gemv_sf_ch2", "gemv_batchf")
FAKE_RE = re.compile(r"XMAD\s+R\d+,\s+(R\d+)(?:\.reuse)?,\s+0x83dc,\s+(R\d+)\s*;.*?/\* (0x[0-9a-f]+) \*/")
LP_RE = re.compile(r"XMAD\s+(R\d+),\s+(R\d+)\.H1(?:\.reuse)?,\s+0xd12d,\s+RZ\s*;.*?/\* (0x[0-9a-f]+) \*/")


def sass_text(cuobjdump: str, obj: Path) -> str:
    return subprocess.check_output([str(cuobjdump), "-sass", str(obj)], text=True, stderr=subprocess.DEVNULL)


def target_chunk(text: str) -> str:
    chunks = re.split(r"\n\s*Function : ", text)[1:]
    out = []
    for kernel in KERNELS:
        target = [c for c in chunks if kernel in c.split("\n", 1)[0]]
        if len(target) > 1:
            raise SystemExit(f"expected at most one {kernel} function, found {len(target)}")
        out.extend(target)
    if not out:
        raise SystemExit("no CH2 target functions found")
    return "\n".join(out)


def collect(text: str):
    lines = target_chunk(text).splitlines()
    fakes = []
    used = set()
    for line in lines:
        if "XMAD.PSL" in line:
            continue
        m = FAKE_RE.search(line)
        if m:
            src, addend, hx = m.groups()
            fakes.append((src, addend, hx))
            used.add((addend, src))
    lps = []
    for line in lines:
        if "XMAD.PSL" in line:
            continue
        m = LP_RE.search(line)
        if m and (m.group(1), m.group(2)) in used:
            lps.append(m.group(3))
    return fakes, lps


LOW_MUL_RE = re.compile(r"XMAD\s+(R\d+),\s+(R\d+)(?:\.reuse)?,\s+0x83dc,\s+RZ\s*;.*?/\* (0x[0-9a-f]+) \*/")
LOW_PSL_RE = re.compile(r"XMAD\.PSL\.CLO\s+(R\d+),\s+(R\d+)(?:\.reuse)?,\s+0x1,\s+RZ\s*;.*?/\* (0x[0-9a-f]+) \*/")
INS_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);\s+/\* (0x[0-9a-f]+) \*/")


def collect_low_pairs(text: str, data: bytes):
    """Locate ch2_mul16_lo's mul+shift expansion by exact ELF code offsets."""
    try:
        from elftools.elf.elffile import ELFFile
        from io import BytesIO
        with BytesIO(data) as f:
            elf = ELFFile(f)
            sections = [sec for sec in elf.iter_sections() if sec.name.startswith('.text.')]
    except Exception as e:
        raise SystemExit(f'pyelftools unavailable for low-half CH2 patch: {e}')
    pairs = []
    chunks = re.split(r"\n\s*Function : ", text)[1:]
    for kernel in KERNELS:
        hits = [c for c in chunks if kernel in c.split('\n', 1)[0]]
        if len(hits) != 1:
            continue
        sec_hits = [sec for sec in sections if kernel in sec.name]
        if len(sec_hits) != 1:
            raise SystemExit(f'expected one .text section for {kernel}, found {len(sec_hits)}')
        base = sec_hits[0]['sh_offset']
        muls = []
        for line in hits[0].splitlines():
            mi = INS_RE.search(line)
            if not mi:
                continue
            addr, ins, hx = mi.groups()
            pos = base + int(addr, 16)
            if "XMAD.PSL" in ins:
                m = LOW_PSL_RE.search(line)
                if m and ".H1" not in ins:
                    _dst, src, _hx = m.groups()
                    hit = None
                    for j in range(len(muls) - 1, -1, -1):
                        if muls[j][0] == src:
                            hit = muls[j]
                            break
                    if hit:
                        pairs.append((hit[3], pos, hit[1], hit[2], hx))
                continue
            m = LOW_MUL_RE.search(line)
            if m:
                dst, src, _hx = m.groups()
                muls.append((dst, src, hx, pos))
    for mp, pp, _src, mh, ph in pairs:
        if data[mp:mp + 8] != bytes.fromhex(mh[2:])[::-1]:
            raise SystemExit(f'low-half mul byte mismatch at {mp:x}')
        if data[pp:pp + 8] != bytes.fromhex(ph[2:])[::-1]:
            raise SystemExit(f'low-half PSL byte mismatch at {pp:x}')
    return pairs


def rewrite_low_psl(hx: str, src_reg: str) -> bytes:
    w = int(hx, 16)
    w &= ~(0xff << 8)
    w |= int(src_reg[1:]) << 8
    w &= ~(0xffff << 20)
    w |= 0x83dc << 20
    w &= ~(1 << 50)  # clear CLO: direct XMAD.PSL form
    return w.to_bytes(8, "little")


def positions(data: bytes | bytearray, hx: str):
    needle = bytes.fromhex(hx[2:])[::-1]
    out = []
    start = 0
    while True:
        p = data.find(needle, start)
        if p < 0:
            return out
        out.append(p)
        start = p + 1


def rewrite_fake(hx: str) -> bytes:
    w = int(hx, 16)
    w &= ~(0xffff << 20)
    w |= 0xD12D << 20
    w |= 0x0020000000000000  # source operand .H1
    w &= ~(0xFF << 39)
    w |= 0xFF << 39          # addend RZ
    return w.to_bytes(8, "little")


def patch(obj: Path, cuobjdump: str) -> None:
    data = bytearray(obj.read_bytes())
    fakes, lps = collect(sass_text(cuobjdump, obj))
    if len(fakes) < 100 or len(lps) < 100:
        raise SystemExit(f"too few CH2 sites: fake={len(fakes)} lp={len(lps)}")

    replacements = []
    for _, _, hx in fakes:
        ps = positions(data, hx)
        if not ps:
            raise SystemExit(f"missing fake XMAD encoding {hx}")
        repl = rewrite_fake(hx)
        replacements.extend((p, repl) for p in ps)
    nop = NOP_WORD.to_bytes(8, "little")
    for hx in set(lps):
        ps = positions(data, hx)
        if not ps:
            raise SystemExit(f"missing low-product encoding {hx}")
        replacements.extend((p, nop) for p in ps)

    low_pairs = collect_low_pairs(sass_text(cuobjdump, obj), bytes(data))
    for mul_pos, psl_pos, src, _mh, ph in low_pairs:
        replacements.append((mul_pos, nop))
        replacements.append((psl_pos, rewrite_low_psl(ph, src)))

    for p, repl in replacements:
        data[p:p + 8] = repl
    obj.write_bytes(data)

    # Audit the final machine code: no low-half 0x83dc XMAD may remain, and the
    # rewritten products must be visible as high-half 0xd12d XMADs.
    after = target_chunk(sass_text(cuobjdump, obj))
    rem = sum(1 for l in after.splitlines() if "XMAD.PSL" not in l and "0x83dc" in l and "XMAD" in l)
    hi_d = sum(1 for l in after.splitlines() if re.search(r"XMAD\s+R\d+,\s+R\d+\.H1(?:\.reuse)?,\s+0xd12d,\s+RZ", l))
    if rem:
        raise SystemExit(f"{rem} low-half 0x83dc XMADs remain")
    print(f"ch2-sass-patch: fake={len(fakes)} lp={len(lps)} words={len(replacements)} hi_d12d={hi_d} low_pairs={len(low_pairs)}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_sass.py <kernels.o> <cuobjdump>")
    patch(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
