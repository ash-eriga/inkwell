#!/usr/bin/env python3
# Convert a post-4095-token ring snapshot to the full-length SWA oracle layout.
# Positions outside the live window are zeroed; the oracle masks them.
import struct, sys
from pathlib import Path

srcp, dstp = sys.argv[1:3]
max_ctx = 4352
swa = 1024
hd = 128
hkv = 8
esz = 2
row = hd * esz
full_len = hd * max_ctx * hkv * esz
ring_len = hd * swa * hkv * esz
conv_len = 79872

def visits():
    out = list(range(24))
    out += list(range(6, 18))
    out += list(range(6, 18))
    return out

def is_full(il): return il % 6 == 5
vs = visits()
assert len(vs) == 48

def read_state(path):
    b = Path(path).read_bytes()
    assert b[:8] == b'INKWST02'
    n = struct.unpack('<Q', b[8:16])[0]
    assert n == 144
    off = 16
    ts = []
    for _ in range(n):
        ln = struct.unpack('<Q', b[off:off+8])[0]; off += 8
        ts.append(b[off:off+ln]); off += ln
    assert off == len(b)
    return ts

ts = read_state(srcp)
out = []
# first 48 K, next 48 V, final 48 conv
for group in range(2):
    for slot, il in enumerate(vs):
        x = bytearray(full_len)
        src = ts[group * 48 + slot]
        if is_full(il):
            assert len(src) == full_len
            x[:] = src
        else:
            assert len(src) == ring_len
            # Snapshot was saved after positions 0..4094. Copy every ring row
            # still relevant to the next window into its linear oracle slot.
            for p in range(3072, 4095):
                jr = p % swa
                for h in range(hkv):
                    so = (h * swa + jr) * row
                    do = (h * max_ctx + p) * row
                    x[do:do+row] = src[so:so+row]
        out.append(bytes(x))
out.extend(ts[96:])
for x in out[96:]: assert len(x) == conv_len
with open(dstp, 'wb') as f:
    f.write(b'INKWST02')
    f.write(struct.pack('<Q', len(out)))
    for x in out:
        f.write(struct.pack('<Q', len(x)))
        f.write(x)
print('wrote', dstp, 'tensors', len(out), 'bytes', Path(dstp).stat().st_size)
