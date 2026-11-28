#!/usr/bin/env python3
"""Convert an INKWST02 K8V4 snapshot from token-major rows to the Front-4E
16-row paired head-major layout, preserving every non-quantized byte exactly."""
import os
import struct
import sys
import numpy as np

MAGIC_IN = b"INKWST02"
MAGIC_OUT = b"INKWST03"
ROWBYTES = 1632


def convert_qpass(buf: memoryview) -> bytes:
    n = len(buf)
    cap, rem = divmod(n - 512, ROWBYTES)
    if rem or cap % 16:
        raise ValueError(f"not a supported K8V4 pass: len={n} cap={cap} rem={rem}")
    out = bytearray(buf)
    nt = cap // 16

    kb = cap * 1024
    vb = cap * 512
    k = np.frombuffer(buf, dtype=np.uint8, count=kb, offset=0).reshape(nt, 16, 8, 16, 8)
    ko = np.empty((8, nt, 8, 16, 2, 8), dtype=np.uint8)
    for half in (0, 1):
        ko[:, :, :, :, half, :] = k[:, 8 * half:8 * half + 8].transpose(2, 0, 1, 3, 4)
    out[:kb] = ko.tobytes()

    v = np.frombuffer(buf, dtype=np.uint8, count=vb, offset=kb).reshape(nt, 16, 8, 16, 4)
    vo = np.empty((8, nt, 8, 16, 2, 4), dtype=np.uint8)
    for half in (0, 1):
        vo[:, :, :, :, half, :] = v[:, 8 * half:8 * half + 8].transpose(2, 0, 1, 3, 4)
    out[kb:kb + vb] = vo.tobytes()

    mb = cap * 96
    mo = kb + vb
    m = np.frombuffer(buf, dtype=np.uint8, count=mb, offset=mo).reshape(nt, 16, 8, 12)
    xo = np.empty((8, nt, 8, 2, 12), dtype=np.uint8)
    for half in (0, 1):
        xo[:, :, :, half, :] = m[:, 8 * half:8 * half + 8].transpose(2, 0, 1, 3)
    out[mo:mo + mb] = xo.tobytes()
    return bytes(out)


def main(src: str, dst: str) -> None:
    with open(src, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC_IN:
        raise SystemExit(f"bad state magic: expected {MAGIC_IN!r}, got {data[:8]!r}")
    count = struct.unpack_from("<Q", data, 8)[0]
    pos = 16
    pieces = [MAGIC_OUT, data[8:16]]
    converted = 0
    for i in range(count):
        n = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        payload = memoryview(data)[pos:pos + n]
        pos += n
        cap, rem = divmod(n - 512, ROWBYTES)
        if n >= 512 and rem == 0 and cap > 0 and cap % 16 == 0:
            payload = memoryview(convert_qpass(payload))
            converted += 1
        pieces.append(struct.pack("<Q", len(payload)))
        pieces.append(payload)
    if pos != len(data):
        raise SystemExit("trailing bytes in state")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        for x in pieces:
            f.write(x)
    os.replace(tmp, dst)
    print(f"converted {converted} K8V4 passes: {src} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: transpose_state.py SRC STATE_DST")
    main(sys.argv[1], sys.argv[2])
