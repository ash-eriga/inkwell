#!/usr/bin/env python3
import argparse
import math
import struct
import sys
from pathlib import Path

MAGIC = b"INKWHEAD"
VERSION = 1
DIM = 2304
VOCAB = 200058
SOFTCAP = 30.0
HEADER_LEN = 24
RECORD_LEN = 8 + DIM * 4 + 16 * 8
HEADER = struct.Struct("<IIIf")
XVEC = struct.Struct(f"<{DIM}f")
PAIR = struct.Struct("<If")


def fail(path: Path, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def check(path: Path) -> int:
    size = path.stat().st_size
    if size < HEADER_LEN:
        fail(path, "missing header")
    if (size - HEADER_LEN) % RECORD_LEN:
        fail(path, "truncated record")
    with path.open("rb") as f:
        header = f.read(HEADER_LEN)
        if header[:8] != MAGIC:
            fail(path, "bad magic")
        version, dim, vocab, softcap = HEADER.unpack_from(header, 8)
        if version != VERSION:
            fail(path, f"bad version {version}")
        if dim != DIM or vocab != VOCAB or softcap != SOFTCAP:
            fail(path, f"bad header constants dim={dim} vocab={vocab} softcap={softcap}")
        count = 0
        while rec := f.read(RECORD_LEN):
            step, token = struct.unpack_from("<II", rec, 0)
            del step
            x = XVEC.unpack_from(rec, 8)
            if not all(math.isfinite(v) for v in x):
                fail(path, f"record {count}: non-finite x")
            top = []
            for j in range(16):
                idx, value = PAIR.unpack_from(rec, 8 + DIM * 4 + j * 8)
                if idx >= VOCAB:
                    fail(path, f"record {count}: top-{j} index {idx} out of range")
                if not math.isfinite(value):
                    fail(path, f"record {count}: non-finite top-{j} logit")
                top.append((idx, value))
            if token >= VOCAB:
                fail(path, f"record {count}: chosen token {token} out of range")
            if top[0][0] != token:
                fail(path, f"record {count}: top-1 {top[0][0]} != chosen token {token}")
            if len({idx for idx, _ in top}) != 16:
                fail(path, f"record {count}: duplicate top-16 index")
            if any(top[j][1] < top[j + 1][1] for j in range(15)):
                fail(path, f"record {count}: top-16 logits are not descending")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="check an inkwell output-head dump")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        for path in args.paths:
            count = check(path)
            print(f"{path}: {count} records OK")
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
