#!/usr/bin/env python3
"""Delete patched-CH2 NOPs from the six K=1.5..4 gemv loops in cuasm text.

The retired XMAD NOPs carry no scoreboard dependencies in the input.  Their
stall/yield bits are transferred to the next surviving instruction so removing
an instruction cannot reduce the scheduled spacing before a dependent consumer.
"""
import re
import sys

inp, outp = sys.argv[1], sys.argv[2]
TARGET = '.text._Z23inkwell_exl3_gemv_moe_s'
# Inclusive patched-cubin byte-address ranges from count_loops.py for K1.5..K4.
RANGES = [
    (0x15778, 0x16310, 'K1.5'), (0x17640, 0x181d8, 'K2'),
    (0x196b8, 0x1a250, 'K2.5'), (0x1b780, 0x1c318, 'K3'),
    (0x1d7f0, 0x1e388, 'K3.5'), (0x20c00, 0x217a8, 'K4'),
]
ins_re = re.compile(r'^(?P<prefix>\s*\[(?P<ctrl>[^\]]+)\])(?P<rest>.*?/\*(?P<addr>[0-9a-f]{4,})\*/\s*(?P<ins>.*)$)')
lines = open(inp).read().splitlines(keepends=True)
# Locate target text section end.
sec = None
for i, line in enumerate(lines):
    if line.startswith('// --------------------- ' + TARGET):
        sec = i
        break
if sec is None:
    raise SystemExit('target section not found')
end = len(lines)
for j in range(sec + 1, len(lines)):
    if lines[j].startswith('// --------------------- .text.') or lines[j].startswith('// --------------------- .nv.info.') or lines[j].startswith('// --------------------- .nv_constant'):
        end = j
        break

entries = []
for i in range(sec, end):
    m = ins_re.match(lines[i])
    if not m:
        continue
    addr = int(m.group('addr'), 16)
    body = m.group('ins').replace('{', ' ').replace('}', ' ')
    body = re.sub(r'/\*.*?\*/', ' ', body).strip()
    entries.append((i, addr, body))

def parse_ctrl(c):
    b, r, w, y, s = c.split(':')
    return [b[1:], r[1:], w[1:], y, int(s[1:])]

def fmt_ctrl(parts):
    return 'B' + parts[0] + ':R' + parts[1] + ':W' + parts[2] + ':' + parts[3] + ':S' + f'{parts[4]:02d}'

removed = []
carry = None
counts = {name: 0 for _, _, name in RANGES}
remove_lines = set()

def merge_wait(dst, src):
    if src == '------':
        return dst
    out = []
    for a, b in zip(dst, src):
        if a == '-':
            out.append(b)
        elif b == '-':
            out.append(a)
        elif a == b:
            out.append(a)
        else:
            raise SystemExit(f'cannot merge distinct wait barriers {dst} + {src}')
    return ''.join(out)

# carry = [stall, yield, waitbar]; barriers are OR-merged into the next survivor.
# Stall is distributed forward if one instruction cannot hold all of it.
for idx, (i, addr, body) in enumerate(entries):
    inside = next((name for lo, hi, name in RANGES if lo <= addr <= hi), None)
    is_nop = inside is not None and body == 'NOP ;'
    if is_nop:
        c = parse_ctrl(ins_re.match(lines[i]).group('ctrl'))
        if c[1] != '-' or c[2] != '-':
            raise SystemExit(f'NOP at {addr:x} has producer scoreboard control {c}; refusing')
        if carry is None:
            carry = [0, False, '------']
        carry[0] += c[4]
        carry[1] |= (c[3] == 'Y')
        carry[2] = merge_wait(carry[2], c[0])
        remove_lines.add(i)
        counts[inside] += 1
        continue
    if carry is not None and inside is not None:
        m = ins_re.match(lines[i])
        c = parse_ctrl(m.group('ctrl'))
        c[0] = merge_wait(c[0], carry[2])
        if carry[1]:
            c[3] = 'Y'
        room = 15 - c[4]
        take = min(room, carry[0])
        c[4] += take
        carry[0] -= take
        if carry[0] == 0:
            carry = None
        else:
            # Yield/wait were already transferred.  Keep only the remaining stall.
            carry[1] = False
            carry[2] = '------'
        old = m.group('ctrl')
        lines[i] = lines[i].replace(f'[{old}]', f'[{fmt_ctrl(c)}]', 1)

if carry is not None:
    raise SystemExit('trailing NOP carry at end of loop; not implemented')
if sorted(counts.values()) != [32] * 6:
    raise SystemExit(f'unexpected NOP counts {counts}')
out = [line for i, line in enumerate(lines) if i not in remove_lines]
open(outp, 'w').write(''.join(out))
print('removed', counts, 'total', sum(counts.values()), 'lines', len(remove_lines))
