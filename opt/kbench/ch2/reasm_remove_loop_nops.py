#!/usr/bin/env python3
"""Delete patched-CH2 NOPs from the six K=1.5..4 gemv loops in cuasm text.

The retired XMAD NOPs carry no scoreboard dependencies in the input.  On
Pascal, stall/yield bits belong to the instruction that has just issued and
control the gap before the following instruction.  They are therefore
transferred backward to the previous surviving instruction.  Wait masks gate
the instruction carrying them, so they are transferred forward to the next
surviving instruction.
"""
import json
import re
import sys

inp, outp = sys.argv[1], sys.argv[2]
TARGET = '.text._Z23inkwell_exl3_gemv_moe_s'
if len(sys.argv) > 5:
    TARGET = sys.argv[5]
# Inclusive patched-cubin byte-address ranges from count_loops.py for K1.5..K4.
RANGES = [
    (0x15778, 0x16310, 'K1.5'), (0x17640, 0x181d8, 'K2'),
    (0x196b8, 0x1a250, 'K2.5'), (0x1b780, 0x1c318, 'K3'),
    (0x1d7f0, 0x1e388, 'K3.5'), (0x20c00, 0x217a8, 'K4'),
]
if len(sys.argv) > 3:
    RANGES = [(r['start'], r['end'], r['k']) for r in json.load(open(sys.argv[3]))]
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

def parse_ctrl(c):
    b, r, w, y, s = c.split(':')
    return [b[1:], r[1:], w[1:], y, int(s[1:])]

def fmt_ctrl(parts):
    return 'B' + parts[0] + ':R' + parts[1] + ':W' + parts[2] + ':' + parts[3] + ':S' + f'{parts[4]:02d}'

def read_entries():
    out = []
    for i in range(sec, end):
        m = ins_re.match(lines[i])
        if not m:
            continue
        addr = int(m.group('addr'), 16)
        body = m.group('ins').replace('{', ' ').replace('}', ' ')
        body = re.sub(r'/\*.*?\*/', ' ', body).strip()
        out.append((i, addr, body))
    return out

if len(sys.argv) > 4 and sys.argv[4] not in ('mov', '-'):
    raise SystemExit(f'unknown option {sys.argv[4]}')
mov_mode = len(sys.argv) > 4 and sys.argv[4] == 'mov'
entries = read_entries()
drop_ctrl = set()
if mov_mode:
    mov_re = re.compile(r'MOV R(\d+), R(\d+) ;')
    ldg_re = re.compile(r'LDG(?:\.[A-Z0-9]+)* R(\d+), \[R(\d+)\] ;')
    converted = {name: 0 for _, _, name in RANGES}
    for lo, hi, name in RANGES:
        inner = [(idx, i, addr, body) for idx, (i, addr, body) in enumerate(entries) if lo <= addr <= hi]
        if len(inner) < 3:
            raise SystemExit(f'{name}: too few loop instructions for MOV removal')
        (_, i0, a0, b0), (_, i1, a1, b1), (_, i2, a2, b2) = inner[:3]
        m0 = mov_re.fullmatch(b0)
        m1 = mov_re.fullmatch(b1)
        ml = ldg_re.fullmatch(b2)
        if not m0 or not m1 or not ml:
            raise SystemExit(f'{name}: loop entry is not MOV/MOV/LDG: {b0!r} {b1!r} {b2!r}')
        d0, s0 = int(m0.group(1)), int(m0.group(2))
        d1, s1 = int(m1.group(1)), int(m1.group(2))
        if d1 != d0 + 1 or s1 != s0 + 1 or d0 & 1 or s0 & 1 or int(ml.group(2)) != d0:
            raise SystemExit(f'{name}: unexpected loop-entry address registers')
        for i in (i0, i1):
            c = parse_ctrl(ins_re.match(lines[i]).group('ctrl'))
            if c[0] != '------' or c[1] != '-' or c[2] != '-' or c[3] == 'Y':
                raise SystemExit(f'{name}: loop-entry MOV has non-stall control {c}')
            lines[i] = re.sub(r'MOV\s+R\d+,\s+R\d+\s*;', 'NOP ;', lines[i], count=1)
            drop_ctrl.add(i)
        lines[i2] = lines[i2].replace(f'[R{d0}]', f'[R{s0}]', 1)
        converted[name] = 2
    if sorted(converted.values()) != [2] * 6:
        raise SystemExit(f'unexpected MOV conversion counts {converted}')
    entries = read_entries()

counts = {name: 0 for _, _, name in RANGES}
remove_lines = set()
nop_ctrls = {}

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

def set_ctrl(i, c):
    m = ins_re.match(lines[i])
    old = m.group('ctrl')
    lines[i] = lines[i].replace(f'[{old}]', f'[{fmt_ctrl(c)}]', 1)

# Mark every deletion before transferring controls.  Besides making consecutive
# NOP runs explicit, this prevents a wait mask from being moved onto another NOP
# that is itself about to be deleted.
for idx, (i, addr, body) in enumerate(entries):
    inside = next((name for lo, hi, name in RANGES if lo <= addr <= hi), None)
    if inside is None or body != 'NOP ;':
        continue
    c = parse_ctrl(ins_re.match(lines[i]).group('ctrl'))
    if c[1] != '-' or c[2] != '-':
        raise SystemExit(f'NOP at {addr:x} has producer scoreboard control {c}; refusing')
    remove_lines.add(i)
    nop_ctrls[i] = c
    counts[inside] += 1

# Wait masks gate issue of the instruction carrying them.  Move them onto the
# next surviving instruction, ORing all masks from a consecutive NOP run.
forward_waits = {}
for idx, (i, addr, body) in enumerate(entries):
    if i not in remove_lines:
        continue
    c = nop_ctrls[i]
    if c[0] == '------':
        continue
    j = idx + 1
    while j < len(entries) and entries[j][0] in remove_lines:
        j += 1
    if j >= len(entries):
        raise SystemExit('trailing NOP wait at end of section; not implemented')
    target = entries[j][0]
    forward_waits[target] = merge_wait(forward_waits.get(target, '------'), c[0])

for i, wait in forward_waits.items():
    c = parse_ctrl(ins_re.match(lines[i]).group('ctrl'))
    c[0] = merge_wait(c[0], wait)
    set_ctrl(i, c)

# Stall and yield have the opposite directionality: they belong to the
# instruction just issued and control the gap before the next instruction.
# Add each NOP run's stalls to the previous surviving instruction.  Do not add
# the deleted issue slot itself; that would give up the intended speedup.  This
# preserves all explicit stall cycles before the next survivor, while still
# reducing producer/consumer spacing by the number of retired NOP issue slots.
saturated = []
idx = 0
while idx < len(entries):
    i, addr, body = entries[idx]
    if i in remove_lines:
        idx += 1
        continue
    j = idx + 1
    stalls = 0
    yield_seen = False
    while j < len(entries) and entries[j][0] in remove_lines:
        nc = nop_ctrls[entries[j][0]]
        if entries[j][0] not in drop_ctrl:
            stalls += nc[4]
            yield_seen |= (nc[3] == 'Y')
        j += 1
    if j != idx + 1:
        c = parse_ctrl(ins_re.match(lines[i]).group('ctrl'))
        wanted = c[4] + stalls
        c[4] = min(15, wanted)
        if c[4] != wanted:
            saturated.append((addr, wanted))
        if yield_seen:
            c[3] = 'Y'
        set_ctrl(i, c)
    idx = j

# Every deleted NOP must have been covered by the survivor scan above.  A NOP
# at the start of the section has no previous instruction that can carry its
# stall, so reject that layout rather than silently dropping control state.
covered = set()
idx = 0
while idx < len(entries):
    if entries[idx][0] in remove_lines:
        idx += 1
        continue
    j = idx + 1
    while j < len(entries) and entries[j][0] in remove_lines:
        covered.add(entries[j][0])
        j += 1
    idx = j
if covered != remove_lines:
    raise SystemExit('leading deleted NOP has no previous surviving instruction')
expected_nops = 34 if mov_mode else 32
if sorted(counts.values()) != [expected_nops] * len(RANGES):
    raise SystemExit(f'unexpected NOP counts {counts}')
out = [line for i, line in enumerate(lines) if i not in remove_lines]
open(outp, 'w').write(''.join(out))
print('removed', counts, 'total', sum(counts.values()), 'lines', len(remove_lines),
      'forward_waits', len(forward_waits), 'saturated', saturated, 'drop_ctrl', len(drop_ctrl))
