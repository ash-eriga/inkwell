#!/usr/bin/env python3
"""Compare a spec run's emitted tokens with a serial run's tokens.
usage: cmp_spec.py <spec.log> <serial.ids> [first_gen_in_spec_log]
serial.ids: whitespace list of `next=` ids (gen[0..]) from a serial run.
"""
import re, sys
spec_log, ser_path = sys.argv[1], sys.argv[2]
ser = [int(x) for x in open(ser_path).read().split()]
log = open(spec_log).read().split('\n')
vt, vr, rv = [], [], []
for i, l in enumerate(log):
    m = re.match(r'spec r(\d+): verify=([0-9.]+)ms accept=(\d+) emit=(\d+) pos=(\d+)', l)
    if m:
        vt.append([int(x) for x in re.findall(r'\d+', log[i+1])])
        vr.append([int(x) for x in re.findall(r'\d+', log[i+2])])
        rv.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))))
# gen index of the first anchor = number of `gen g: ... next=` lines in the spec log (serial warmup)
n0 = sum(1 for l in log if re.match(r'gen \d+: id=', l))
em, gi = [], n0 - 1          # anchor = gen[n0-1]
rows_ok = rows_bad = 0
bad = []
for r in range(len(vt)):
    n = rv[r][2]
    if vt[r][0] != em[-1] if em else False:
        print(f"r{r+1}: anchor mismatch vt0={vt[r][0]} last_emitted={em[-1]}")
    for j in range(8):
        prefix_ok = all(vt[r][k] == ser[gi + k] for k in range(j + 1))
        if prefix_ok and gi + j + 1 < len(ser):
            if vr[r][j] == ser[gi + j + 1]:
                rows_ok += 1
            else:
                rows_bad += 1
                bad.append((r + 1, j, vr[r][j], ser[gi + j + 1]))
    em += vr[r][:n + 1]
    gi += n + 1
mis = [i for i, (a, b) in enumerate(zip(em, ser[n0:])) if a != b]
print(f"spec tokens: {len(em)} rounds {len(vt)} | verify rows vs serial: ok={rows_ok} bad={rows_bad}")
for b in bad[:10]:
    print(f"   row mismatch round={b[0]} j={b[1]} spec_veri={b[2]} serial={b[3]}")
if mis:
    i = mis[0]
    print(f"first emitted mismatch at spec idx {i} (gen idx {n0+i})")
    print("  spec:", em[max(0,i-3):i+6]); print("  ser :", ser[n0+max(0,i-3):n0+i+10])
else:
    print(f"emitted tokens match serial for all {min(len(em), len(ser)-n0)} compared")
