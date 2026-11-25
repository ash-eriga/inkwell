#!/usr/bin/env python3
import statistics
import sys
from pathlib import Path

for name in sys.argv[1:]:
    rows = []
    for line in Path(name).read_text().splitlines():
        if not line.startswith('SPEED ctx='):
            continue
        kv = dict(x.split('=', 1) for x in line.split()[1:] if '=' in x)
        rows.append(kv)
    pairs = sorted({int(x['tag'].split('-p')[1].split('-')[0]) for x in rows})
    ctl = [float(x['ms_tok']) for x in rows if x['arm'] == 'ctl']
    cand = [float(x['ms_tok']) for x in rows if x['arm'] == 'cand']
    wins = 0
    deltas = []
    bad = []
    for p in pairs:
        a = [x for x in rows if x['tag'].startswith(f's{rows[0]["ctx"]}-p{p}-') and x['arm'] == 'ctl']
        b = [x for x in rows if x['tag'].startswith(f's{rows[0]["ctx"]}-p{p}-') and x['arm'] == 'cand']
        if a and b:
            d = float(b[0]['ms_tok']) - float(a[0]['ms_tok'])
            deltas.append(d)
            wins += d < 0
        for x in a + b:
            if x['rc'] != '0' or float(x['peak']) > 5120:
                bad.append(x)
    print(f"{name}: pairs={len(pairs)} ctl_med={statistics.median(ctl):.3f} cand_med={statistics.median(cand):.3f} "
          f"delta_med={statistics.median(cand)-statistics.median(ctl):+.3f} pair_wins={wins}/{len(pairs)} "
          f"pair_deltas={','.join(f'{x:+.3f}' for x in deltas)} bad={len(bad)}")
