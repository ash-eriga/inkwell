#!/usr/bin/env python3
import re
from pathlib import Path

root = Path(__file__).parent / 'artifacts'
gpu = root / 'gpu'

def read(p):
    try: return Path(p).read_text(errors='replace')
    except FileNotFoundError: return ''

def decode_ms(path, ngen):
    m = re.findall(r'decode 8 tok in ([0-9.]+)s', read(path))
    return float(m[-1]) * 1000 / ngen if m else None

def generated(path):
    m = re.search(r'generated text: "(.*)"\ndecode 8 tok', read(path), re.S)
    return m.group(1).encode().decode('unicode_escape') if m else ''

print('== oracle comparisons ==')
for n in (2, 4):
    print(f'-- {n}k --')
    print(read(gpu / f'tf{n}k_compare.txt').strip() or 'missing')

print('== ctx512 ==')
print(read(root / 'ctx512-valid/summary.log').strip() or 'missing')

print('== 4K free-run decode ==')
for name in ('free4k_master2', 'free4k_fixed'):
    ms = decode_ms(gpu / f'{name}.log', 64)
    text = generated(gpu / f'{name}.log')
    print(f'{name}: ms/tok={ms:.3f}' if ms else f'{name}: missing')
    print(text[:240].replace('\n', '\\n'))

print('== vtest ==')
vt = read(gpu / 'vtest8_wrap_fixed.log')
for line in vt.splitlines():
    if line.startswith('VTEST8 total') or 'MISMATCH' in line:
        print(line)

print('== status ==')
for p in sorted(gpu.glob('*.status')) + sorted((root / 'ctx512-valid').glob('status')):
    print(f'{p.name}: {p.read_text().strip()}')
