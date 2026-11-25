#!/usr/bin/env python3
"""Fail-closed build-time generator for the CH2 NOP-deleted sm_61 cubin.

Usage: build_nopless.py <kernels.o> <output.cubin> <workdir> [cuda-home]
The output is written only after every structural check passes.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from io import BytesIO
from pathlib import Path

class Ch2Fail(Exception):
    pass

def env_first(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None

def run(cmd, cwd, log, env, capture=False):
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE if capture else lf,
                           stderr=subprocess.STDOUT, text=True)
    if capture:
        Path(log).write_text(r.stdout or '')
    if r.returncode != 0:
        raise Ch2Fail(f"command failed rc={r.returncode}: {' '.join(map(str, cmd))} (log {log})")
    return r.stdout or ''

def target_chunk(sass):
    chunks = re.split(r"\n\s*Function : ", sass)[1:]
    hits = [c for c in chunks if 'gemv_moe_s' in c.split('\n', 1)[0]]
    if len(hits) != 1:
        raise Ch2Fail(f'expected one gemv_moe_s SASS function, found {len(hits)}')
    return hits[0]

def loop_rows(sass):
    chunk = target_chunk(sass)
    ins = []
    for a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", chunk):
        op = re.sub(r"^@!?P\d\s+", "", o.strip()).split()[0].split(".")[0]
        ins.append((int(a, 16), op, o.strip()))
    rows = []
    for addr, op, full in ins:
        if op.split(".")[0] != 'BRA':
            continue
        m = re.search(r"0x([0-9a-f]+)", full)
        if not m:
            continue
        target = int(m.group(1), 16)
        if target >= addr:
            continue
        body = [(a, o, f) for a, o, f in ins if target <= a <= addr]
        psl = sum(1 for _, o, full in body if full.startswith('XMAD.PSL') and '0x83dc' in full)
        if psl not in (32, 64):
            continue
        cnt = Counter(o for _, o, _ in body)
        shifts = frozenset(int(im[0], 16) for _, o, full in body if o == 'SHF'
                           for im in [re.findall(r",\s*0x([0-9a-f]+),", full)] if im)
        rows.append({'start': target, 'end': addr, 'n': len(body), 'psl': psl, 'cnt': cnt,
                     'k': SHIFT_SETS.get(shifts, 'unknown')})
    rows.sort(key=lambda x: (x['end'], -x['n']))
    dedup = []
    for r in rows:
        if dedup and dedup[-1]['end'] == r['end']:
            continue
        dedup.append(r)
    return dedup

# K label by the set of SHF immediates in the loop (ptxas loop layout order is not source order).
SHIFT_SETS = {
    frozenset((2, 3, 5, 6, 8, 9, 11)): 'K1.5',
    frozenset((2, 4, 6, 8, 10, 12, 14)): 'K2',
    frozenset((3, 5, 8, 10, 13, 15, 18)): 'K2.5',
    frozenset((4, 7, 11, 14, 18, 21, 25)): 'K3',
    frozenset((3, 6, 9, 12, 15, 18, 21)): 'K3.5',
    frozenset((4, 8, 12, 16, 20, 24, 28)): 'K4',
}
# Per-K loop shape without NOPs: (instructions, SHF, LDG). K2.5-K4 get their other
# ring words from donor lanes via SHFL.IDX (one LDG stream). K1.5/K2 use the TOP32
# two-word stream with W1 from a donor lane via SHFL.IDX (one LDG stream, no lo shift).
LOOP_SHAPE = {
    'K1.5': (225, 32, 4), 'K2': (225, 32, 4),
    'K2.5': (233, 36, 4), 'K3': (233, 36, 4), 'K3.5': (233, 36, 4), 'K4': (233, 36, 4),
}

def require_loop_shape(rows, nops):
    by_k = {}
    for r in rows:
        if r['k'] in LOOP_SHAPE:
            if r['k'] in by_k:
                raise Ch2Fail(f"duplicate CH2 loop for {r['k']}")
            by_k[r['k']] = r
    if set(by_k) != set(LOOP_SHAPE):
        raise Ch2Fail(f'expected six CH2 loops, found {sorted(by_k)}')
    for name, (n, shf, ldg) in LOOP_SHAPE.items():
        r = by_k[name]
        c = r['cnt']
        want_n = n + (32 if nops else 0)
        if r['n'] != want_n or c['NOP'] != (32 if nops else 0):
            raise Ch2Fail(f'{name} loop count mismatch: ins={r["n"]} NOP={c["NOP"]} (want {want_n})')
        for op, want in [('PSL', 32), ('XMAD', 64), ('SHF', shf), ('IDP', 32), ('FFMA', 32), ('LDS', 8), ('LDG', ldg)]:
            got = r['psl'] if op == 'PSL' else c[op]
            if got != want:
                raise Ch2Fail(f'{name} loop {op} count mismatch: {got} != {want}')

def instruction_offsets(sass):
    chunk = target_chunk(sass)
    exits, shfls = [], []
    for a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", chunk):
        op = re.sub(r"^@!?P\d\s+", "", o.strip()).split()[0]
        if op == 'EXIT':
            exits.append(int(a, 16))
        elif op.startswith('SHFL'):
            shfls.append(int(a, 16))
    return exits, shfls

def nvinfo_attrs(blob):
    pos = 0
    out = []
    while pos + 4 <= len(blob):
        vfmt, at = blob[pos], blob[pos + 1]
        key = (at << 8) + vfmt
        if vfmt == 1:
            out.append((key, None, pos, 4)); pos += 4
        elif vfmt in (2, 3):
            out.append((key, int.from_bytes(blob[pos + 2:pos + 4], 'little'), pos, 4)); pos += 4
        elif vfmt == 4:
            n = int.from_bytes(blob[pos + 2:pos + 4], 'little')
            vals = [int.from_bytes(blob[pos + 4 + 4*i:pos + 8 + 4*i], 'little') for i in range(n // 4)]
            out.append((key, vals, pos, 4 + n)); pos += 4 + n
        else:
            raise Ch2Fail(f'unknown nv.info format {vfmt}')
    if pos != len(blob):
        raise Ch2Fail('trailing garbage in nv.info')
    return out

def patch_and_verify_nvinfo(cubin, sass, pyelftools_note):
    try:
        from elftools.elf.elffile import ELFFile
    except Exception as e:
        raise Ch2Fail(f'pyelftools unavailable in CuAssembler python: {e}')
    data = bytearray(cubin.read_bytes())
    with cubin.open('rb') as f:
        elf = ELFFile(f)
        secs = [s for s in elf.iter_sections() if 'gemv_moe_s' in s.name]
        infos = [s for s in secs if s.name.startswith('.nv.info.')]
        if len(infos) != 1:
            raise Ch2Fail(f'expected one target nv.info section, found {len(infos)}')
        sec = infos[0]
        base, blob = sec['sh_offset'], bytes(data[sec['sh_offset']:sec['sh_offset'] + sec['sh_size']])
    attrs = nvinfo_attrs(blob)
    exits, shfls = instruction_offsets(sass)
    for key, actual, label in [(0x1c04, exits, 'EXIT'), (0x2804, shfls, 'COOP/SHFL')]:
        hits = [x for x in attrs if x[0] == key]
        if len(hits) != 1:
            raise Ch2Fail(f'expected one {label} nv.info attribute, found {len(hits)}')
        _, vals, pos, _size = hits[0]
        if vals != actual:
            if not isinstance(vals, list) or len(vals) != len(actual):
                raise Ch2Fail(f'{label} metadata length mismatch: {None if not isinstance(vals, list) else len(vals)} vs {len(actual)}')
            # CuAssembler reconstructs offset labels at control-word boundaries.  The
            # survivor sequence is unchanged, so rewrite the same-length table to the
            # true instruction starts before accepting the cubin.
            for i, v in enumerate(actual):
                data[base + pos + 4 + 4*i:base + pos + 8 + 4*i] = v.to_bytes(4, 'little')
    cubin.write_bytes(data)
    # Re-read and require exact equality.
    with cubin.open('rb') as f:
        elf = ELFFile(f)
        sec = next(s for s in elf.iter_sections() if s.name.startswith('.nv.info.') and 'gemv_moe_s' in s.name)
        blob = sec.data()
    attrs = nvinfo_attrs(blob)
    ex = next(v for k, v, _p, _s in attrs if k == 0x1c04)
    co = next(v for k, v, _p, _s in attrs if k == 0x2804)
    if ex != exits or co != shfls:
        raise Ch2Fail('EXIT/COOP metadata verification failed after rewrite')

def res_usage_ok(cuobj, cubin):
    txt = subprocess.check_output([str(cuobj), '-res-usage', str(cubin)], text=True)
    blocks = re.split(r'(?m)^\s*Function\s+', txt)
    hits = [b for b in blocks[1:] if 'gemv_moe_s' in b.split('\n', 1)[0]]
    if len(hits) != 1:
        raise Ch2Fail(f'expected one gemv_moe_s resource block, found {len(hits)}')
    m = re.search(r'REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)', hits[0])
    if not m:
        raise Ch2Fail('missing resource usage for gemv_moe_s')
    reg, stack, shared, local = map(int, m.groups())
    if reg != 40 or stack != 0 or local != 0 or shared != 9248:
        raise Ch2Fail(f'resource mismatch REG={reg} STACK={stack} SHARED={shared} LOCAL={local}')

def generate(obj, outcubin, workdir, cuda_home):
    home = Path.home()
    cuasm = Path(env_first('INKWELL_CUASM_ROOT', 'CUASMROOT') or home / '.cache/inkwell-sass-reasm/CuAssembler')
    venvpy = Path(env_first('INKWELL_CUASM_PYTHON', 'VENVPY') or home / '.cache/inkwell-sass-reasm/cuasm-venv/bin/python')
    missing = []
    if not (cuasm / 'bin/cuasm.py').is_file(): missing.append(str(cuasm / 'bin/cuasm.py'))
    if not venvpy.is_file(): missing.append(str(venvpy))
    if missing:
        raise Ch2Fail('CuAssembler unavailable (' + ', '.join(missing) + '); using the normal ptxas kernel')
    pin = '96a9f72baf00f40b9b299653fcef8d3e2b4a3d49'
    try:
        head = subprocess.check_output(['git', '-C', str(cuasm), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = ''
    if head != pin:
        raise Ch2Fail(f'CuAssembler pin mismatch: {head or "no git metadata"} != {pin}')
    parser_txt = (cuasm / 'CuAsm/CuAsmParser.py').read_text(errors='replace')
    nvinfo_txt = (cuasm / 'CuAsm/CuNVInfo.py').read_text(errors='replace')
    if "if (self.__mCuInsAsmRepos is None" not in parser_txt:
        raise Ch2Fail('CuAssembler local repository-selection patch missing')
    if 'if attr in d: # updated autogen/manual/offset-label attributes' not in nvinfo_txt:
        raise Ch2Fail('CuAssembler local nv.info-update patch missing')
    cuda = Path(cuda_home)
    cuobj = cuda / 'bin/cuobjdump'
    if not cuobj.is_file():
        raise Ch2Fail(f'cuobjdump missing at {cuobj}')
    ch2 = Path(__file__).resolve().parent
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    extract = workdir / 'extract'
    extract.mkdir()
    env = os.environ.copy()
    env['PYTHONPATH'] = str(cuasm) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PATH'] = str(cuda / 'bin') + os.pathsep + env.get('PATH', '')

    run([str(cuobj), '-xelf', 'all', str(obj)], extract, workdir / 'extract.log', env)
    cubins = sorted(extract.glob('*.sm_61.cubin'))
    if len(cubins) != 1:
        raise Ch2Fail(f'expected one embedded sm_61 cubin, found {len(cubins)}')
    unpatched = workdir / 'kernels_unpatched.sm_61.cubin'
    shutil.copyfile(cubins[0], unpatched)

    sassu = run([str(cuobj), '-sass', str(unpatched)], workdir, workdir / 'unpatched.sass', env, capture=True)
    (workdir / 'kernels_unpatched.sass').write_text(sassu)
    run([str(venvpy), str(ch2 / 'reasm_augment_repos.py'), 'kernels_unpatched.sass', 'Repos.sm_61.production.txt'],
        workdir, workdir / 'augment_repos.log', env)

    patched = workdir / 'kernels_patched_nops.sm_61.cubin'
    shutil.copyfile(unpatched, patched)
    pout = run([str(venvpy), str(ch2 / 'patch_sass.py'), str(patched), str(cuobj)], workdir, workdir / 'patch.log', env, capture=True)
    pm = re.search(r'fake=(\d+) lp=(\d+) words=(\d+) hi_d12d=(\d+)', pout)
    if not pm or tuple(map(int, pm.groups())) != (463, 467, 1710, 468):
        raise Ch2Fail('patched-with-NOPs baseline pattern mismatch: ' + (pout.strip() or 'no patch output'))
    sassp = run([str(cuobj), '-sass', str(patched)], workdir, workdir / 'patched.sass.stdout', env, capture=True)
    (workdir / 'kernels_patched_nops.sass').write_text(sassp)
    require_loop_shape(loop_rows(sassp), nops=True)

    run([str(venvpy), str(cuasm / 'bin/cuasm.py'), str(patched), '-o', 'kernels_patched_nops.cuasm'],
        workdir, workdir / 'cuasm_disassemble.log', env)
    run(['python3', str(ch2 / 'reasm_loop_ranges.py'), 'kernels_patched_nops.sass', 'loop_ranges.json'],
        workdir, workdir / 'loop_ranges.log', env)
    run([str(venvpy), str(ch2 / 'reasm_remove_loop_nops.py'), 'kernels_patched_nops.cuasm', 'kernels_nopless.cuasm', 'loop_ranges.json'],
        workdir, workdir / 'remove_nops.log', env)
    nopless = workdir / 'kernels_patched_nopless.sm_61.cubin'
    run([str(venvpy), str(ch2 / 'reasm_assemble_with_repos.py'), 'kernels_nopless.cuasm', 'Repos.sm_61.production.txt', str(nopless)],
        workdir, workdir / 'assemble_nopless.log', env)
    sassn = run([str(cuobj), '-sass', str(nopless)], workdir, workdir / 'nopless.sass.stdout', env, capture=True)
    (workdir / 'kernels_patched_nopless.sass').write_text(sassn)
    rows_n = loop_rows(sassn)
    require_loop_shape(rows_n, nops=False)
    patch_and_verify_nvinfo(nopless, sassn, 'metadata')
    res_usage_ok(cuobj, nopless)

    digest = hashlib.sha256(nopless.read_bytes()).hexdigest()
    tmp = outcubin.with_suffix(outcubin.suffix + '.tmp')
    shutil.copyfile(nopless, tmp)
    os.replace(tmp, outcubin)
    (workdir / 'success.txt').write_text(f'sha256={digest} bytes={outcubin.stat().st_size}\n')
    print(f'ch2nopless: generated sha256={digest} bytes={outcubin.stat().st_size}')

def main():
    obj = Path(sys.argv[1]).resolve()
    outcubin = Path(sys.argv[2]).resolve()
    workdir = Path(sys.argv[3]).resolve()
    cuda_home = sys.argv[4] if len(sys.argv) > 4 else str(Path.home() / 'lfm-gtx1060-bench/cuda-12.6')
    try:
        generate(obj, outcubin, workdir, cuda_home)
    except Ch2Fail as e:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / 'failure.txt').write_text(str(e) + '\n')
        print('ch2nopless: disabled: ' + str(e))
        return 2
    except Exception as e:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / 'failure.txt').write_text(str(e) + '\n' + traceback.format_exc())
        print('ch2nopless: disabled: internal generator error (see failure.txt)')
        return 2
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit('usage: build_nopless.py <kernels.o> <output.cubin> <workdir> [cuda-home]')
    raise SystemExit(main())
