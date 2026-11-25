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

def target_chunk(sass, kernel='gemv_moe_s'):
    chunks = re.split(r"\n\s*Function : ", sass)[1:]
    hits = [c for c in chunks if kernel in c.split('\n', 1)[0]]
    if len(hits) != 1:
        raise Ch2Fail(f'expected one {kernel} SASS function, found {len(hits)}')
    return hits[0]

def loop_rows(sass, kernel='gemv_moe_s'):
    chunk = target_chunk(sass, kernel)
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
        k = SHIFT_SETS.get(shifts, 'unknown')
        if k == 'unknown':
            if cnt['SHF'] == 24 and cnt['PRMT'] == 8:
                k = 'K2'
            elif cnt['SHF'] == 8 and cnt['PRMT'] == 20:
                k = 'K4'
        rows.append({'start': target, 'end': addr, 'n': len(body), 'psl': psl, 'cnt': cnt,
                     'k': k})
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
    frozenset((4, 7, 11, 14, 18, 21, 25)): 'K3.5',
    frozenset((3, 6, 9, 12, 15, 18, 21)): 'K3',
    frozenset((4, 8, 12, 16, 20, 24, 28)): 'K4',
}
# Per-K fixed-stride loop shape without NOPs:
# (instructions, SHF, LDG, SHFL).  The expert kernel carries one copy for
# tn=72 and one for tn=144; both must have exactly the same shape.
LOOP_SHAPE = {
    'K1.5': (215, 32, 4, 4), 'K2': (215, 24, 4, 4),
    'K2.5': (219, 36, 4, 4), 'K3': (219, 36, 4, 4),
    'K3.5': (223, 36, 4, 8), 'K4': (211, 8, 4, 4),
}
# Dense uses the generic runtime-stride implementation.  K3.5 is the only
# dense/generic case that still needs the W2 donor shuffle.
DENSE_LOOP_SHAPE = {
    'K3': (223, 36, 4, 4), 'K3.5': (228, 36, 4, 8), 'K4': (216, 8, 4, 4),
}

def require_loop_shape(rows, nops, shape=LOOP_SHAPE, label='CH2', copies=1):
    by_k = {}
    for r in rows:
        if r['k'] in shape:
            by_k.setdefault(r['k'], []).append(r)
    if set(by_k) != set(shape) or any(len(v) != copies for v in by_k.values()):
        raise Ch2Fail(f'expected {len(shape) * copies} {label} loops ({copies} copies each), found '
                      f'{sorted((k, len(v)) for k, v in by_k.items())}')
    for name, (n, shf, ldg, shfl) in shape.items():
        want_n = n + (32 if nops else 0)
        for r in by_k[name]:
            c = r['cnt']
            if r['n'] != want_n or c['NOP'] != (32 if nops else 0):
                raise Ch2Fail(f'{name} loop count mismatch: ins={r["n"]} NOP={c["NOP"]} (want {want_n})')
            for op, want in [('PSL', 32), ('XMAD', 64), ('SHF', shf), ('IDP', 32), ('FFMA', 32),
                             ('LDS', 4), ('LDG', ldg), ('SHFL', shfl)]:
                got = r['psl'] if op == 'PSL' else c[op]
                if got != want:
                    raise Ch2Fail(f'{name} loop {op} count mismatch: {got} != {want}')
            want_prmt = {'K2': 8, 'K4': 20}.get(name, 0)
            if c['PRMT'] != want_prmt:
                raise Ch2Fail(f'{name} loop PRMT count mismatch: {c["PRMT"]} != {want_prmt}')

def instruction_offsets(sass, kernel='gemv_moe_s'):
    chunk = target_chunk(sass, kernel)
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

def patch_and_verify_nvinfo(cubin, sass, kernels):
    try:
        from elftools.elf.elffile import ELFFile
    except Exception as e:
        raise Ch2Fail(f'pyelftools unavailable in CuAssembler python: {e}')
    data = bytearray(cubin.read_bytes())
    with cubin.open('rb') as f:
        elf = ELFFile(f)
        sections = list(elf.iter_sections())
    for kernel in kernels:
        infos = [sec for sec in sections if sec.name.startswith('.nv.info.') and kernel in sec.name]
        if len(infos) != 1:
            raise Ch2Fail(f'expected one {kernel} nv.info section, found {len(infos)}')
        sec = infos[0]
        base, blob = sec['sh_offset'], bytes(data[sec['sh_offset']:sec['sh_offset'] + sec['sh_size']])
        attrs = nvinfo_attrs(blob)
        exits, shfls = instruction_offsets(sass, kernel)
        for key, actual, label in [(0x1c04, exits, 'EXIT'), (0x2804, shfls, 'COOP/SHFL')]:
            hits = [x for x in attrs if x[0] == key]
            if len(hits) != 1:
                raise Ch2Fail(f'{kernel}: expected one {label} nv.info attribute, found {len(hits)}')
            _, vals, pos, _size = hits[0]
            if vals != actual:
                if not isinstance(vals, list) or len(vals) != len(actual):
                    raise Ch2Fail(f'{kernel}: {label} metadata length mismatch')
                for i, v in enumerate(actual):
                    data[base + pos + 4 + 4*i:base + pos + 8 + 4*i] = v.to_bytes(4, 'little')
    cubin.write_bytes(data)
    for kernel in kernels:
        with cubin.open('rb') as f:
            elf = ELFFile(f)
            sec = next(s for s in elf.iter_sections() if s.name.startswith('.nv.info.') and kernel in s.name)
            blob = sec.data()
        attrs = nvinfo_attrs(blob)
        exits, shfls = instruction_offsets(sass, kernel)
        ex = next(v for k, v, _p, _s in attrs if k == 0x1c04)
        co = next(v for k, v, _p, _s in attrs if k == 0x2804)
        if ex != exits or co != shfls:
            raise Ch2Fail(f'{kernel}: EXIT/COOP metadata verification failed after rewrite')


def brx_count(sass, kernel):
    chunk = target_chunk(sass, kernel)
    n = 0
    for _a, o in re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?);", chunk):
        op = re.sub(r"^@!?P\d\s+", "", o.strip()).split()[0]
        n += op == 'BRX'
    return n

def indirect_target_records(blob):
    vals = [v for k, v, _p, _s in nvinfo_attrs(blob) if k == 0x3404]
    if len(vals) != 1:
        raise Ch2Fail(f'expected one indirect-branch nv.info attribute, found {len(vals)}')
    vals = vals[0]
    records = []
    for i in range(len(vals) - 4):
        n = vals[i + 2]
        if vals[i + 1] == 0 and 1 < n <= 16 and i + 3 + n <= len(vals):
            records.append((vals[i], vals[i + 3:i + 3 + n]))
    return records

def kernel_section(cubin, kernel, prefix):
    from elftools.elf.elffile import ELFFile
    with cubin.open('rb') as f:
        elf = ELFFile(f)
        hits = [sec for sec in elf.iter_sections() if sec.name.startswith(prefix) and kernel in sec.name]
        if len(hits) != 1:
            raise Ch2Fail(f'{kernel}: expected one {prefix} section, found {len(hits)}')
        sec = hits[0]
        return sec['sh_offset'], sec['sh_size'], sec.data()

def repair_indirect_branch_tables(patched_cubin, patched_sass, nopless_cubin, nopless_sass, kernels):
    """Copy CuAssembler's corrected indirect targets into raw constant-bank jump tables.

    cuasm represents .nv.constant2 as opaque bytes.  NOP deletion shifts case
    labels but cannot rewrite those bytes.  Its nv.info indirect-target list is
    label-based and is correct after reassembly, so use the input/output target
    lists as the authoritative old->new mapping.  Any mismatch is fail-closed.
    """
    data = bytearray(nopless_cubin.read_bytes())
    for kernel in kernels:
        want_brx = brx_count(patched_sass, kernel)
        if brx_count(nopless_sass, kernel) != want_brx:
            raise Ch2Fail(f'{kernel}: BRX count changed during reassembly')
        _ioff, _isz, iblob = kernel_section(patched_cubin, kernel, '.nv.info.')
        _ooff, _osz, oblob0 = kernel_section(nopless_cubin, kernel, '.nv.info.')
        rin = indirect_target_records(iblob)
        rout = indirect_target_records(oblob0)
        if len(rin) != want_brx or len(rout) != want_brx:
            raise Ch2Fail(f'{kernel}: indirect metadata records {len(rin)}/{len(rout)} != BRX {want_brx}')
        mapping = {}
        for (bi, ti), (bo, to) in zip(rin, rout):
            if len(ti) != len(to):
                raise Ch2Fail(f'{kernel}: indirect target count mismatch at BRX {bi:x}/{bo:x}')
            for old, new in zip(ti, to):
                if old in mapping and mapping[old] != new:
                    raise Ch2Fail(f'{kernel}: inconsistent indirect target mapping for {old:x}')
                mapping[old] = new
        ioff, isz, iconst = kernel_section(patched_cubin, kernel, '.nv.constant2.')
        ooff, osz, oconst = kernel_section(nopless_cubin, kernel, '.nv.constant2.')
        if isz != osz or isz % 4:
            raise Ch2Fail(f'{kernel}: constant2 size mismatch/alignment {isz}/{osz}')
        expected = bytearray(iconst)
        changed = 0
        for pos in range(0, isz, 4):
            old = int.from_bytes(iconst[pos:pos + 4], 'little')
            if old in mapping:
                expected[pos:pos + 4] = mapping[old].to_bytes(4, 'little')
                changed += mapping[old] != old
        current = bytes(data[ooff:ooff + osz])
        if current == bytes(expected):
            continue
        if current != bytes(iconst):
            raise Ch2Fail(f'{kernel}: reassembled constant2 was neither input nor repaired target table')
        data[ooff:ooff + osz] = expected
    nopless_cubin.write_bytes(data)
    for kernel in kernels:
        _ioff, isz, iconst = kernel_section(patched_cubin, kernel, '.nv.constant2.')
        ooff, osz, oconst = kernel_section(nopless_cubin, kernel, '.nv.constant2.')
        expected = bytearray(iconst)
        # Recompute the same map for verification.  It is deliberately exact:
        # every constant-bank byte must either be unchanged or the nv.info target.
        _x, _y, iblob = kernel_section(patched_cubin, kernel, '.nv.info.')
        _x, _y, oblob0 = kernel_section(nopless_cubin, kernel, '.nv.info.')
        rin, rout = indirect_target_records(iblob), indirect_target_records(oblob0)
        mapping = {old: new for (_bi, ti), (_bo, to) in zip(rin, rout) for old, new in zip(ti, to)}
        for pos in range(0, isz, 4):
            old = int.from_bytes(iconst[pos:pos + 4], 'little')
            if old in mapping:
                expected[pos:pos + 4] = mapping[old].to_bytes(4, 'little')
        if bytes(data[ooff:ooff + osz]) != bytes(expected):
            raise Ch2Fail(f'{kernel}: indirect constant2 verification failed after rewrite')


def res_usage_ok(cuobj, cubin, kernel, want_reg, want_shared):
    txt = subprocess.check_output([str(cuobj), '-res-usage', str(cubin)], text=True)
    blocks = re.split(r'(?m)^\s*Function\s+', txt)
    hits = [b for b in blocks[1:] if kernel in b.split('\n', 1)[0]]
    if len(hits) != 1:
        raise Ch2Fail(f'expected one {kernel} resource block, found {len(hits)}')
    m = re.search(r'REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)', hits[0])
    if not m:
        raise Ch2Fail(f'missing resource usage for {kernel}')
    reg, stack, shared, local = map(int, m.groups())
    if reg != want_reg or stack != 0 or local != 0 or shared != want_shared:
        raise Ch2Fail(f'{kernel} resource mismatch REG={reg} STACK={stack} SHARED={shared} LOCAL={local}')

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
    if not pm or tuple(map(int, pm.groups())) != (1078, 1088, 4388, 1086):
        raise Ch2Fail('patched-with-NOPs baseline pattern mismatch: ' + (pout.strip() or 'no patch output'))
    sassp = run([str(cuobj), '-sass', str(patched)], workdir, workdir / 'patched.sass.stdout', env, capture=True)
    (workdir / 'kernels_patched_nops.sass').write_text(sassp)
    require_loop_shape(loop_rows(sassp), nops=True, copies=2)
    require_loop_shape(loop_rows(sassp, 'gemv_sf_ch2'), nops=True, shape=DENSE_LOOP_SHAPE, label='dense CH2')

    run([str(venvpy), str(cuasm / 'bin/cuasm.py'), str(patched), '-o', 'kernels_patched_nops.cuasm'],
        workdir, workdir / 'cuasm_disassemble.log', env)
    run(['python3', str(ch2 / 'reasm_loop_ranges.py'), 'kernels_patched_nops.sass', 'loop_ranges.json', 'gemv_moe_s', '12'],
        workdir, workdir / 'loop_ranges.log', env)
    run(['python3', str(ch2 / 'reasm_loop_ranges.py'), 'kernels_patched_nops.sass', 'loop_ranges_dense.json', 'gemv_sf_ch2', '3'],
        workdir, workdir / 'loop_ranges_dense.log', env)
    run([str(venvpy), str(ch2 / 'reasm_remove_loop_nops.py'), 'kernels_patched_nops.cuasm', 'kernels_moe_nopless.cuasm', 'loop_ranges.json'],
        workdir, workdir / 'remove_nops.log', env)
    run([str(venvpy), str(ch2 / 'reasm_remove_loop_nops.py'), 'kernels_moe_nopless.cuasm', 'kernels_nopless.cuasm', 'loop_ranges_dense.json', '-', '.text._Z24inkwell_exl3_gemv_sf_ch2'],
        workdir, workdir / 'remove_nops_dense.log', env)
    nopless = workdir / 'kernels_patched_nopless.sm_61.cubin'
    run([str(venvpy), str(ch2 / 'reasm_assemble_with_repos.py'), 'kernels_nopless.cuasm', 'Repos.sm_61.production.txt', str(nopless)],
        workdir, workdir / 'assemble_nopless.log', env)
    sassn = run([str(cuobj), '-sass', str(nopless)], workdir, workdir / 'nopless.sass.stdout', env, capture=True)
    (workdir / 'kernels_patched_nopless.sass').write_text(sassn)
    rows_n = loop_rows(sassn)
    require_loop_shape(rows_n, nops=False, copies=2)
    require_loop_shape(loop_rows(sassn, 'gemv_sf_ch2'), nops=False, shape=DENSE_LOOP_SHAPE, label='dense CH2')
    patch_and_verify_nvinfo(nopless, sassn, ['gemv_moe_s', 'gemv_sf_ch2'])
    repair_indirect_branch_tables(patched, sassp, nopless, sassn, ['gemv_moe_s', 'gemv_sf_ch2'])
    res_usage_ok(cuobj, nopless, 'gemv_moe_s', 40, 9248)
    res_usage_ok(cuobj, nopless, 'gemv_sf_ch2', 40, 12848)

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
