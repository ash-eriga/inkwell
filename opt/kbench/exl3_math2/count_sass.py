#!/usr/bin/env python3
"""Parse cuobjdump -sass output and -Xptxas -v log for Round 2 EXL3 kernels.

Counts ISSUED warp-instructions per weight (corrected issue-bound cost model from opt/ledger.md):
  - Every issued instruction counts as 1.0 issue slot.
  - Per-pipe demand (SHF half-rate, SFU/I2F quarter-rate, LDST/SHFL quarter-rate) is reported separately.
  - Also parses registers/thread and stack spills from -Xptxas -v if passed as argv[2].
"""
import re
import sys
from collections import Counter

# (weights_per_loop_iter, batch_B, label)
META = {
    'k_shipped_m8_k2':       (32, 1, 'Shipped sgemv_run M=8 (unroll-4, I2F)'),
    'k_m8_opt_k2ILi0':       (32, 1, 'M=8-Opt C0c (L1-rot+mul16_imm+I2F, exact)'),
    'k_m8_opt_k2ILi1':       (32, 1, 'M=8-Opt C3b (Folded FP32 per-tile)'),
    'k_m8_opt_k2ILi2':       (32, 1, 'M=8-Opt C4  (IDP4A(-510)+XMAD.S16)'),
    'k_m8_opt_k2ILi3':       (32, 1, 'M=8-Opt C6  (dp4a-8)'),
    'k_paired_m16_k2ILi0':   (32, 1, 'M=16-Paired C0c (Zero-Centered I2F, exact)'),
    'k_paired_m16_k2ILi1':   (32, 1, 'M=16-Paired C1c (Magic FADD, exact)'),
    'k_paired_m16_k2ILi2':   (32, 1, 'M=16-Paired C3b (Folded FP32 per-tile)'),
    'k_paired_m16_k2ILi3':   (32, 1, 'M=16-Paired C3a (Folded FP32 global)'),
    'k_paired_m16_k2ILi4':   (32, 1, 'M=16-Paired C4  (IDP4A(-510)+XMAD.S16)'),
    'k_paired_m16_k2ILi5':   (32, 1, 'M=16-Paired C6  (dp4a-8)'),
    'k_paired_m16_k2ILi6':   (32, 1, 'M=16-Paired LEA+2*DP2A.LO'),
    'k_strided_m32_k2ILi0':  (32, 1, 'M=32-Strided C0c (SHFL+ISCADD+I2F, exact)'),
    'k_strided_m32_k2ILi1':  (32, 1, 'M=32-Strided C4  (SHFL+ISCADD+XMAD.S16)'),
    'k_strided_m32_k2ILi2':  (32, 1, 'M=32-Strided C6  (SHFL+ISCADD+dp4a-8)'),
    'k_lut128k_m16_k2':      (32, 1, 'M=16-LUT128K-Exact (128KB LDG LUT)'),
    'k_lut64k_m16_k2':       (32, 1, 'M=16-LUT64K-BS8   (64KB LDG LUT)'),
    'k_batch_m16_c0cILi1':   (16, 1, 'M=16-Batch B=1 C0c (exact)'),
    'k_batch_m16_c0cILi2':   (16, 2, 'M=16-Batch B=2 C0c (exact)'),
    'k_batch_m16_c0cILi3':   (16, 3, 'M=16-Batch B=3 C0c (exact)'),
    'k_batch_m16_c0cILi4':   (16, 4, 'M=16-Batch B=4 C0c (exact)'),
    'k_batch_m16_c4ILi2':    (16, 2, 'M=16-Batch B=2 C4  (int16 x)'),
    'k_batch_m16_c4ILi4':    (16, 4, 'M=16-Batch B=4 C4  (int16 x)'),
}

SHF_OPS  = {'SHF', 'SHR', 'SHL', 'BFE', 'BFI', 'PRMT'}
SFU_OPS  = {'I2F', 'F2I', 'F2F'}
LDST_OPS = {'LDG', 'LDS', 'LD', 'STG', 'STS', 'SHFL'}
MEM_CTRL = {'LDG', 'LDS', 'LD', 'STG', 'STS', 'BRA', 'ISETP', 'DEPBAR', 'NOP', 'BAR', 'STL', 'LDL'}

def parse_ptxas(path):
    info = {}
    if not path:
        return info
    cur = None
    for line in open(path):
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = m.group(1)
            info[cur] = {'regs': -1, 'spill': 0, 'smem': 0}
        elif cur:
            ms = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
            if ms:
                info[cur]['spill'] = int(ms.group(2)) + int(ms.group(3))
            mr = re.search(r'Used (\d+) registers(?:.*?(\d+) bytes smem)?', line)
            if mr:
                info[cur]['regs'] = int(mr.group(1))
                if mr.group(2):
                    info[cur]['smem'] = int(mr.group(2))
    return info

def parse_sass(path, ptxas_path=None):
    pinfo = parse_ptxas(ptxas_path)
    with open(path) as f:
        text = f.read()
    funcs = re.split(r'\n\s*Function : ', text)[1:]
    rows = []
    for fn in funcs:
        header = fn.splitlines()[0].strip()
        insns = []
        for line in fn.splitlines():
            m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:@\!?P\d+\s+)?([A-Z0-9_.]+)\s*(.*?);', line)
            if m:
                addr = int(m.group(1), 16)
                full_op = m.group(2)
                ops = m.group(3)
                base_op = full_op.split('.')[0]
                if base_op == 'IDP':
                    base_op = 'IDP4A' if '.4A' in full_op else 'IDP2A'
                insns.append((addr, base_op, full_op, ops))
        cands = []
        for idx, (addr, base_op, full_op, ops) in enumerate(insns):
            if base_op == 'BRA':
                tm = re.search(r'0x([0-9a-f]+)', ops)
                if tm:
                    target = int(tm.group(1), 16)
                    if target < addr:
                        sidx = next((i for i, x in enumerate(insns) if x[0] >= target), None)
                        if sidx is not None and (idx - sidx) >= 30:
                            cands.append((idx - sidx, sidx, idx))
        w, B, label = 32, 1, header
        for k, v in META.items():
            if k in header:
                w, B, label = v
                break
        if not cands:
            continue
        inner = [c for c in cands if not any(x[1] == 'BAR' for x in insns[c[1]:c[2]+1])]
        _, sidx, eidx = max(inner if inner else cands, key=lambda x: x[0])
        loop_insns = insns[sidx:eidx + 1]
        cnt = Counter(x[1] for x in loop_insns)
        n_tot = len(loop_insns)
        # Note: if unroll factor was doubled/halved by compiler, detect via LDG/IDP4A count!
        ipw = n_tot / w
        ipwt = n_tot / (w * B)
        n_shf = sum(cnt[op] for op in SHF_OPS)
        n_sfu = sum(cnt[op] for op in SFU_OPS)
        n_ldst = sum(cnt[op] for op in LDST_OPS)
        n_pure = n_tot - sum(cnt[op] for op in MEM_CTRL)
        pi = pinfo.get(header, {'regs': -1, 'spill': 0, 'smem': 0})
        rows.append((label, B, w, n_tot, ipw, ipwt, n_pure/w, n_shf/w, n_sfu/w, n_ldst/w, pi['regs'], pi['spill'], pi['smem'], cnt))
        print(f"\n=== {label} (w={w}, B={B}, insns={n_tot}, regs={pi['regs']}, spill={pi['spill']}B, smem={pi['smem']}B) ===")
        print(f"  ISSUED: {ipw:.3f} instr/w ({ipwt:.3f} instr/w/tok) | core={n_pure/w:.3f}/w | SHF={n_shf/w:.3f} I2F={n_sfu/w:.3f} LS+SHFL={n_ldst/w:.3f}")
        print(f"  Ops: " + ", ".join(f"{op}:{c}" for op, c in sorted(cnt.items())))

    print("\n" + "=" * 126)
    print(f"{'Kernel':45s} {'B':>2s} {'regs':>4s} {'spil':>4s} {'smem':>5s} {'ins/w':>6s} {'/w/tk':>6s} {'core':>6s} {'SHF/w':>6s} {'I2F/w':>6s} {'LS/w':>6s}")
    print("-" * 126)
    for r in rows:
        print(f"{r[0]:45s} {r[1]:2d} {r[10]:4d} {r[11]:4d} {r[12]:5d} {r[4]:6.3f} {r[5]:6.3f} {r[6]:6.3f} {r[7]:6.3f} {r[8]:6.3f} {r[9]:6.3f}")

if __name__ == '__main__':
    parse_sass(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
