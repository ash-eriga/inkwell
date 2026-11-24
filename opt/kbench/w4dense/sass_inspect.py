#!/usr/bin/env python3
"""Inspect SASS (cuobjdump -sass) and ptxas resource usage (-Xptxas -v) for w4dense kernels on sm_61.

Reports per kernel:
  - registers/thread, stack spills (bytes), static smem
  - Pascal sm_61 theoretical occupancy (blocks/SM, warps/SM, % of 64 warps/SM)
  - inner-loop total issued instructions/weight, core ops/weight, I2F/w, FFMA/w, LDG/w, LDS/w
"""
import re
import sys
from collections import Counter

SHF_OPS  = {'SHF', 'SHR', 'SHL', 'BFE', 'BFI', 'PRMT'}
SFU_OPS  = {'I2F', 'F2I', 'F2F'}
LDST_OPS = {'LDG', 'LDS', 'LD', 'STG', 'STS', 'SHFL'}
MEM_CTRL = {'LDG', 'LDS', 'LD', 'STG', 'STS', 'BRA', 'ISETP', 'DEPBAR', 'NOP', 'BAR', 'STL', 'LDL'}

def sm61_occupancy(regs, threads, smem_bytes):
    """Pascal sm_61 (GP106): 65536 regs/SM (256-reg alloc granularity per warp),
    96 KB smem/SM (256B alloc granularity), max 2048 thr/SM (64 warps/SM), max 32 blks/SM."""
    warps_per_blk = (threads + 31) // 32
    if regs <= 0:
        return 0, 0, 0.0
    # Register allocation granularity on Pascal is 256 regs per warp
    regs_per_warp = ((regs * 32 + 255) // 256) * 256
    # Actually 65536 regs split across 4 warp schedulers? On sm_61 max warps = floor(65536/regs_per_warp)
    # clamped by regs per block aligned to 256 * warps_per_blk
    max_warps_reg = (65536 // regs_per_warp)
    blks_reg = max_warps_reg // warps_per_blk
    blks_thr = 2048 // threads
    if smem_bytes > 0:
        smem_aligned = ((smem_bytes + 255) // 256) * 256
        blks_smem = (96 * 1024) // smem_aligned
    else:
        blks_smem = 32
    blks = min(32, blks_reg, blks_thr, blks_smem)
    warps = blks * warps_per_blk
    return blks, warps, 100.0 * warps / 64.0

def parse_ptxas(path):
    info = {}
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

def short_name(mangled):
    # Readable short name for w4d:: templates + inkwell_exl3_gemv_s/sf
    if 'inkwell_exl3_gemv_sf' in mangled:
        return 'shipped:inkwell_exl3_gemv_sf'
    if 'inkwell_exl3_gemv_s' in mangled and 'sf' not in mangled:
        return 'shipped:inkwell_exl3_gemv_s'
    if 'gemv_wn_n32_f32' in mangled:
        m = re.search(r'ILi(\d+)ELi(\d+)ELb(\d+)ELi(\d+)ELi(\d+)ELb(\d+)', mangled)
        if m:
            bits, g, asym, nw, wpt, stg = map(int, m.groups())
            return f"n32_w{bits}_g{g}_{'asym' if asym else 'sym'}_w{nw}_k{wpt}_{'sm' if stg else 'ldg'}"
    if 'gemv_w4_n32_dp2a_i16' in mangled:
        m = re.search(r'ILi(\d+)ELb(\d+)ELi(\d+)ELi(\d+)', mangled)
        if m:
            g, asym, nw, wpt = map(int, m.groups())
            return f"n32_dp2a_g{g}_{'asym' if asym else 'sym'}_w{nw}_k{wpt}"
    if 'unembed_argmax_n32_f32' in mangled:
        m = re.search(r'ILi(\d+)ELi(\d+)ELb(\d+)ELi(\d+)', mangled)
        if m:
            bits, g, asym, nw = map(int, m.groups())
            return f"n32_argm_w{bits}_g{g}_{'asym' if asym else 'sym'}"
    if 'gemv_w4_f32' in mangled:
        m = re.search(r'ILi(\d+)ELb(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb(\d+)', mangled)
        if m:
            g, asym, r, nw, mode, stg = map(int, m.groups())
            mname = ['DUAL_I2F', 'BFE_I2F', 'LOP3_MAG', 'LOP3_FLD'][mode]
            return f"rm_w4_g{g}_{'asym' if asym else 'sym'}_{mname}_w{nw}_r{r}"
    if 'gemv_w4_dp2a_i16' in mangled:
        m = re.search(r'ILi(\d+)ELb(\d+)ELi(\d+)ELi(\d+)', mangled)
        if m:
            g, asym, r, nw = map(int, m.groups())
            return f"rm_dp2a_g{g}_{'asym' if asym else 'sym'}_w{nw}_r{r}"
    if 'unembed_argmax_f32' in mangled:
        m = re.search(r'ILi(\d+)ELi(\d+)ELb(\d+)ELi(\d+)', mangled)
        if m:
            bits, g, asym, nw = map(int, m.groups())
            return f"rm_argm_w{bits}_g{g}_{'asym' if asym else 'sym'}"
    return None

def weights_per_iter(mangled):
    if 'gemv_wn_n32_f32' in mangled or 'unembed_argmax_n32_f32' in mangled:
        m = re.search(r'ILi\d+ELi(\d+)', mangled)
        return int(m.group(1)) if m else 128
    if 'gemv_w4_n32_dp2a_i16' in mangled:
        m = re.search(r'ILi(\d+)', mangled)
        return int(m.group(1)) if m else 128
    if 'unembed_argmax_f32' in mangled:
        return 16
    m = re.search(r'gemv_w[456]_(?:f32|dp2a_i16).*?ILi\d+ELb\d+ELi(\d+)ELi(\d+)', mangled)
    if m:
        return 8 * int(m.group(1))
    return 8

def threads_of(mangled):
    if 'gemv_wn_n32_f32' in mangled:
        m = re.search(r'ILi\d+ELi\d+ELb\d+ELi(\d+)', mangled)
        return int(m.group(1)) * 32 if m else 256
    if 'gemv_w4_n32_dp2a_i16' in mangled:
        m = re.search(r'ILi\d+ELb\d+ELi(\d+)', mangled)
        return int(m.group(1)) * 32 if m else 256
    if 'unembed_argmax' in mangled:
        return 256
    m = re.search(r'gemv_w[456]_(?:f32|dp2a_i16).*?ILi\d+ELb\d+ELi\d+ELi(\d+)', mangled)
    return int(m.group(1)) * 32 if m else 256

def parse_sass(sass_path, ptxas_path):
    pinfo = parse_ptxas(ptxas_path)
    text = open(sass_path).read()
    funcs = re.split(r'\n\s*Function : ', text)[1:]

    print(f"{'Kernel':36s} {'thr':>3s} {'reg':>3s} {'spil':>4s} {'blk':>3s} {'occ%':>5s} {'w/it':>4s} {'ins':>4s} {'ins/w':>6s} {'core':>5s} {'I2F':>5s} {'FFMA':>5s} {'LDG':>5s} {'LDS':>5s} {'SHF':>5s}")
    print("-" * 124)

    for fn in funcs:
        hdr = fn.splitlines()[0].strip()
        sname = short_name(hdr)
        if not sname:
            continue
        insns = []
        for line in fn.splitlines():
            m = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:@\!?P\d+\s+)?([A-Z0-9_.]+)\s*(.*?);', line)
            if m:
                addr = int(m.group(1), 16)
                full_op = m.group(2)
                ops = m.group(3)
                base_op = full_op.split('.')[0]
                if base_op == 'IDP':
                    base_op = 'IDP2A' if '.2A' in full_op else 'IDP4A'
                insns.append((addr, base_op, full_op, ops))

        # Find innermost compute loop (backward BRA without BAR)
        cands = []
        for i, (addr, base, full, ops) in enumerate(insns):
            if base == 'BRA':
                tm = re.search(r'0x([0-9a-f]+)', ops)
                if tm:
                    tgt = int(tm.group(1), 16)
                    if tgt < addr:
                        s = next((j for j, x in enumerate(insns) if x[0] >= tgt), None)
                        if s is not None and (i - s) >= 15:
                            body = insns[s:i+1]
                            if not any(x[1] == 'BAR' for x in body):
                                # Prefer loops with FFMA or IDP2A
                                n_mac = sum(1 for x in body if x[1] in ('FFMA', 'IDP2A', 'IDP4A'))
                                if n_mac >= 4:
                                    cands.append((n_mac, i - s + 1, s, i))

        pi = pinfo.get(hdr, {'regs': -1, 'spill': 0, 'smem': 0})
        thr = threads_of(hdr)
        # Dynamic smem for K=2304, split_k=1 (9408B for smem f32, 4992B for i16, 192B for ldg)
        if 'shipped' in sname:
            dyn_sm = 12288
        elif '_ldg' in sname:
            dyn_sm = 192
        elif 'dp2a' in sname:
            dyn_sm = 4992
        else:
            dyn_sm = 9408
        blks, warps, occ = sm61_occupancy(pi['regs'], thr, pi['smem'] + dyn_sm)

        if not cands:
            continue
        # For shipped exl3, multiple K switch arms exist; skip detailed per-w here or pick K=4
        if 'shipped' in sname:
            print(f"{sname:36s} {thr:3d} {pi['regs']:3d} {pi['spill']:4d} {blks:3d} {occ:5.1f}  (7-way EXL3 switch: ~8.4-9.5 ins/w per exl3_math2)")
            continue

        # Pick innermost compute loop (smallest loop containing >= 4 MACs and no BAR;
        # avoids picking the outer row-stride loop in persistent unembed_argmax!)
        cands.sort(key=lambda x: (x[1], -x[0]))
        _, n_tot, s, e = cands[0]
        sub = insns[s:e+1]
        cnt = Counter(x[1] for x in sub)

        # Auto-detect compiler unroll factor via FFMA or IDP2A count
        w_base = weights_per_iter(hdr)
        if 'dp2a' in sname:
            # Each weight = 0.5 IDP2A (2 weights per IDP2A)
            w_eff = cnt['IDP2A'] * 2
        else:
            # Each weight = 1 FFMA (+ 1..2 scale/combine FFMA per 8 weights)
            # Round down to multiple of w_base
            n_f = cnt['FFMA']
            mult = max(1, round(n_f / (w_base * 1.15)))
            # More exact: use I2F count if present (exact 1 I2F per weight!)
            if cnt['I2F'] > 0:
                w_eff = cnt['I2F']
            elif 'LOP3_MAG' in sname:
                w_eff = cnt['FADD']
            elif 'LOP3_FLD' in sname:
                w_eff = cnt['LOP3']
            else:
                w_eff = w_base * mult

        n_pure = n_tot - sum(cnt[op] for op in MEM_CTRL)
        n_shf  = sum(cnt[op] for op in SHF_OPS)
        print(f"{sname:36s} {thr:3d} {pi['regs']:3d} {pi['spill']:4d} {blks:3d} {occ:5.1f} {w_eff:4d} {n_tot:4d} "
              f"{n_tot/w_eff:6.3f} {n_pure/w_eff:5.2f} {cnt['I2F']/w_eff:5.2f} "
              f"{(cnt['FFMA']+cnt['IDP2A'])/w_eff:5.2f} {cnt['LDG']/w_eff:5.2f} {cnt['LDS']/w_eff:5.2f} {n_shf/w_eff:5.2f}")

if __name__ == '__main__':
    parse_sass(sys.argv[1], sys.argv[2])
