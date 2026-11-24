#!/usr/bin/env python3
"""Parse cuobjdump -sass output of sass_kernels.o and count inner-loop opcodes and sm_61 slot-equivalents."""
import re
import sys
from collections import Counter

# Measured Pascal sm_61 throughput table (lane-ops/clock/SM):
#   FFMA 100, IDP4A 101, IDP2A 100, XMAD 103, SHF 53 (half rate), I2F 28 (quarter rate),
#   LOP/IADD/FADD full rate (100).
# Shifter-pipe ops (SHR/SHL/BFE/BFI/PRMT) share the 53/clk shifter pipe; LDG/LDS use 32/clk LD/ST pipe.
TPUT = {
    'FFMA': 100.0, 'FMUL': 100.0, 'FADD': 100.0, 'FADD32I': 100.0,
    'IDP4A': 101.0, 'IDP2A': 100.0, 'IDP': 101.0,
    'XMAD': 103.0,
    'SHF': 53.0, 'SHR': 53.0, 'SHL': 53.0, 'BFE': 53.0, 'BFI': 53.0, 'PRMT': 53.0,
    'I2F': 28.0, 'F2I': 28.0, 'F2F': 28.0,
    'LOP': 100.0, 'LOP3': 100.0, 'LOP32I': 100.0,
    'IADD': 100.0, 'IADD3': 100.0, 'IADD32I': 100.0, 'ISCADD': 100.0, 'LEA': 100.0,
    'MOV': 100.0, 'MOV32I': 100.0, 'SEL': 100.0, 'ISETP': 100.0, 'BRA': 100.0, 'DEPBAR': 100.0,
    'LDG': 32.0, 'LDS': 32.0, 'LD': 32.0, 'STG': 32.0,
}

WPI = {
    'k_base_k2ILi0': 8,
    'k_base_k2ILi1': 8,
    'k_base_k2ILi2': 8,
    'k_opt_k2_m32ILi0': 32,
    'k_opt_k2_m32ILi1': 32,
    'k_opt_k2_m32ILi2': 32,
    'k_opt_k2_m32ILi3': 32,
    'k_opt_k3_m32_c4': 32,
    'k_opt_k4_m32_c4': 32,
    'k_opt_k15_m32_c4': 32,
    'k_fail_incr_k2': 8,
    'k_fail_xmad_shift': 8,
    'k_fail_byte_share': 4,
}

MEM_OPS = {'LDG', 'LDS', 'LD', 'STG'}
CTRL_OPS = {'BRA', 'ISETP', 'MOV', 'MOV32I', 'NOP', 'DEPBAR', 'LEA', 'ISCADD'}
# Note: address arithmetic (IADD32I/IADD/SHR on pointers) is included in total_all;
# we report BOTH "decode+MAC" (excluding LDG/LDS/loop-ctrl/addr) AND "full loop" (every instruction in the loop).

def parse_sass(path):
    with open(path) as f:
        text = f.read()
    funcs = re.split(r'\n\s*Function : ', text)[1:]
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
                    if '.4A' in full_op:
                        base_op = 'IDP4A'
                    elif '.2A' in full_op:
                        base_op = 'IDP2A'
                insns.append((addr, base_op, full_op, ops))
        # Find largest backward branch loop
        loop_range = None
        for idx, (addr, base_op, full_op, ops) in enumerate(insns):
            if base_op == 'BRA':
                tm = re.search(r'0x([0-9a-f]+)', ops)
                if tm:
                    target = int(tm.group(1), 16)
                    if target < addr:
                        sidx = next((i for i, x in enumerate(insns) if x[0] >= target), None)
                        if sidx is not None and (idx - sidx) >= 15:
                            if loop_range is None or (idx - sidx) > (loop_range[1] - loop_range[0]):
                                loop_range = (sidx, idx)
        w = 8
        for k, v in WPI.items():
            if k in header:
                w = v
                break
        print(f"\n=== {header} (weights/iter={w}) ===")
        if not loop_range:
            print("  [no loop found]")
            continue
        sidx, eidx = loop_range
        loop_insns = insns[sidx:eidx + 1]
        cnt = Counter(x[1] for x in loop_insns)
        print(f"  Loop insns: {len(loop_insns)} (0x{insns[sidx][0]:x}..0x{insns[eidx][0]:x})")
        tot_all = 0.0
        tot_nomem = 0.0
        for op, c in sorted(cnt.items()):
            tp = TPUT.get(op, 100.0)
            sl = c * (100.0 / tp) / w
            tot_all += sl
            if op not in MEM_OPS and op not in {'BRA', 'ISETP', 'DEPBAR', 'NOP'}:
                tot_nomem += sl
            print(f"    {op:8s}: {c:4d} ({c/w:5.3f}/w)  tput={tp:5.1f}  slots/w={sl:6.3f}")
        xmad_psl = sum(1 for x in loop_insns if x[1] == 'XMAD' and '.PSL' in x[2])
        xmad_s16 = sum(1 for x in loop_insns if x[1] == 'XMAD' and '.S16' in x[2])
        print(f"  XMAD breakdown: total={cnt['XMAD']} ({cnt['XMAD']/w:.3f}/w), .PSL={xmad_psl}, .S16={xmad_s16}")
        # Core decode+MAC slots (only compute & bit-extract ops: SHF/BFE/SHR/SHL/PRMT/XMAD/IDP4A/IDP2A/I2F/FADD/FFMA/LOP/IADD-extract)
        core_ops = {'SHF', 'BFE', 'SHR', 'SHL', 'PRMT', 'XMAD', 'IDP4A', 'IDP2A', 'I2F', 'FADD', 'FADD32I', 'FFMA', 'LOP', 'LOP3', 'LOP32I', 'IADD'}
        # Note: subtract loop pointer-update IADDs/SHRs if we want pure decode, so print BOTH raw compute+extract AND full-loop!
        core_sl = sum(cnt[op] * (100.0 / TPUT[op]) / w for op in core_ops if op in cnt)
        print(f"  ==> Core compute+extract slots/w = {core_sl:.3f} | Non-mem slots/w = {tot_nomem:.3f} | Full loop (incl LDG/LDS/ctrl) = {tot_all:.3f}")

if __name__ == '__main__':
    parse_sass(sys.argv[1] if len(sys.argv) > 1 else 'opt/kbench/exl3_math/sass_dump.txt')
