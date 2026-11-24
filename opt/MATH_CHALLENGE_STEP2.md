# Step 2: put the 2-XMAD mul1 product into the shipped kernel, as a verified cubin rewrite (sm_61), proved in Lean 4

This continues `opt/MATH_CHALLENGE.md`. Step 1 proved the arithmetic. Step 2 turns it into machine code that we can ship. The work runs on a CPU alone: nvcc, ptxas, cuobjdump and nvdisasm all run without a GPU. The Inkwell team runs the silicon tests and the A/B.

**Context.** This is still about decompressing neural-network weights with the public EXL3 "mul1" codebook. The rewrite edits the SASS that NVIDIA's compiler produced from our own kernel source (`crates/inkwell-cuda/src/kernels.cu`), to remove redundant instructions. The integer results stay bit-identical, so model quality cannot change.

## 1. Step 1 results, and why they do not ship as they are
What step 1 proved, and what we keep:
- `prod2_eq` / `sCand_eq`: (c·w) mod 2^32 = 0xD12D·w + ((0x83DC·w) mod 2^16) << 16 for 16-bit w, which is 2 XMAD.
- `window_eq_view_halves`: one 32-bit view at bit offset D gives window D as its high half and window D+16 as its low half.
- The exhaustive CPU checks.

Why the 5.50–6.09 instr/w table does not reach the GPU:
1. **ptxas never emits the 2-XMAD form for mul1.** Every form we tried compiles to at least 3 instructions (3–4 XMAD, or 2 XMAD + SHL) with nvcc 12.6, sm_61:
   - the C split `h*0xD12D + ((h*0x83DC) << 16)`, which is folded back into one 32-bit multiply;
   - PTX `mul.wide.u16` plus `mad.lo.u32 q, 65536`;
   - an opaque constant from device memory, with `c * (v >> 16)`;
   - `cvt.u32.u16` plus `mul.lo.u32`;
   - `mad.lo.u32 w, 0x83DC0000, t`.

   ptxas also never reads `v >> 16` through the `.H1` selector inside the multiply. It emits SHR or BFE first.

   The 2-XMAD form is encodable all the same. ptxas emits both forms elsewhere: `XMAD R4, R3, 0x28, RZ` followed by `XMAD.PSL R23, R3.H1, 0x28, R4` in address arithmetic, and `XMAD R0, R2.H1, 0xd12d, RZ` for a PTX `mul.wide.u16` on the high half.
2. **The view cover assumed that one thread owns all 256 windows of a tile.** The shipped kernel gives each lane 8 consecutive windows per tile.
   - A variant with 32 windows per lane was built and measured. It issued 25% fewer instructions per weight and ran 40% slower: 48 registers (5 blocks/SM), or 40 registers with spills, and fewer warps on the small real grids.
   - With 8 windows per lane, (D, D+16) pairs exist only for K = 4.
3. **The cost model left out per-tile overhead.** That is LDG, LDS, IADD address arithmetic, the alignment SHFs and loop control: 1.4–2.1 instr/w in the loops below.
4. **Odd K has no pairs.** In the bench, the paired M=16 core ("pw16") ran 1.14–1.16× faster per K. On the real mix of all K it ran only 1.03× faster. MoE launches mix K, and the K = 2.5 / 3 / 3.5 blocks, which have no pairs, set the tail.
5. **K set.** The decode path uses K = 1.5–5: experts use 1.5–4, and dense attention uses 3–5. K = 6 occurs only in the vision encoder.
6. **The 9.47 in step 1 was the K = 3.5 loop.** The shipped main loops cost 8.41–9.47 instr/w, and the weighted mean for the experts is 8.81 (§2).

The arithmetic is right. What is missing is a way to get it past ptxas into the shipped layout. The one gain that applies to every K is the 2-XMAD product plus the instructions it makes dead.

## 2. The shipped loop, measured
Build: nvcc 12.6 with `-O3 -std=c++17 --use_fast_math -gencode arch=compute_61,code=sm_61`.

| Kernel | Registers | Spills | Role |
|---|---|---|---|
| `inkwell_exl3_gemv_moe_s` | 40 | none | experts |
| `inkwell_exl3_gemv_s` | 40 | none | dense |
| `inkwell_exl3_gemv_sf` | 48 | none | dense, fused |

The counts come from `opt/tools/sass_loops.py` run on `cuobjdump -sass`. The main loop is unrolled 4×, so one pass covers 4 tiles × 8 windows = 32 weights per lane.

Main loops of `gemv_moe_s` (per 32 weights). The `gemv_sf` loops differ by at most 5 instructions.

| K | Expert share | Loop instr | instr/w | XMAD | SHR | BFE | SHF | LOP32I (0xffff) | LDG | LDS | IADD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 24.4% | 269 | 8.41 | 96 | 4 | 28 | 4 | 0 | 8 | 8 | 16 |
| 2 | 31.7% | 270 | 8.44 | 96 | 4 | 28 | 4 | 0 | 8 | 8 | 16 |
| 2.5 | 21.3% | 295 | 9.22 | 96 | 4 | 24 | 12 | 4 | 12 | 8 | 24 |
| 3 | 14.3% | 299 | 9.34 | 96 | 4 | 20 | 16 | 8 | 12 | 8 | 24 |
| 3.5 | 5.8% | 303 | 9.47 | 96 | 4 | 16 | 20 | 12 | 12 | 8 | 24 |
| 4 | 2.5% | 302 | 9.44 | 96 | 4 | 12 | 20 | 16 | 12 | 8 | 24 |
| 5 | dense only | 300 | 9.38 | 96 | 4 | 12 | 24 | 16 | 12 | 8 | 24 |

- Each loop also holds IDP = I2F = FFMA = 32.
- The expert share is the fraction of the 3.06 G expert weights decoded per token.
- The weighted mean is 8.81 instr/w.

The source that produces these loops (`kernels.cu`):
```c
__device__ __forceinline__ uint32_t sgemv_mul16(uint32_t w) {   // (w & 0xffff) * 0x83DCD12D mod 2^32
    const uint32_t h = w & 0xffffu;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}
template <int D> __device__ __forceinline__ uint32_t sgemv_win(uint32_t hi, uint32_t lo) {
    if constexpr (D <= 16) return hi >> (16 - D);
    else return __funnelshift_r(lo, hi, 48 - D);
}
// per tile, lane t: 3 __ldg words W0..W2, hi = shf(W1,W0,o), lo = shf(W2,W1,o)
// window j = 0..7 at compile-time offset D_j = j*KA + H*((j+1)>>1)
pp[j] = sgemv_mul16(sgemv_win<D_j>(hi, lo));
f[j]  = (float)(int)__dp4a(pp[j], 0x01010101u, 0u);
a     = fmaf(x, f[j], a);
```

Each window becomes one of the following, according to its offset D in the 64-bit stream hi:lo (MSB-first):

| D | Extraction | Then |
|---|---|---|
| 0 | `SHR.U32 w, hi, 0x10` | 3 XMAD + IDP + I2F + FFMA |
| 1–15 | `BFE.U32 w, hi, 0x10pp` | same |
| 16 | `LOP32I.AND w, hi, 0xffff` | same |
| 17–47 | `SHF.R.W u, lo, s, hi` + `LOP32I.AND w, u, 0xffff` | same |

The product is always this sequence, where Rc = `MOV32I 0x83dcd12d`:
```
XMAD          a, w, Rc, RZ
XMAD.MRG      b, w, Rc.H1, RZ
XMAD.PSL.CBCC p, w.H1, b.H1, a
IDP.4A.U8.U8  s, p, c[0x2][..], RZ      ; 0x01010101
```

## 3. Track A (main deliverable): a verified rewrite of the shipped cubin
### Rewrite rules
| Rule | Old | New | Saved |
|---|---|---|---|
| R1 product | the XMAD / XMAD.MRG / XMAD.PSL.CBCC triple on Rc | `XMAD a, w, 0xd12d, RZ` + `XMAD.PSL p, w, 0x83dc, a` | 1 per weight |
| R2 D = 0 | `SHR.U32 w, hi, 0x10` + product | product on `hi.H1` | 1 per tile |
| R3 masks | `LOP32I.AND w, u, 0xffff` before the product | product on `u` (XMAD reads only the selected half) | 1 per window with D ≥ 16 |
| R4 K = 4 pairs | BFE for D = 4, 8, 12 and SHF + LOP for D = 20, 24, 28 | `SHF.R.W` views at D = 4, 8, 12; window D uses `.H1`, window D+16 uses `.H0` | 3 more per tile |

Expected main loop after the rewrite (count model; your tool reports the real count):

| K | Now | After | instr/w |
|---|---|---|---|
| 1.5 | 269 | 233 | 7.28 |
| 2 | 270 | 234 | 7.31 |
| 2.5 | 295 | 255 | 7.97 |
| 3 | 299 | 255 | 7.97 |
| 3.5 | 303 | 255 | 7.97 |
| 4 | 302 | 238 | 7.44 |
| 5 | 300 | 248 | 7.75 |

- Experts, weighted: 8.81 → 7.58 instr/w, which is −14%. That is ~3.8 G fewer lane-instructions per token. If the loops stay issue-bound, the ceiling is about −1.7 ms/token (1,280 lanes at ~1.7 GHz).
- The dense `gemv_s` / `gemv_sf` loops at K = 3–5 fall from 9.44–9.53 to 7.53–8.06 instr/w, on top of that.
- The rewrite keeps the launch geometry, the register count, and each warp's issue timing. Those are the three things that broke the 32-window variant.

### Site conditions
Each condition is checked by the tool, never assumed.
- Rc has exactly one reaching definition, a `MOV32I 0x83dcd12d`. Other XMADs (address math by 0x28, 0xa0, …) are never touched.
- Every definition the rewrite deletes is dead afterwards, by liveness over the whole CFG, the remainder loops included.
- No new registers are used. The REGCOUNT attribute does not change.
- The rewrite must be re-derived from ptxas output on every build. No hand-edited cubin is checked in. A site that does not match a rule exactly is skipped and reported, never forced.

### Control codes
Conservative rules, so that no instruction left in place issues earlier relative to its warp's previous instruction:
- When instruction i is deleted, add its stall count to i−1. If the sum passes 15, skip the site.
- OR i's wait-barrier mask into i+1.
- Skip any site where i sets a read or write barrier, or where i is dual-issued with a neighbour.
- Clear the reuse flags on every edited instruction and on its two neighbours.
- These rules keep every fixed-latency distance and every scoreboard wait. The gain is the freed dispatch slots, which other warps use.
- Tightening the stalls further is a separate, optional deliverable with its own measurement.

### Tooling
- Use CuAssembler or any sm_61 assembler.
- **Gate 0:** disassemble and reassemble our unpatched cubin, and get byte-identical `.text.*` and `.nv.info.*` sections. Edit nothing before this passes.
- Integration: `nvcc -dryrun` lists the compile pipeline (cicc → ptxas → fatbinary → host). Insert the patch between ptxas and fatbinary. Ship a script that build.rs can call to produce `kernels.o` with the patched cubin.
- The unpatched build stays selectable with `INKWELL_NO_SASSPATCH=1` at build time. That is our killswitch.

### Lean 4 (Mathlib; pin the toolchain and the Mathlib commit)
1. **Semantics.** A `BitVec 32` model of exactly the forms in the rules: `SHR.U32`, `BFE.U32`, `SHF.R.W`, `LOP32I.AND`, XMAD with `.H0`/`.H1` on a and b, a 16-bit immediate b, `.PSL`, `.MRG`, `.CBCC`, and `IDP.4A.U8.U8`. Cite the public source for each. These are hypotheses about the hardware, and our silicon test checks them.
2. **Model sanity theorem.** ptxas's own triple equals the 32-bit product mod 2^32 for all inputs. A model of MRG or CBCC that fails this is wrong.
3. **Rule theorems.** For R1–R4, the new sequence produces the same p as the old one for all register inputs, and p = (c · window_D) mod 2^32 with step 1's window definition. Reuse `prod2_eq` and `window_eq_view_halves`.
4. **Translation validation.**
   - The tool writes one certificate per site: old instructions, new instructions, and the register facts it relied on.
   - A separate checker re-verifies each certificate against the rule list and against its own liveness and reaching-definition analysis of the original SASS.
   - The checker must reject planted bad sites in its tests: a live MRG result, a non-mul1 constant, a masked value used elsewhere.

## 4. Track B (do it first; budget about one day): a source-level route
If some CUDA C or inline PTX makes ptxas emit `XMAD` + `XMAD.PSL` for mul1 with no third instruction, Track A shrinks to a source patch. Search systematically:
- PTX forms;
- the `-Xptxas` options;
- every toolkit that still targets sm_61.

Report every form and version tried. Accept a result only if the full production loop count drops in `sass_loops.py`. A toolkit other than 12.6 is acceptable only if all of `kernels.cu` builds with it and no other loop regresses.

## 5. Track C (optional): settle odd K in the lane layout
Given hi:lo and the 8 offsets D_j of one K, find the minimum number of single-issue ALU ops that make all 8 windows available as XMAD-selectable halves.

- **Conjecture.** The minimum equals the number of windows with D ∉ {0, 16, 32, 48}, minus the number of (D, D+16) pairs.
- **Proof.** Prove it in Lean for an explicit op set: SHF in every mode, SHR, SHL, BFE, PRMT in every mode, LOP3, IADD, ISCADD, LEA, and the XMAD forms.
- **Or refute it.** Give one op that yields two needed windows. That would cut extraction for K = 2.5 / 3 / 3.5, which are 41.4% of expert weights.
- **Shifted alignment.** Include a free choice of the per-lane alignment B, which shifts every D_j by the same amount.
- **Value of a negative result.** A proof tells us to stop looking.

## 6. Workflow and gates
1. **Falsify on CPU.**
   - Run Gate 0.
   - Search ptxas for Track B.
   - Patch one loop and diff it against the rule list.
   - Optionally emulate the edited straight-line region against the original on random registers.
2. **Lean.** `lake build` passes with no `sorry`. `#print axioms` shows only propext, Classical.choice and Quot.sound, plus `Lean.ofReduceBool` if `native_decide` or `bv_decide` is used and declared.
3. **Tool and checker.** The tool implements exactly the rules the Lean file proves. Every certificate checks.
4. **Port (done by the Inkwell team), after we re-check steps 2–3 on our machine:**
   - an exhaustive microkernel for each rule (all 2^16 windows × each offset class), bit-identical to the unpatched cubin;
   - `sgemv_gold` for MoE and dense on real dumps, bit-identical (not "within tolerance");
   - end-to-end md5 5cbe8e06;
   - an interleaved A/B of 16 pairs, accepted if the median is faster and the candidate wins ≥12/16;
   - nsys per kernel.

## 7. Deliverables
1. The rewrite tool and the certificate checker (Rust or Python), with positive and negative tests.
2. The patched build for `gemv_moe_s`, `gemv_s` and `gemv_sf`. The opt-in pw16 loops are optional.
3. A per-loop before/after table from `sass_loops.py`, with registers, sites patched and skipped per rule, and each skip's reason.
4. The build integration script (dryrun pipeline) and the `INKWELL_NO_SASSPATCH` switch.
5. The Lean project with its semantics, sanity theorem and rule theorems, plus a build log and an axioms log.
6. The Track B search log. The Track C result, if attempted.

## 8. Supplied
- `crates/inkwell-cuda/src/kernels.cu` and `build.rs` at the commit of this file. The compile flags are in §2.
- `opt/tools/sass_loops.py`.
- `opt/MATH_CHALLENGE.md`, for the window and decode definitions.
- On request: the unpatched `kernels.cubin`, its SASS, and real tile dumps for CPU tests.
