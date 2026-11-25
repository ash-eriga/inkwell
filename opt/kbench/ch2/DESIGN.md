# Challenge 2 design note: exact EXL3 tile cores for the abstract sm_61 warp

**Status.** This is a complete exact scalar/window core and a concrete lane-map skeleton, with a
conservative SIMT cost model.  It is **not a proof of optimality**.  The genuinely new useful
result is the exact, proved pairing/core construction plus several scoped obstructions.  The x
layout needed for the headline vector-load count is stated as assumption A-X below; a fallback
cost without that assumption is also given.

## 1. Mathematical core

For a 16-bit window `w`, step 1's identity is

\[
cw \bmod 2^{32}=0x{\tt D12D}w+
  \bigl((0x{\tt 83DC}w)\bmod 2^{16}\bigr)2^{16}.
\]

Thus two XMADs produce `(c*w) mod 2^32`.  One unsigned `IDP.4A` against `0x01010101`
computes the byte sum `s`.  Initializing the IDP accumulator to `0x4B000000` instead of zero
produces the exact affine integer image `s+0x4B000000`, whose bit pattern is the fp32 number
`2^23+s` because `0 <= s <= 1005`.  One full-rate `FADD` of `-2^23` produces `fp32(s)` exactly,
and one FFMA forms the required term.  This is the F6 trick; no quarter-rate I2F is used.

Extraction uses the cyclic 32-bit funnel view `view(p)`.  The window starting at `p` is exactly
the high half of `view(p)`.  The window starting at `p+16` is exactly its low half.  Since XMAD
can select either half, a pair of windows separated by exactly 16 ring bits costs one SHF for
extraction, not two.  Contrary to a naive reading of the F3 table, K=1.5 has a perfect pairing:
for odd `i`, window `(i+11) mod 256` starts exactly 16 bits after window `i`, and those 128
pairs are disjoint and cover all 256 windows.  K=2 and K=4 also have perfect pairings:
`(16m+r,16m+r+8)` with `0<=r<8`, and `(8m+r,8m+r+4)` with `0<=r<4`, respectively.
K=2.5/3/3.5/5 have no exact +16 pairs.

The per-weight full-rate core is therefore exactly five issue slots:

```
XMAD       p, v.H?, 0xd12d, RZ
XMAD.PSL   p, v.H?, 0x83dc, p
IDP.4A     bits, p, ones, bias23       // bits = 0x4B000000+s
FADD       fs, bits, neg23             // fp32(s), exact
FFMA       acc[n], fs, x[k], acc[n]
```

No correctness-critical mask is needed: only the selected 16-bit half enters the XMAD.

## 2. Lane maps

All maps are bijections.  `ord` sorts by cyclic ring start position; ties do not occur.  Slots in a
pair are ordered high half first, low half second.

| K | lane map for lane `l`, slot order |
|---|---|
| 1.5 | pairs `P_m=(2m+1,(2m+12) mod 256)`, sorted by `b0` of the first member; lane `l` gets `P` indices `4l..4l+3` |
| 2 | pairs `(16m+r,16m+r+8)`, `0<=m<16`, `0<=r<8`, sorted by `b0`; lane `l` gets four consecutive pairs |
| 2.5 | windows `8l..8l+7` (equivalently the original consecutive schedule) |
| 3 | windows `8l..8l+7` |
| 3.5 | windows `8l..8l+7` |
| 4 | pairs `(8m+r,8m+r+4)`, `0<=m<32`, `0<=r<4`, sorted by `b0`; lane `l` gets four consecutive pairs |
| 5 | windows `8l..8l+7` |

Why these are reasonable secondary choices:

* K=1.5/2/4: four pairs per lane, so four extraction views per lane.
* Odd/unpaired K: eight views per lane, but consecutive windows give ring locality and only four
  distinct x columns per lane.
* The largest number of distinct output columns `n` in these lane groups is 4 for K=1.5/4 and 2
  for the other maps, so the accumulator pressure is small.

## 3. Steady-state cost

Per lane, one pass decodes `Wp=8` weights.  Counts below are the uniform straight-line counts;
lanes that happen to need less work are padded because the warp is SIMT.

* scalar core: `40` full-rate instructions (`5*8`);
* extraction SHF: `4` for K=1.5/2/4, otherwise `8`;
* ring loads: conservatively `2` `LDG.128` per tile per lane (the computed simple lane maps need
  at most two aligned four-word chunks; average is about 1.25–1.56);
* x loads: `x128` vector shared-memory loads, with `x128=2` for K=1.5/2 and `x128=1` for the
  other K under assumption A-X;
* loop/address/control: `1` per tile in the steady state, amortized by multi-tile unrolling.

Thus `N=40+Eshf+2+x128+1`, `H=Eshf`, and `Q=4*x128` (an `LDS.128` is counted as four passes).
Global-load latency is hidden by multi-tile software pipelining; the issue count is what enters
`N`.

| K | share | Eshf | x128 | N | H | Q | C_K=max(N,2H,4Q)/8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 24.4% | 4 | 2 | 49 | 4 | 8 | **6.125** |
| 2 | 31.7% | 4 | 2 | 49 | 4 | 8 | **6.125** |
| 2.5 | 21.3% | 8 | 1 | 52 | 8 | 4 | **6.500** |
| 3 | 14.3% | 8 | 1 | 52 | 8 | 4 | **6.500** |
| 3.5 | 5.8% | 8 | 1 | 52 | 8 | 4 | **6.500** |
| 4 | 2.5% | 4 | 1 | 48 | 4 | 4 | **6.000** |

Headline score:

\[
E = 0.244(6.125)+0.317(6.125)+(0.213+0.143+0.058)(6.5)+0.025(6)
  = \mathbf{6.277125}.
\]

Against today's `E=8.81`, this is a model-level saving of `2.533` issue-slot equivalents, or about
`3.5 ms/token` using the challenge's `1.4 ms` conversion, provided the loop remains issue-bound.

### A-X: x-layout assumption

The vector x count assumes a lossless per-block x permutation (and, where useful, limited
duplication) such that each lane's distinct x columns are covered by the stated number of
`LDS.128` instructions.  For the unpaired consecutive lane map this is easy: only four distinct
four-column sets occur and they partition the 16 columns.  For K=4 the four distinct sets also
partition the columns.  For K=1.5/2 the sets have size 6/8 and overlap; I did **not** complete a
proof that one non-duplicated permutation realizes all required pairs of contiguous runs.  This is
the one substantive layout assumption.  The x staging itself remains ring-independent and is filled
once per block.

A conservative scalar-x fallback, replacing those vector loads by one conflict-free `LDS.32` per
distinct x column, gives approximately:

| K | fallback C |
|---|---:|
| 1.5 | 6.625 |
| 2 | 6.875 |
| 2.5/3/3.5 | 6.875 |
| 4 | 6.375 |

Fallback `E ~= 6.78`.  Bank conflicts could make this worse; it is only a fallback bound, not the
proposed design.

## 4. Register allocation (peak <= 40)

A concrete allocation for the worst case (K=1.5/2 paired map) is:

| class | registers |
|---|---:|
| output accumulators (max 4 distinct n) | 4 |
| x registers | 8 |
| ring words from two `LDG.128` | 8 |
| funnel views | 4 |
| scalar temporaries `p,bits/fs` plus forwarding slack | 4 |
| base pointers, ring/x addresses, loop counter | 6 |
| per-lane shift/count invariants | 4 |
| load-in-flight/software-pipeline slack | 2 |
| **total** | **40** |

Odd-K maps need fewer x and accumulator registers but four more views, so the same bound covers
them.  No spills are required.  If wider latency hiding wants more tile lookahead, the current
design has no spare registers; that is a porting tradeoff, not a correctness issue.

## 5. Shared memory

* x staging: 9 KiB as in the existing design.
* ring-dependent tables: none.
* constant tables: none required; `ones`, `bias23` and `neg23` are uniform immediates/constants.
* additional layout metadata: at most a few hundred bytes if a permutation is materialized.

The design therefore stays below the 16 KiB block budget even before using the spare ~6.75 KiB.
No shared-memory table depends on the ring.

## 6. Lean 4 artifact

The Lean project is in `lean-challenge2/`.

Pinned versions:

* Lean toolchain `leanprover/lean4:v4.37.0-rc1`
* Mathlib commit `aad247f928b1eb7c58478742119fdd15a37f0ffd`

Important files:

* `Exl/Basic.lean`: exact reference ring/window/codebook definitions and structural facts.
* `Exl/Extraction.lean`: cyclic funnel-view semantics, exact high/low half extraction theorem for
  every valid K and every ring, and the exact 2-XMAD product theorem.
* `Exl/Program.lean`: straight-line programs as data, an integer ISA interpreter, per-window core
  correctness, F6 affine correctness, cost counts from program data, and the compiler-multiply
  sanity theorem.

Representative proved theorems:

* `window_eq_view_halves`: for every valid K, ring and `i<256`, the reference window is exactly
  the high half of one funnel view and the low half of the view 16 bits earlier.
* `candidate_correct` / `candidate_std_correct`: the funnel/2-XMAD candidate equals `s(w_i)` for
  every ring.
* `coreProgram_window_hi` and `coreProgram_window_lo`: execution of the program-as-data core on
  the appropriate view yields `sIdx` of the reference window.
* `coreProgramF6_correct`: the F6 program yields the exact affine integer image
  `s + 0x4B000000`.
* `core_count` and `coreF6_count`: the data-level core has exactly five instructions, by `rfl`.
* `compiler_mul32_sanity`: the documented three-XMAD compiler sequence is exact 32-bit
  multiplication for all 32-bit register inputs.
* `shf_counts`: the finite schedule extraction counts are kernel-decided.

Verification performed here:

```
cd lean-challenge2
lake build                  # passes
lake env lean Axioms.lean   # see axioms-final.log
```

There is no `sorry` in the `Exl` sources.  The main theorems use only the ordinary Lean/Mathlib
axioms `propext`, `Classical.choice`, and `Quot.sound`; finite cost facts are axiom-free.
`Exl/Basic.lean` also contains exploratory `native_decide`/`bv_decide` facts, but the extraction
and program correctness chain does not depend on them.

## 7. Scoped lower bounds and negative results

These are not full ISA-wide impossibility proofs, but they constrain the obvious design space.

1. **Five-slot scalar core under s-form.** If each weight must independently produce an fp32 copy
   of `s` and consume it in one FFMA, then the route used here has a hard five-instruction floor:
   two XMADs for the exact 16x16 constant multiplication, one byte reduction, one conversion, one
   FFMA.  Eliminating any one requires sharing arithmetic across weights or a different exact
   algebraic decomposition of `s`.
2. **Bit sensitivity.** The Lean finite theorem `s_bit_sensitive` shows that for every window and
   every one of its 16 bit positions, flipping that bit changes `s`.  Thus no exact method can
   ignore a fixed bit of the window.  This rules out direct byte truncations and most mask-only
   compression schemes.
3. **No single affine product.** `no_single_affine` proves the stated obstruction for a direct
   unwrapped affine product form over the initial counterexamples; the byte carries are essential.
4. **Pairing boundary.** In this ISA the only zero-cost extraction sharing mechanism found is the
   exact 16-bit half separation.  Nearby windows can share a funnel view, but the non-half-aligned
   window still needs a shift/extract instruction, so the issue count does not improve.  Hence the
   odd-K extraction penalty is structural unless a genuinely multi-window extraction instruction
   sequence is found.
5. **Fixed-point float reinterpretations.** Subnormal interpretation of integer `s` is exact but
   is separated from `s*x` by a factor `2^149`; for arbitrary finite fp32 x, every fixed rescaling
   either overflows or underflows in worst case.  Fixed-exponent biased representations introduce
   cancellation error.  These are not exact replacements under §1.4 without an approved error
   analysis.
6. **Byte-convolution linearization.** Writing the multiplication in byte columns gives exact
   carry structure, but the carry correction is not a small linear floor of the input bytes in any
   tested low-complexity form.  In particular, the residual after the best small-coefficient floor
   forms still depends on high bits of both input bytes.  I did not find a sub-three-instruction
   exact replacement for `2 XMAD + IDP`.

A full lower bound against arbitrary clever use of `PRMT`, `IDP.2A`, `LOP3`, lane shuffles and
cross-weight amortization remains open.  Consequently the costs above are upper bounds for a
correct design, not a proof that they are optimal.

## 8. Porting notes for Inkwell

* Interleave independent weight chains freely; correctness does not depend on the displayed
  serial order.  The cost model counts issue slots, not dependency latency.
* The IDP accumulator constant is `0x4B000000`; the FADD constant is fp32 `-2^23`.
* Accumulator ownership is determined by the lane map and the `(k_i,n_i)` bijection.  Within a
  tile all slots writing the same lane-local accumulator have the same `n`; across tiles the
  accumulators live for the whole k loop.  End-of-job cross-lane reduction is outside the
  steady-state pass and is amortized over 72/144 tiles.
* The cyclic tile halo must be included in the second ring chunk when a lane's chunk crosses the
  ring end.  The cost already uses the conservative maximum of two chunks.
