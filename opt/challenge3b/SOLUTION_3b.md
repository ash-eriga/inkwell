# Challenge 3b — The Whole Decode Loop at its Mathematical Limit

> Source: GPT-6.2 Sol (Droid, Devbox), 2026-11-25. Original file is 17,558 bytes
> (sha256 6b9e8e719064a439ed4a2b49df825a7cccfa8651e88d24cacf187b26d3b8c40b).
> §0–§4 below are verbatim; §5–§9 are Sol's own summaries of the omitted text.
> `candidate_3b.c` (sha256 29215b60…408c, 7,563 bytes) re-verified locally against
> the Challenge 3 ref harness: ALL TESTS PASSED (BIT-EXACT).

**Task.** Bound SM-cycles/weight for the *whole* EXL3 `mul1` decode loop on `sm_61` —
core (`2 XMAD + IDP.4A + I2F + FFMA` per weight) **plus** window extraction (`SHF`),
ring loads (`LDG`), donor shuffles (`SHFL`), x delivery (`LDS`), and loop control —
under the Challenge-3 contract (bit-exact, fixed `FFMA.FTZ` accumulation order,
≤ 40 regs/thread, zero VRAM growth, any 0%-bloat load-time bit bijection allowed),
and construct a loop that attains the bound as tightly as possible.

**Result (both directions delivered).**

| quantity | value |
|---|---|
| Whole-loop floor (weighted over K-mix) | **≥ 1.2308 SM-cycle/weight** |
| Construction (weighted) | **1.2387 SM-cycle/weight** (0.63% above floor) |
| Shipped kernel floor (Challenge 2/3 accounting) | 1.4434 SM-cycle/weight |
| Construction vs shipped | **−14.18% cycles** (−9.79% instructions: 6.4421 vs 7.141 instr/w) |
| Time model (5.4332 ms/token @ 1.00 c/w) | 6.730 vs 7.842 ms/token → **1.112 ms/token saved** |
| Bit-exactness | `ref/verify` with `CAND=candidate_3b.c`: **ALL TESTS PASSED (BIT-EXACT)** |

Per-K numbers (cycles per 32 weights per lane, A2 dual-issue pipe model):

| K | SHF/32w | PRMT/32w | LDG/32w | primaries N | secondaries | cycles/32w | **constr. c/w** | **floor c/w** |
|---|---|---|---|---|---|---|---|---|
| 1.5 | 30 | 0 | 8  | 161 | 46 | 40.25 | 1.2578 | 1.2500 |
| 2   | 24 | 4 | 8  | 155 | 50 | 38.75 | 1.2109 | 1.2031 |
| 2.5 | 30 | 0 | 8  | 161 | 46 | 40.25 | 1.2578 | 1.2500 |
| 3   | 30 | 0 | 8  | 161 | 46 | 40.25 | 1.2578 | 1.2500 |
| 3.5 | 30 | 0 | 12 | 161 | 50 | 40.25 | 1.2578 | 1.2500 |
| 4   | 8  | 4 | 8  | 139 | 50 | 34.75 | 1.0859 | 1.0781 |
| **weighted** | | | | | | | **1.2387** | **1.2308** |

(K5: construction 1.2578, floor 1.2500; zero weight in the mix. Shares: 24.4/31.7/21.3/14.3/5.8/2.5 %.)

---

## 1. Assumptions (carried from Challenge 3, restated)

**A1 (ISA subset + measured pipes, PROBLEM.md §3.2–3.3).** Core-ALU 4/clk
(`FFMA/FADD/FMUL/XMAD/IDP.4A/LOP3/IADD/MOV/SEL/ISETP`), Shift 2/clk
(`SHF/SHR/SHL/BFE/BFI/ISCADD/IADD3`), **Permute 2/clk on its own pipe (`PRMT`,
silicon-verified separate from Shift: `SHF+PRMT` mix sustains 1.84 pairs/clk)**,
CVT/SFU 1/clk (`I2F/F2F/POPC/FLO/BREV`), SMEM/MIO 1/clk (`SHFL`, `LDS` passes:
1/2/4 for 32/64/128-bit), LSU ~1/clk issue (`LDG`, any width).

**A2 (cycle model, PROBLEM.md §3.3 model 2).** Per pass of $W_p$ weights:
$$\mathrm{Cycles}_{SM} \ge \max\!\left(\frac{N_{\text{ALU+Shift}}}{4},\; \frac{N_{\text{bundles}}}{4},\; \frac{H}{2},\; \frac{M}{2},\; V,\; S,\; 0.75(V+F)\right),$$
$N_{\text{ALU+Shift}}$ = Core-ALU+Shift instructions, $N_{\text{bundles}} = N - N_{\text{dual\_pairs}}$
(primary issue bundles; each primary can carry one dual-issued secondary),
$H$ = shift-pipe ops, $M$ = `PRMT`, $V$ = CVT ops, $S$ = SMEM passes, $F$ = `SHFL`.

**A3 (strict contract).** Fixed per-lane `FFMA.FTZ` chains and warp `FADD.FTZ` tree;
every weight must present the exact value $f_i = \mathrm{fl}_{f32}(s(w_i))$ as the
multiplicand of its specific `FFMA`. (Relaxed in §6.)

**A4 (resources).** ≤ 40 registers/thread, ≤ 16 KiB smem/block, zero VRAM growth.

**A5 (program class).** Warp-uniform straight-line inner loop; 0%-bloat load-time
bijective re-layout of the packed rings allowed.

---

## 2. Geometry lemmas (the whole loop's structure)

**L4 (window geometry; lead16/h5, exhaustive over the closed forms).**
For each $K$, window $j \in [0,8)$ of lane $t$ starts at ring bit
$q_{t,j} = 32 w_0(t) + o_t + D_j$ with $o_t = ((S_{8t}-16) \bmod R) \bmod 32$,
$D_j = j k_a + H\lfloor(j+1)/2\rfloor$. Exhaustive tabulation gives, per warp-tile:

| K | $o_t$ values | free (16-aligned) lane-windows | byte-aligned non-free, warp-uniform | diff-16 window pairs | max span $o_t{+}D_7{+}16$ |
|---|---|---|---|---|---|
| 1.5 | {1,5,…,29}×4 | 16 | none | none | 56 b → 2 words |
| 2 | {2,18}×16 | 32 | **j=3 (all 32 lanes)** | none | 48 b → 2 words |
| 2.5 | {2,6,…,30}×4 | 16 | none | none | 64 b → 2 words |
| 3 | {3,11,19,27}×8 | 16 | none | none | 64 b → 2 words |
| 3.5 | {3,7,…,31}×4 | 16 | none | none | 72 b → 3 words |
| 4 | {20}×32 | 64 | **j=1, j=5 (all lanes)** | (0,4),(1,5),(2,6),(3,7) | 64 b → 2 words |
| 5 | {5,13,21,29}×8 | 16 | none | none | 80 b → 3 words |

*Verified independently by lead16 (formula recompute + optimal-pairing brute-force
matching) and h5; both correct an earlier brief estimate of 31 SHF/32w for K3.5 to 30.
Note: h5's late addendum revives 31 by charging +1 SHF on the 8 "spill" lanes
(o_t ∈ {27, 31}, window 7 spans bits 52–72). That charge is a bookkeeping error:
window 7 there is one ordinary funnel from the bracketing pair (W1, W2) —
e.g. o = 31 ⇒ p = 56 ⇒ `SHF.L.W(W2, W1, 24)` — and lead16/lead18 executed exactly
that 1-op schedule bit-exactly on those lanes (4000 rings × 32 lanes, 0 mismatches).
The third word is a *load* cost (K3.5/K5: 12 LDG/32w vs 8, LSU term, non-binding),
not an extraction-op cost. K3.5 = 30 stands (h7 independently agrees).*

**L4′ (bit-locality).** Every window of every lane lies inside 2 (resp. 3)
consecutive ring words starting at $w_0(t)$; immediate-offset `LDG.32`s from one
per-lane base register fetch them with **no donor `SHFL`s at all**.

## 3. Theorem (whole-loop floor)

**Theorem (3b).** Under A1–A5, every bit-exact implementation of the whole decode
loop satisfies, per 32 weights per lane,
$$\mathrm{Cycles}_{SM} \;\ge\; \frac{128 + E_{\min}(K) + 2}{4},$$
with $E_{\min}(K) = 30, 24, 30, 30, 30, 8\,(30)$ for $K = 1.5, 2, 2.5, 3, 3.5, 4\,(5)$;
weighted over the K-mix: **≥ 1.2308 cycle/weight**.

**Proof.** Three additive lower bounds on $N_{\text{ALU+Shift}}$ (the binding A2 term
throughout), plus non-binding checks on every other pipe term.

*(i) Core ≥ 128.* By Challenge 3 (L1+L3, machine-checked): the hash needs ≥ 3
Core-ALU ops/weight and the fixed chain needs 1 `FFMA`/weight (Core-ALU); the
conversion sits on the CVT pipe (`I2F`, $V = 32$) — any `FADD` re-route only
*adds* to $N$. Hence $N_{\text{core}} \ge 4 \times 32 = 128$.

*(ii) Extraction ≥ $E_{\min}(K)$.* The hash's first `XMAD` consumes $w$ as a 16-bit
half-register operand. A window is available at zero $N$-ops iff it already occupies
a half-word of a loaded ring word, i.e. iff its ring position $q$ is 16-aligned —
an intrinsic property of $q \bmod 16$, unaffected by load width or count (L4′).
Every other window needs ≥ 1 instruction to present it. The only instruction that
presents a misaligned window **without** entering $N_{\text{ALU+Shift}}$ is `PRMT`
(own pipe, dual-issues as secondary; CVT-pipe `BREV/FLO/POPC` cannot shift,
SMEM/LSU ops cost their own pipe passes and never produce register halves cheaper).
`PRMT` is byte-granular, so it serves exactly the byte-aligned non-free windows
(L4: warp-uniform ones exist only for K2 {j=3} and K4 {j=1,5}).
One 32-bit instruction result covers ≤ 2 windows (its `.H0/.H1` halves) and only
when their offsets differ by exactly 16; the maximum diff-16 matching among the
remaining windows (L4: only K4, matching {(0,4),(2,6)} after (1,5) routes to one
`PRMT`) gives
$$E_{\min} = 4 \times \big( |\text{non-free, non-byte windows per lane-tile}| - |\text{max diff-16 matching}| \big),$$
evaluating to $\{30, 24, 30, 30, 30, 8, 30\}$ per 32 weights (lead16: matching
proven optimal by exhaustive search over all matchings of 8 windows).

*(iii) Control ≥ 2.* Any tile loop needs per iteration ≥ 1 base/induction update
(`IADD`-class, in $N$ even as half-rate `IADD3`) and ≥ 1 termination condition
(`ISETP`, or a fused `IADD.CC` carry — then the base advance still stands alone).
With a 4-tile (32-weight) unroll: ≥ 2 per 32 weights. (Deeper unrolls amortize
further; the tabulated floor keeps the conservative 2.)

*Other terms, non-binding at the floor:* $V = 32 \Rightarrow 32$;
$H/2 = E/2 \le 15$; $M/2 \le 2$; $S = 16$ (4 conflict-free `LDS.128` × 4 passes);
$0.75(V{+}F) = 24$ ($F = 0$); LSU: 8–12 `LDG` ≈ 8–12. All $< (128{+}E{+}2)/4 \ge 34.5$.
Secondaries (32 `I2F` + 0–4 `PRMT` + 8–12 `LDG` + 4 `LDS` + `BRA` + `DEPBAR` ≈ 46–54)
pair under the $\ge 138$ primaries, so $N_{\text{bundles}} = N_{\text{ALU+Shift}}$. $\square$

*Relayout clause.* A 0%-bloat bijection cannot lower $E_{\min}$: §5.

## 4. Construction (attains the floor to within 0.63%)

`ref/candidate_3b.c`, verified bit-exact by the harness
(`ALL TESTS PASSED (BIT-EXACT)`: 65,536 exhaustive states; 9,683,904 tile weights;
8,256 GEMV outputs).

**Per lane per tile** (SASS-level idiom, warp-uniform):
1. **Ring loads (L4′).** `LDG.32 R4,[R2.64+i0]`; `LDG.32 R5,[R2.64+i0+4]`
   (+`LDG.32 R6,[R2.64+i0+8]` for K3.5/K5) — per-lane base register computed once
   per tile-column pass, per-tile advance folded into immediates. No `SHFL`.
2. **Extraction (per window $j$, stream bit $p = o_t + D_j$):**
   - $p \bmod 16 = 0$: **free** — the window *is* `.H0`/`.H1` of a loaded word;
     the first `XMAD` reads it with its half-selector (0 instructions);
   - $p \bmod 8 = 0$: **one `PRMT`** (K2 j=3; K4 j=1,j=5). For K4 the pair
     (j=1, p=24; j=5, p=40) comes from **a single `PRMT sel=0x0765`** whose `.H1`
     and `.H0` feed the two hash chains (lead18-verified on 2000 rings × 32 lanes);
   - else: **one `SHF.L.W`** on the raw bracketing word pair — no base-alignment
     funnels (the shipped kernel spends 2 extra `SHF`/lane/tile on those).
3. **Core (unchanged, proven optimal).** Per weight: `XMAD; XMAD.PSL` (hash
   $p = w \cdot \texttt{0x83DCD12D} \bmod 2^{32}$), `IDP.4A.U8.U8` (bytesum),
   `I2F.F32.S32`, `FFMA.FTZ` into the fixed chain.
4. **x delivery.** One `LDS.128`/lane/tile from a shared-memory x copy permuted at
   block start so that lane groups $r_0 = 2(t \bmod 4)$ are conflict-free
   (4 floats: `x[r0], x[r0+1], x[r0+8], x[r0+9]`, each used by both chains).
5. **Control (per 32-weight unrolled iteration).** 2 `IADD` (ring base, x base) +
   1 `ISETP` + 1 `BRA` + 1 `DEPBAR`.

**Pipe accounting (per 32 weights, A2; lead16/lead18/h7 three-way agreement):**
primaries $N = 128 + E + 3$; secondaries 46–54 ≤ primaries ⇒ bundles = primaries;
$H/2 \le 15$, $M/2 \le 2$, $V = 32$, $S = 16$, $0.75V = 24$, LSU ≤ 12 — the
$N/4$ term binds everywhere. Register estimate ≈ 22 live (base pair, 3 ring words,
4 x, U/p/s temps, 2 accumulators, counter/predicate) ≤ 40. ✓ A4.

**Weighted result: 1.2387 c/w vs shipped 1.4434 (−14.18%); vs floor 1.2308 (+0.63%).**
Instruction counts: 5.906–6.594 instr/w, weighted 6.4421 vs shipped 7.141 (−9.79%).
The earlier engineering study's 6.131 instr/w is a lower *raw* count, but its schedule
is not dual-issue/pipe-costed; in SM-cycle terms the present construction is the
tightest known (1.2387 c/w vs shipped 1.4434).

---

## 5–9 (summaries by Sol; full text not transferred)

**§5 Relayout bound.** A 0%-bloat bijection cannot beat the contiguous-window optimum. Consecutive windows overlap 11–15 bits for all 256 cyclic pairs and all K, so windows form an inseparable 256-cycle. Contiguity-in-order is preserved by exactly the dihedral permutations (exhaustive 10! at n=10: exactly 20; 4M samples at n=16: zero non-dihedral). Rotations never increase free or byte-aligned counts (all 16 rotations × 7 K); reflections force one `BREV`/window (CVT → 2.00 c/w, worse). Status: part exhaustive, part 4M-sample evidence.

**§6 Relaxed-A3 corollary.** Any in-envelope FP32 summation order does not move the bound: (a) ≥ 1.000 multiply-op/weight in any order; (b) biased-direct-FFMA deviates 7.4e-4…7.8e-3 relscale vs envelope 2e-8…2.5e-7, so the exact per-weight conversion stays forced; (c) bucketing costs ≥ ~1.70 Core-ALU ops/w > 1 FFMA/w. Floor ≥ 1.2308 stands.

**§7 Ruled out.** Donor-SHFL retention; shipped base-alignment funnels; XMAD-as-shifter (can't cross .H0/.H1); PRMT beyond byte-aligned windows; wider LDG.64/.128 (w0 parity non-uniform; LSU non-binding); reflection relayouts (BREV); deeper unroll (residual gap is only 3-vs-2 control ops).

**§8 Validation log.** Harness 4 suites bit-exact; extraction tables three-way agreement (K3.5 = 30); 4000 rings × 256 windows × 7 K bit-exact; PRMT refinement; rotation scan; shipped floor 1.4434 reproduced.

**§9 Bottom line.** Floor 1.2308 weighted (per-K 1.078–1.250); construction 1.2387 (0.63% from floor, −14.18% vs shipped, ≈1.11 ms/token saved), bit-exact, no donor shuffles, no weight relayout.

## GPU porting notes (Sol)

Changes vs the shipped kernel:
(a) replace donor SHFLs with per-lane immediate-offset LDG.32s;
(b) drop the 2 base-alignment funnels and extract per-window directly (free halves via XMAD `.H0/.H1`, K2 j=3 via one PRMT, K4 pairs via funnels plus the single `PRMT 0x0765` for j=1/j=5);
(c) permute the smem x copy at block start for conflict-free LDS.128.
