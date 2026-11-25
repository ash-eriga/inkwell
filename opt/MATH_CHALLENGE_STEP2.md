# Challenge 2: the cheapest exact SIMT program for the EXL3 trellis dot product, proved in Lean 4

**What we want.** A mathematical design, not a code patch.
- Design a family of exact programs, one per bit rate K, for an abstract GPU warp that is specified in full below (§2).
- Each program computes a 16×16 trellis-tile dot product (§1) using as few issue slots per weight as possible under a measured cost model (§2.5).
- Prove in Lean 4 that each program is exact, and prove its cost.
- Lower bounds, impossibility results and new reformulations count as results too.

**Self-contained.** Everything you need is in this file.
- You do not need our repository, our binaries, a GPU, the CUDA toolkit, SSH, or any access to our machine. Do not request any of them.
- If something seems missing, write down your assumption and continue. We check assumptions when you hand over.
- **The Inkwell team does all porting:** CUDA, SASS, the build, silicon tests and A/B timing. Your programs only need to be correct and cheap in the model of §2.

**Context.** This is decompression of neural-network weights in the public EXL3 format ("mul1" codebook). It is not cryptography. Every input is known, and the target is bit-exact evaluation with fewer instructions. Any loss of model quality is rejected (§4.2).

## 1. The object (exact)

### 1.1 A tile
A tile stores 256 weights as a cyclic bitstream, the **ring**, of R bits.
- The bit rate is K ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5}. Write K = ka + h/2 with h ∈ {0, 1}. Then R = 256·ka + 128·h.
- The ring is NW = R/32 32-bit words, read **MSB-first**: ring bit p is bit (31 − (p mod 32)) of word ⌊p/32⌋.
- **Schedule.** S_i = (i+1)·ka + h·⌊(i+1)/2⌋ for i = 0..255. For half-integer K, weight i advances the stream by ka bits if i is even and by ka+1 if i is odd.
- **Window.** Window i starts at bit b_i = (S_i − 16) mod R. Its value w_i is the 16 ring bits b_i, b_i+1, …, b_i+15, taken cyclically mod R. The first is the most significant:

  w_i = Σ_{m=0}^{15} ring[(b_i + m) mod R] · 2^(15−m)

- Consecutive windows overlap in 16 − (b_{i+1} − b_i) bits.
- The same definition as C, which is our reference implementation:
  ```c
  r_bits = h ? 256*ka + 128 : 256*ka;
  s_i    = h ? (i+1)*ka + ((i+1)>>1) : (i+1)*ka;
  b0     = s_i - 16 + (s_i < 16 ? r_bits : 0);
  nw = r_bits>>5; i0 = b0>>5; s0 = 48 - (b0 & 31); i1 = (i0+1 == nw) ? 0 : i0+1;
  w_i    = (uint32_t)((((uint64_t)ring[i0] << 32) | ring[i1]) >> s0) & 0xffff;
  ```

### 1.2 The codebook
- s(w) = bytesum((c·w) mod 2^32), where c = 0x83DCD12D and bytesum(a) is the sum of the four bytes of a.
- s(w) ∈ [0, 1005], and s takes 913 distinct values over the 65,536 windows.
- The weight is an affine function of s, applied once per output (§1.4). **The only quantity you must reproduce per weight is the integer s(w_i).**

### 1.3 Where each weight goes
Write i = 8t + j, with t = ⌊i/8⌋ and j = i mod 8. Weight i of a tile sits at tile row and column

  k_i = 2·(t mod 4) + (j mod 2) + 8·(⌊j/2⌋ mod 2),  n_i = ⌊t/4⌋ + 8·⌊j/4⌋.

This map is a bijection {0..255} → {0..15}².

### 1.4 The matrix product
- A matrix with input size k_dim and output size n_dim is stored as tiles T(a, q), with a < k_dim/16 and q < n_dim/16.
- Tile T(a, q) starts at word (a·n_dim/16 + q)·NW. NW is a multiple of 4, so tile offsets are multiples of 16 bytes.
- The input x is an fp32 vector of length k_dim, with arbitrary finite values. For each output column, the target is

  A[16q + n] = Σ_a Σ_{i : n_i = n} s(w_i^{T(a,q)}) · x[16a + k_i].

- The runtime then applies y = KINV·A + KOFF·Σx once per output. This step is outside the challenge.
- **Float rule.** Each term s·x is formed as an fp32 product inside an FFMA (or an FMUL), as today. The **order** of the fp32 accumulation is free, and we validate order changes for numeric noise on our side.
- Any other float formula needs our approval. Examples are summing x values first, or bucketing by s. If your idea needs one, give its worst-case error against exact real arithmetic, and we decide.

### 1.5 Test vectors
Take the ring words ring[m] = (0x9E3779B9·(m+1)) mod 2^32. For K = 2:
- w_0..w_4 = 6e42 b909 e427 909e 4278 (hex);
- s_0..s_4 = 652 661 891 696 491;
- Σ_i s_i = 132,911 and Σ_i i·s_i = 16,619,831.

For K = 2.5 with the same formula, so NW = 20:
- w_0..w_4 = 09d2 4e93 3a4f d278 49e3;
- s_0..s_4 = 641 815 737 538 653;
- Σ s_i = 132,937 and Σ i·s_i = 16,818,789.

## 2. The machine: one warp of an sm_61 GPU (GTX 1060)

### 2.1 Execution model
- A **warp** is 32 lanes that execute one instruction stream (SIMT). Every instruction runs on all 32 lanes at once.
- Each lane has its own 32-bit registers. Immediates and constant-bank operands are the same for every lane.
- A lane can differ from the others only through register contents. Examples are per-lane offsets, shift amounts, or per-lane slices of a table. Such registers can be computed once, before the loop over tiles. They then cost registers, but no per-tile instructions.
- Predication (`@P`) is allowed. A per-tile program should be straight-line: one program per K, with offsets fixed at compile time.
- A warp handles one or more output columns q. It loops over the tiles a of its k-range, with one pass of your program per tile or per group of g tiles.

### 2.2 Memory
- **Global.** The ring words of any tile, loaded by any lane, with any width. Each load instruction costs 1 issue slot, at 32, 64 or 128 bits per lane. Loading a word into several lanes is allowed. Latency is hundreds of cycles (not measured here), so programs must issue loads well ahead of use. Registers held for loads in flight count against the register budget.
- **Shared.** 32 banks of 4 bytes. A warp's LDS takes one pass per distinct 32-bit word requested from the most-loaded bank. Lanes that read the same word cost nothing extra (broadcast).
  - Our block (8 warps) stages x here once per launch, in a layout of your choice. Any per-block precomputed table also lives here (§2.6).
  - Table contents may depend on x or on constants. They may not depend on the ring.
- **Constants.** Uniform 32-bit values, free as an operand (§2.3).

### 2.3 Instructions
All integer arithmetic is mod 2^32. lo(a) = a mod 2^16, hi(a) = ⌊a/2^16⌋, and byte_j(a) = ⌊a/2^(8j)⌋ mod 2^8. Operand b may be a register, an immediate, or a uniform constant-bank word. Immediates are 20-bit for most ops, 32-bit for the `…32I` forms, and 16-bit for XMAD.

| Op | Semantics |
|---|---|
| `IADD d, a, b` / `IADD3 d, a, b, c` | a ± b; ±a ± b ± c |
| `ISCADD d, a, b, n` | (a·2^n) + b, with immediate n ∈ [0, 31] |
| `LOP3.LUT d, a, b, c, f` | any 3-input bitwise function f (8-bit truth table) |
| `LOP32I d, a, imm32` | a AND/OR/XOR imm32 |
| `SHL d, a, n`, `SHR d, a, n` | a·2^n mod 2^32, ⌊a/2^n⌋. A register n ≥ 32 gives 0 |
| `SHF.R.W d, a, n, b` | ⌊(b·2^32 + a) / 2^(n mod 32)⌋ mod 2^32 (funnel shift right) |
| `SHF.L.W d, a, n, b` | ⌊((b·2^32 + a)·2^(n mod 32)) / 2^32⌋ mod 2^32 (funnel shift left, high word) |
| `BFE.U32 d, a, (p, ℓ)` | ⌊a/2^p⌋ mod 2^ℓ |
| `BFI d, a, (p, ℓ), b` | b with bits [p, p+ℓ) replaced by the low ℓ bits of a |
| `PRMT d, a, sel, b` | byte r of d is byte sel_r of the 8-byte value b·2^32 + a, where sel_r ∈ 0..7 is nibble r of sel |
| `XMAD d, a.Hx, b.Hy, c` | (A·B) + C, where A = lo(a) or hi(a) and B = lo(b) or hi(b) or a 16-bit immediate |
| `  .PSL` | the product is multiplied by 2^16 first |
| `  .CBCC` | C = c + lo(b)·2^16, instead of c |
| `  .CHI` / `.CLO` | C = hi(c) / lo(c) |
| `  .MRG` | after the sum, the high half of d is replaced by lo(b) |
| `IDP.4A.U8.U8 d, a, b, c` | Σ_{j<4} byte_j(a)·byte_j(b) + c (an `.S8` form exists for each side) |
| `IDP.2A.LO/HI d, a, b, c` | lo(a)·byte_{0/2}(b) + hi(a)·byte_{1/3}(b) + c |
| `POPC`, `FLO`, `BREV`, `SEL`, `MOV`, `ISETP` | population count, find leading one, bit reverse, select, move, set predicate |
| `I2F.F32.S32 d, a` | fp32(a), exact for \|a\| < 2^24 |
| `FADD`, `FMUL`, `FFMA` | fp32, round to nearest even |
| `SHFL.IDX/UP/DOWN/BFLY d, a, b` | d = register a of lane σ(lane, b) |
| `LDS.32/.64/.128`, `LDG.32/.64/.128` | loads (§2.2) |

- **What is documented and what is not.**
  - The PTX-visible semantics come from NVIDIA's public PTX ISA manual: add, shifts, funnel shifts, bfe, bfi, prmt, lop3, dp4a/dp2a, popc, cvt, shfl.
  - The XMAD flags come from public reverse engineering of Maxwell/Pascal SASS (maxas). Treat them as hypotheses that we check on silicon.
  - **Sanity condition on your model:** the compiler's 32-bit multiply must come out exact in it, for all a and b:

    `XMAD t, a, b, RZ; XMAD.MRG u, a, b.H1, RZ; XMAD.PSL.CBCC d, a.H1, u.H1, t` gives d = a·b mod 2^32.

- There is no single-instruction 32×32-bit multiply on sm_61.

### 2.4 Resources
- **Registers.** The hard cap is 40 per thread for the whole loop. That covers your temporaries, loads in flight, x values, accumulators, per-lane invariants, pointers and the loop counter.
  - The register file holds 65,536 registers per SM, allocated per thread in multiples of 8.
  - With our 256-thread blocks, resident warps per SM depend on the register count:

    | Registers/thread | 32 | 40 | 48 | 56 | 64 |
    |---|---|---|---|---|---|
    | Warps/SM | 64 | 48 | 40 | 32 | 32 |

- **Shared memory.** 96 KiB per SM. To keep 6 blocks per SM, a block may use ≤ 16 KiB. Staging x for a full k-range takes 9 KiB today (2304 fp32), which leaves ~6.75 KiB for tables. A more compact x staging frees more.
- **Scale.** 10 SMs. The SM clock is ~1.76 GHz under sustained load. DRAM bandwidth is 192 GB/s.

### 2.5 Cost model (measured on our GTX 1060)
- Each SM has 4 schedulers. Each issues at most one warp-instruction per cycle.
- Some instruction classes run on pipes slower than issue. We measured the rates on the GTX 1060 with 32 warps per SM, each running 8 independent chains of one instruction. The unit is warp-instructions per cycle per SM; issue alone allows 4.

| Class | Instructions | Rate | Dependent-chain time, 1 warp |
|---|---|---|---|
| **Full** | FFMA, FADD, LOP3, IDP.4A, XMAD | 3.7–3.9 (issue-bound) | ~9 cycles |
| **Half** | SHF, SHR, BFE, BFI, PRMT, ISCADD | 1.93–2.00 | ~9 cycles |
| **Quarter** | I2F, POPC, SHFL, LDS.32 | 1.00 | I2F ~15; LDS ~26; SHFL ~29 cycles |

- **Not isolated.** IADD ran 1:1 with IDP.4A at 3.87 combined, which fits full rate. Treat IADD, LOP32I, MOV and SEL as full rate.
- **IADD3** ran at 1.87 per cycle in our test. Treat it as half rate.
- **LDS with bank conflicts.** LDS.32 at 32 random addresses ran at 0.45 per cycle, so random conflicts cost ~2.2 passes. Same-word reads (broadcast) ran at the full 1.00.
- **Wide LDS.** Not measured. Count an LDS.64 as 2 passes and an LDS.128 as 4 (128 bytes per pass).
- **Slow pipes run beside issue.**
  - 3 FFMA + 1 I2F sustained 3.98 instructions per cycle.
  - 3 FFMA + 1 LDS sustained 3.67.
  - 1 FFMA + 1 I2F (or LDS, or SHFL) was held to one pair per cycle by the slow op.
- **Shared or separate slow pipes.** Not yet known: whether the half-rate classes share one pipe, and whether I2F, SHFL and LDS share one. Assume they share (the conservative case). If your scheme only pays off when they are separate, say so, and we will measure it.
- **Cost of a program.** For one pass that decodes Wp weights per lane, count:
  - N, all instructions, including loads, address arithmetic, x reads and loop control;
  - H, the half-rate ones;
  - Q, the quarter-rate ones, with LDS counted in passes.

  Then cycles per pass per SM ≈ max(N/4, H/2, Q), and the cost per weight in issue-slot equivalents is

  **C_K = max(N, 2H, 4Q) / Wp.**

  - One I2F per weight alone forces C_K ≥ 4. F6 (§4.1) removes that floor.
  - A table lookup per weight by LDS or SHFL adds another 4 per weight on the quarter-rate side.
- **Headline score.** E = Σ_K share_K · C_K, with the expert shares of §3.
- **How costs turn into time.** At today's E = 8.81, the expert loop issues 842 M warp-instructions per token. At 40 issue slots per cycle and ~1.76 GHz, that is ~12.0 ms, and nsys measured ~16 ms. Each 1.0 of E saved is worth ~1.4 ms of the ~38 ms token time, as long as the loop stays issue-bound. The DRAM floor for the expert weights is ~4.5 ms per token, so issue stays the binding resource down to about E ≈ 3.3.

### 2.6 Parallel shape (experts, per token)
- 96 launches per token. Each has 1,152 independent jobs (tile column × expert × matrix).
- Each job is a 16-output dot product over 144 tiles (k_dim = 2304) or 72 tiles (k_dim = 1152).
- Today: one warp per job, 8 warps per block, 144 blocks per launch, and 60 resident blocks (10 SMs × 6), which is 2.4 waves.
- Any other mapping is allowed:
  - several jobs per warp;
  - a job split along k across warps, which costs one fp32 atomic add per output per split;
  - any lane-to-window map.

  State the mapping and count its overhead. Fewer, fatter warps lose latency hiding and fill the last wave worse (§4.2, L1).

## 3. Today's program, in the model
- Lane t owns windows 8t..8t+7. Their start offsets relative to a per-lane base B_t are D_j = j·ka + h·⌊(j+1)/2⌋, the same for every lane.
- **Per tile:**
  - 2 or 3 `LDG.32` and 2 `SHF` build the 64-bit stream hi:lo starting at B_t;
  - 2 `LDS.64` read the 4 x values;
  - 4–6 `IADD` do address arithmetic.
- **Per window:**
  - extraction: 1 `SHR` at D = 0, 1 `BFE` at D = 1–15, 1 `LOP32I` mask at D = 16, and `SHF` + `LOP32I` at D = 17–47;
  - the product: `XMAD`, `XMAD.MRG`, `XMAD.PSL.CBCC` with c in a register;
  - then `IDP.4A` (against 0x01010101), `I2F` and `FFMA`.
- 40 registers, 0 spills, unrolled over 4 tiles. Counted from the compiler's output for the main loop:

| K | Expert share | instr/weight (C_K today) |
|---|---|---|
| 1.5 | 24.4% | 8.41 |
| 2 | 31.7% | 8.44 |
| 2.5 | 21.3% | 9.22 |
| 3 | 14.3% | 9.34 |
| 3.5 | 5.8% | 9.47 |
| 4 | 2.5% | 9.44 |
| 5 | dense layers only | 9.38 |

- E = 8.81. Experts decode 3.06 G weights per token at a mean of 2.26 bits.
- **Where the 8.81 goes:**

  | Part | instr/weight |
  |---|---|
  | Product | 3 |
  | IDP + I2F + FFMA | 3 |
  | Window extraction | 1.0–1.5 |
  | Loads, x reads, address math and loop control | 1.4–2.1 |

## 4. What is known

### 4.1 Facts (checked on CPU or proved in step 1; use them or prove them)
- **F1, 2-XMAD product.** For w < 2^16: c·w mod 2^32 = 0xD12D·w + ((0x83DC·w) mod 2^16)·2^16. That is `XMAD p, w, 0xd12d, RZ` followed by `XMAD.PSL p, w, 0x83dc, p`. Our compiler never emits this form, but we can, so you may use it.
- **F2, register halves.** XMAD reads only the selected 16-bit half. A register holding ring bits [D, D+32), MSB-first, has window D in its high half and window D+16 in its low half. Neither needs masking.
- **F3, which windows pair up.** Count, per K, the windows i for which some window starts exactly 16 bits later:

  | K | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 5 |
  |---|---|---|---|---|---|---|---|
  | Windows | 128/256 | 256/256 | 0 | 0 | 0 | 256/256 | 0 |

  In today's lane map (8 consecutive windows per lane), only K = 4 has pairs inside one lane.
- **F4, sliding recurrence.** Let w' = ((w·2^K) mod 2^16) + b. Here hi = ⌊w / 2^(16−K)⌋ is the K bits shifted out and b the K bits shifted in. Then

  c·w' ≡ (c·w)·2^K + c·b − (0xD12D·hi mod 2^16)·2^16 (mod 2^32).

  The correction depends only on the 2K bits (hi, b). The same holds for half-integer K, step by step.
- **F5, residues.** bytesum(a) ≡ a (mod 255). But s(w) ≢ c·w (mod 255) for 65,278 of the 65,536 windows, because the reduction mod 2^32 subtracts ⌊c·w/2^32⌋. Since s ≤ 1005, s = (P mod 255) + 255·u with P = c·w mod 2^32 and u ∈ {0, 1, 2, 3}.
- **F6, the float stage.** `IDP.4A p, 0x01010101, 0x4B000000` yields the bit pattern of the fp32 number 2^23 + s. One `FADD` of −2^23 then gives fp32(s) exactly. So `I2F` can be traded for a full-rate `FADD`, which matters if the conversion pipe binds (§2.5).
- **F7, step 1's bound.** If c·w is formed as written, with 2 XMAD, a counting argument gives ≥ 5.63–6.19 instr/weight. That count left out per-tile overhead, so it is optimistic. Beating it needs a different way to obtain s.

### 4.2 What failed, and why (do not repeat without a new idea)
- **L1, fewer instructions but fatter lanes.** A variant with 32 windows per lane issued 25% fewer instructions and ran 40% slower. It needed 48 registers (40 warps/SM), or 40 with spills, and it had fewer warps per launch.
- **L2, gains on some K only.** A core that paired windows ran 1.14–1.16× faster per K in isolation, but only 1.03× on the real mix. K = 2.5 / 3 / 3.5 had no pairs, and a launch is as slow as its slowest blocks. **Score every K by its share.** K = 1.5 / 2 / 2.5 are 77% of the work.
- **L3.** A 65,536-entry table of v(w) in global memory: 15× slower.
- **L4.** Approximations, such as int8 rounding of v, byte-sum shortcuts and 2-byte codebooks: 0.6×–23× of a quantisation step in error. Rejected.
- **L5.** Decoding once for several tokens works (2.5× at 4 tokens), but our use is one token at a time.

## 5. The ask

### 5.1 Free choices
Everything below is yours to choose, per K:
- the bijection from windows to (lane, slot), and from jobs to warps;
- per-lane loop invariants, including lane-indexed tables held in registers and read with `SHFL`;
- how ring words are loaded: which lanes, what widths, and how much redundancy;
- tables in shared memory, filled once per block, within §2.4, with the fill cost reported;
- any exact transform of x staged per block, with its cost and memory reported;
- any exact arithmetic route to s(w_i), or to per-column integer partial sums;
- accumulation order (§1.4).

Directions worth a look. None of them is known to work.
- **Recurrences.** F4 as `ISCADD` plus a 2K-bit table lookup. For K ≤ 2 the table has ≤ 16 entries, so it fits one register per lane read with `SHFL`, or a conflict-free LDS. By §2.5, one lookup per weight costs 4 on the quarter-rate side, so lookups must be shared between weights. Also: a recurrence on s itself, or on (P mod 255, u).
- **Shared structure.** Of the 16-bit windows of one ring, of c = 0x83DCD12D, or of the carries in bytesum.
- **Split windows.** Split a window into parts and combine small tables, keeping within the bank and register limits.
- **Lane maps.** Ones that create the pairs of F2 for K = 1.5, or that make the extraction offsets cheaper for odd K.
- **Multi-value instructions.** `PRMT`, `IDP.2A` or `IDP.4A` tricks that serve two windows with one instruction.
- **Bounds.** A proof that the extraction plus product cannot go below some per-weight floor in this ISA, for a stated op set.

### 5.2 Exactness contract
Choose one of the following and prove it for every ring.
- **(a) s-form.** For each (n, k) of the tile, name a program point and a register that holds s(w) of that weight, or an exact affine image α·s + β with stated α and β. That register must feed exactly one product with x_k into an accumulator owned by column n.
- **(b) sum-form.** Your own contract, proved to give A[n] = Σ s·x in exact arithmetic, plus the fp32 error bound of §1.4.

### 5.3 Deliverables
1. **Design note.**
   - The idea and the mathematics behind it, with a proof sketch.
   - Per K: the lane map, C_K by the formula of §2.5 with its class counts n_q, and peak live registers for the whole loop (≤ 40, as an explicit allocation).
   - Shared memory used, and E.
2. **Programs.** Per K, the steady-state pass as straight-line text, one instruction per line, in the notation of §2.3 (`XMAD.PSL r3, r2.H1, 0x83dc, r3`). Add the prologue that sets up per-lane invariants and tables.
3. **Lean 4 + Mathlib**, with the toolchain and Mathlib commit pinned.
   - The reference of §1 and the ISA of §2.3, as models.
   - The programs as data, and an interpreter for them.
   - For each K, a theorem that the contract of §5.2 holds for every ring.
   - The sanity theorem of §2.3.
   - The cost (n_q and the live-register peak), computed from the program data and checked by `decide` or `rfl`.
   - Gates: `lake build` passes, with no `sorry`. `#print axioms` on the main theorems shows only propext, Classical.choice and Quot.sound, plus `Lean.ofReduceBool` if `native_decide` or `bv_decide` is used and declared.
4. **CPU simulator** (C, Rust or Python).
   - It runs the program text of item 2 directly.
   - It tests every K: the vectors of §1.5, all 2^16 window values in every offset class, and ≥ 10^6 random tiles, against an independent transcription of §1.
5. **Optional.** Lower bounds, or refutations of the directions above, in the model.

### 5.4 What counts
- Any exact scheme with **E < 8.81** under §2.5 and within §2.4 is useful. Below 7.5 is a clear win, and below 6.5 is a large one.
- A scheme that wins only on some K is still useful if E drops, but read L2 first.
- A proof that some route cannot beat a stated bound is also a result. It tells us where to stop.

## 6. After you hand over (our side, for context)
1. We re-run your Lean build and your simulator.
2. We port your programs into our kernel by whatever route works (CUDA, inline PTX, or a verified edit of the compiled binary).
3. We check:
   - an exhaustive microkernel against the reference;
   - bit-identical gold GEMV outputs on real weights, or for order changes, a numeric-noise check;
   - the end-to-end output hash;
   - an interleaved 16-pair A/B timing on the GTX 1060.
4. We report the measured times back to you.
