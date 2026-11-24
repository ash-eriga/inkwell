# Mathematical Reformulations of EXL3 `mul1` GEMV Decode on Pascal `sm_61`

## 1. Executive Summary & Headline Verdict

To saturate the GTX 1060's DRAM bandwidth (~204 GB/s) at the SAGE MoE average bitrate of 2.26 bits/weight (722.1 G weights/s), a batch-1 GEMV inner loop must cost at most **2.66 (~2.7) nominal full-rate issue slots/weight** (at 10 SMs × 1.5 GHz × 128 cores = 1920 G ops/s, or **2.08 slots/weight** at the measured 100 lane-ops/clk/SM = 1500 G ops/s).

Through exhaustive CPU verification (all 65,536 windows, all 4,063,232 incremental transitions for $k=1\dots5$, and all 7 EXL3 $K$ geometries) and exact `nvcc 12.6` / `cuobjdump -sass` inspection for `sm_61`, we establish two main results:

1. **Large real speedup (1.86×–2.61× fewer issue slots/weight than today's `sgemv_run`):**
   - **Today's baseline (`sgemv_run` `MAC=0`, `K=2.0`):** `10.597` pure decode+MAC slots/weight (`15.006` full loop incl. `LDG`/pointer/control) — notably **higher** than previously assumed because `ptxas 12.6` re-folds `sgemv_mul16` into a **3-XMAD** 32×32 multiply (`XMAD` + `XMAD.MRG` + `XMAD.PSL.CBCC`) rather than 2 XMADs, and emits 7 `BFE` + 1 `SHF` + 1 `SHR` per 8 weights.
   - **Best combined inner loop (stock `nvcc 12.6` PTX, `K=2.0`, `M=32`, C4):** **`5.699` pure decode+MAC slots/weight** (**`6.719` full loop**) — a **1.86× pure / 2.23× full-loop reduction** with `2.0e-5` max relative error on Hadamard-like `x` and `1.2e-5` on all-positive `x` (10× *more* accurate than baseline FP32 on nonzero-mean `x`). Across the SAGE expert $K$-mix (`M=64`), weighted average is **`5.908` pure slots/weight**.
   - **Best combined inner loop (native `sm_61` 2-XMAD SASS via cubin peephole, `K=2.0`, C4):** **`4.728` pure slots/weight** (**`5.748` full loop**; **`4.937` SAGE $K$-mix average**) — a **2.24× pure / 2.61× full-loop reduction**.
   - **Best bit-exact FP32 loop (C1c zero-centered magic `-8389118.f`):** **`6.728` pure slots/weight** (`5.757` native 2-XMAD), **63× more accurate** than baseline FP32 on nonzero-mean `x` (`1.8e-6` vs `1.2e-4`) at zero extra instruction cost over `MAC=1`.
   - **Best native-FP32 zero-`x`-quantization low-slot loop (C3b per-tile folded `0x3F800000 + 255*bs`):** **`5.791` pure slots/weight** (`4.820` native 2-XMAD), `1.47e-5` max relerr on Hadamard-like `x`.

2. **Hard architectural lower bound proves `<= 2.7` slots/weight is unreachable for lossless batch-1 `mul1` GEMV on `sm_61`:**
   - Computing $P = (h \cdot \text{0x83DCD12D}) \bmod 2^{32}$ on Pascal `sm_61` (no native 32-bit integer multiplier; widest full-rate multiplier is 16×16 `XMAD`) mathematically requires **$\ge 2$ `XMAD`s (`1.942` slots/weight)** in hardware SASS (**3 `XMAD`s = `2.913` slots/weight** in stock `ptxas 12.6`).
   - Accumulating $x \cdot \text{bytesum}(P)$ to $\sim 10^{-5}$ accuracy requires $\ge 16$-bit precision on $x$ (8-bit `IDP4A` has an irreducible `~3e-3` quantization floor even with fixed `sx` and per-16 scaling), which costs **$\ge 1.961$ slots/weight** (`1 IDP4A + 1 XMAD.S16.S16`, or `1 IDP4A + 1 FFMA = 1.990`, or `2 IDP2A = 2.000`).
   - **ALU lower bound alone (before extracting a single bit from the bitstream): `1.942 + 1.961 = 3.903` slots/weight** (native 2-XMAD) or **`4.874` slots/weight** (stock `ptxas`). Including optimal paired window extraction (`0.708–1.353`, SAGE avg `1.035`), the serial-issue floor is **`4.937` slots/weight** (SAGE avg) — reaching **42%–54% of DRAM bandwidth** (up from 14%–25% today), or up to **53%–68%** if the shifter pipe dual-issues fully in parallel with ALU.
   - Reaching $\le 2.7$ slots/weight at batch-1 on `sm_61` requires either a **lossy** codebook change (not lossless `mul1`) or **batching $B \ge 3$ vectors** (amortizing decode across $B$ FMAs: `(3.76 + B)/B = 2.25` slots/MAC at $B=3$).

---

## 2. Throughput Model & Key `ptxas 12.6` Codegen Finding

### Pascal `sm_61` Issue-Slot Weights
Normalized to 100 measured full-rate lane-ops/clock/SM:

| Opcode(s) | Measured tput (lane-ops/clk/SM) | Slot-equivalents / op |
|---|---:|---:|
| `FFMA`, `FADD`, `FADD32I`, `LOP`, `LOP32I`, `IADD`, `IADD3`, `IDP2A` | 100 | `1.000` |
| `IDP4A` (`IDP.4A.U8.U8`) | 101 | `0.990` |
| `XMAD` (incl. `.PSL`, `.S16.S16`) | 103 | `0.971` |
| `SHF`, `SHL`, `SHR`, `BFE`, `PRMT` (half-rate shifter pipe) | 53 | `1.887` |
| `I2F` (quarter-rate SFU/conversion pipe) | 28 | `3.571` |
| `LDG`, `LDS` (quarter-rate LD/ST pipe; reported separately in full-loop column) | 32 | `3.125` |

### Critical Discovery: `sgemv_mul16` Compiles to 3 `XMAD`s + `BFE` Today
Inspecting `cuobjdump -sass` on `k_base_k2` (using the exact `sgemv_mul16` from `crates/inkwell-cuda/src/kernels.cu:2437`) reveals that `nvcc 12.6` re-folds `h * 0xD12Du + ((h * 0x83DCu) << 16)` back into a 32×32 multiply `h * 0x83DCD12Du`, emitting **3 `XMAD`s** (`XMAD`, `XMAD.MRG`, `XMAD.PSL.CBCC`) plus `BFE.U32` per window (4.800 slots/weight for extract+mul alone).

Passing `C_mul1 = 0x83DCD12D` as a non-folded register/cbank parameter with PTX `mul.wide.u16` + `mad.lo.u32(..., 65536, t)` eliminates all `BFE`/`LOP`/`SHR` instructions and reads `U.H1` / `U.H0` directly:
```sass
XMAD     R_t, R_U.H1, c[0x0][C].H0, RZ ;   // 0.971 slots — reads .H1/.H0 directly, ZERO BFE/LOP!
XMAD     R_u, R_U.H1, c[0x0][C].H1, RZ ;   // 0.971 slots
XMAD.PSL R_p, R_u,    0x1,          R_t ;  // 0.971 slots — ZERO SHL!
```
We exhaustively tested 25+ PTX/C++ patterns (`mul.wide.u16`, `mad.wide.u16`, `mul24`/`mad24`, `uint16_t`, `(U & 0xFFFF0000) * 0x83DC + t`). While `sm_61` hardware natively supports the 2-instruction sequence `XMAD R_t, R_U.H1, 0xd12d, RZ; XMAD.PSL R_p, R_U.H1, 0x83dc, R_t;` (verified valid SASS opcodes emitted by `ptxas` inside `k7`), `ptxas 12.6` cannot emit it without either a redundant 3rd `XMAD` (`XMAD.PSL ..., 0x1`) or a dead low-half `XMAD` because PTX has no 16×32-multiply intrinsic. A trivial post-link cubin peephole replacing `XMAD u, h, ch, RZ; XMAD.PSL p, u, 0x1, t` → `XMAD.PSL p, h, ch, t` recovers that **0.971 slots/weight** on top of pure PTX.

---

## 3. Master Results Table (All Evaluated Tricks)

All SASS counts are measured directly from `opt/kbench/exl3_math/sass_kernels.o` (`sm_61`) via `opt/kbench/exl3_math/count_sass.py` at `K=2.0` unless noted. Relative errors are `max|err|/max|y|` vs FP64 `ref_gemv` on `K=2304, N=1152` (Dist A = Hadamard-like zero-mean `x` with 6× outliers; Dist B = adversarial all-positive `x ∈ [0.5, 1.5]`).

| ID | Trick / Formulation | Exact (`h=0..65535`)? | GEMV `max_rel` (Dist A / Dist B) | Core Opcodes / Weight (`cuobjdump -sass`) | Pure Decode+MAC Slots/w | Full Loop Slots/w | vs Today (`10.60` pure / `15.01` full) | Status |
|---|---|---:|---:|---|---:|---:|---:|---|
| **Base-0** | Today `sgemv_run` `MAC=0` (`I2F`, `M=8`, unrot) | Exact | `1.73e-6` / `1.16e-4` | `0.875 BFE, 0.125 SHF, 0.125 SHR*, 3 XMAD, 1 IDP4A, 1 I2F, 1 FFMA` | **10.597** | **15.006** | `1.00×` / `1.00×` | Baseline |
| **Base-1** | Today `sgemv_bench` `MAC=1` (`0x4B000000 - 2^23`) | Exact | `1.73e-6` / `1.16e-4` | `1.125 shift, 3 XMAD, 1 IDP4A, 1 FADD, 1 FFMA` | **8.026** | **12.560** | `1.32×` / `1.19×` | Baseline |
| **Base-2** | Today `sgemv_bench` `MAC=2` (`dp2a-16`, unmatched `sx`) | Approx | `8.13e-5` / `3.41e-5` | `1.125 shift, 3 XMAD, 2 IDP2A` | **7.036** | **11.444** | `1.51×` / `1.31×` | Baseline |
| **A1+A2** | Load-time ring rotation (0% bloat) + `M=32` paired `(D,D+16)` `.H1/.H0` (`K=2.0`) | **Exact** (256k/256k) | `0.0` (exact win) | `0.438 SHF` (0 `BFE`, 0 align `SHF`, 0 `LOP`) | `-1.298` extract | `-2.5+` (w/ LDG) | **Saves 1.30–2.5+ sl/w** | **WIN** |
| **A3** | `(D-1)` `SHF` + `IADD(U+U)` pairing for `K=1.5, 2.5, 3.0, 5.0` | **Exact** (all 7 `K`) | `0.0` (exact win) | `K=3.0`: `0.469 SHF + 0.469 IADD`; `K=1.5`: `0.531 SHF + 0.156 IADD` | `1.353` (`K=3`) / `1.041` (`K=1.5`) | — | **Saves 0.53–0.85 sl/w** | **WIN** |
| **B1** | Direct `.H1/.H0` `mul16` (`mul.wide.u16` + `mad.lo.u32(*65536)`) | **Exact** (65536/65536) | `0.0` | `3 XMAD` (`ptxas`) / `2 XMAD` (native SASS), `0 BFE/LOP` | `2.913` / `1.942` | — | **0 BFE/LOP; -0.97 native** | **WIN** |
| **C1c** | **Zero-centered exact magic** `dp4a(P,1s,0x4B000000) - 8389118.f` + `FFMA` | **Exact** (65536/65536) | **`1.38e-6` / `1.84e-6`** | `1 IDP4A, 1 FADD32I, 1 FFMA` | `2.990` (MAC) / **6.728** comb (`5.757` nat) | **8.037** | **1.58×** (1.84× nat); **63× lower err on Dist B** | **WIN (Exact FP32)** |
| **C3a** | **Max-scale folded magic** `dp4a.u32(P,0xFFFFFFFF,0x3F800000)` global fold | **Exact** identity | `1.07e-4` / `8.46e-3` | `1 IDP4A, 1 FFMA` (0 `FADD`, 0 `I2F`) | `1.990` (MAC) / **5.728** comb (`4.757` nat) | **6.943** | **1.85×** (2.23× nat) | **WIN (zero-mean `x` only)** |
| **C3b** | **Max-scale folded magic** `0x3F800000 + 255*bs` w/ per-tile cancel (`sxt[a]`) | **Exact** identity | **`1.47e-5`** / `1.96e-4` | `1 IDP4A, 1 FFMA, 0.0625 FADD` | `2.053` (MAC) / **5.791** comb (`4.820` nat) | **7.259** | **1.83×** (2.20× nat) | **WIN (Native FP32)** |
| **C4** | **Zero-centered `IDP4A(-510)` + `XMAD.S16.S16`** (`x16` pairs, `blk=16`) | **Exact** `bs-510` | **`2.02e-5` / `1.25e-5`** | `1 IDP4A, 1 XMAD.S16.S16` (0 `I2F`, 0 `FADD`, 2× less `x` smem) | `1.961` (MAC) / **5.699** comb (`4.728` nat) | **6.719** (`5.748` nat) | **1.86×** (`2.24×` nat) / **2.23×** (`2.61×` nat) full | **BEST OVERALL** |
| **C5-fix** | Fixed `dp2a-16` (`matched sx_q`, `blk=16`) | Approx | `2.02e-5` / `3.10e-5` | `2 IDP2A` (slower & larger `x` splat than C4) | `2.000` (MAC) / `5.738` comb | — | Dominated by C4 | OK (dominated) |
| **A4** | `XMAD`-based bit-shift extraction (`XMAD` + `XMAD.CHI`) | Exact (65536/65536) | `0.0` | `0.875 BFE, 0.75 SHL, 0.375 SHR, 4 XMAD, 1 IDP4A` (compiler failed `.CHI`) | **10.008** (`1.942` ideal/win) | **12.164** | **FAIL**: 2× worse than paired `SHF` (`0.825`) | **FAIL** |
| **A5** | Lossless load-time codebook/entropy re-encoding | — | — | Theoretical: Irwin-Hall $H(\text{bs}) \approx 9.24$ b/w; pre-shifted = 2× bloat | — | — | **FAIL**: violates ~2.26 b/w by 2×–4× | **FAIL** |
| **B2** | Incremental update $P_{i+1} = (P_i \ll k) + vC - u(C_L \ll 16)$ | Exact (4,063,232/4,063,232) | `0.0` | `1.625 BFE, 0.875 SHL, 0.375 SHR, 6.625 XMAD, 1 IDP4A` | **13.222** | **15.894** | **FAIL**: >2× slower + serial chain kills ILP | **FAIL** |
| **B3** | Byte-shared `P = (Q[A]<<8) + Q[B]` via 1 KB smem LUT | Exact (65536/65536) | `0.0` | `1.25 LDS (3.91 sl!), 1.5 SHR, 0.25 SHL, 1.5 ISCADD, 1.25 LOP32I` | **7.013** (excl. LDS) | **16.763** | **FAIL**: `LDS` quarter-rate + bank conflicts | **FAIL** |
| **B4** | 65,536-entry LUT (`uint16_t` 128 KB) in L2 / smem | Exact | `0.0` | Exceeds 48 KB smem/block & L1; 128 KB random `LDG` = 3.125 sl + L2 bound | — | — | **FAIL**: 1.44 TB/s L2 demand (4× over GP106 L2) | **FAIL** |
| **B5** | Carry-free / modular `bytesum` bypassing $P = h \cdot C$ | Impossible | — | Carries occur on ~75% of byte adds; $\text{bs}(A+B)$ needs full 32-bit add | — | — | **FAIL**: mathematically impossible | **FAIL** |
| **C2** | Naive folded magic `0x4B000000` (no `FADD`) | Exact identity | **`2.26e-2` / `1.98e+0`** | `1 IDP4A, 1 FFMA` | `1.990` (MAC) | — | **FAIL**: loses 15.8 bits of mantissa (198% err!) | **FAIL** |
| **C6** | `dp4a-8` (even fixed `matched sx_q`, `blk=16`) | Approx | **`4.51e-3` / `2.82e-3`** | `1 IDP4A` | `0.990` (MAC) | — | **FAIL**: 300× over `1e-5` error floor | **FAIL** |

*\*Note on `k_base_k2` `SHR`: 3 `SHR`s appear in the loop; 1 is the `D=0` window shift `hi >> 16` (counted in pure decode = `1.125` shifter ops/w) and 2 are 64-bit pointer sign-extensions (counted in full loop). For $K \ge 2.5$, `lo` requires a second alignment `SHF`, raising baseline extraction to `1.25` shifter ops/w (`2.358` slots/w).*

---

## 4. Detailed Analysis of Winning Reformulations

### 4.1 Load-Time Lossless Ring Rotation (A1) + Paired 32-Bit Extraction (A2) + `(D-1)` `IADD` Trick (A3)
In `crates/inkwell-cuda/src/kernels.cu:1580`, `tile_perm` maps ring positions $i = 8t + j$ to column $c = \lfloor t/4 \rfloor + (j \mathbin{\&} 4 ? 8 : 0)$. Crucially, **32 consecutive ring positions (`i = 32c .. 32c+31`) all belong to the exact same column pair `(c, c+8)`**. Today's `sgemv_run` splits those 32 windows across 4 lanes (`M=8` each) and then warp-reduces them.

1. **Lossless load-time ring rotation (0.00% memory overhead):** Rotate each tile's cyclic bitstream left by $b_0(0) = (K_A - 16) \bmod R$ bits once at model load time. Window $i=0$ now starts at bit 0 of word 0.
   - At **`M = 32` windows/lane** (8 lanes/tile, 4 tiles/warp, **zero warp-shuffle reduction**): bit stride per lane is $32K$, so **`o = 0` for every lane on all integer $K \in \{2,3,4,5\}$** (0 alignment `SHF`s!) and `o ∈ {0, 16}` on half-$K$ (`0.0625` align `SHF`/w).
   - At **`M = 64` windows/lane** (4 lanes/tile, 8 tiles/warp): bit stride is $64K \equiv 0 \pmod{32}$, so **`o = 0` for every lane on ALL 7 $K \in \{1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0\}$**!
2. **Paired `(D, D+16)` extraction via `XMAD` `.H1`/`.H0` selectors:** One `SHF.L.W` extracts `U = bits[D .. D+32)`; `U.H1` is window $D$ and `U.H0` is window $D+16$ with **zero `BFE`, zero `LOP`, zero `SHR`**. Aligned words ($D \equiv 0 \pmod{16}$) use `W[k].H1` and `W[k].H0` directly (**zero shifts**).
3. **`(D-1)` `SHF` + `IADD(U+U)` trick for odd-stride $K \in \{1.5, 2.5, 3.0, 5.0\}$:** When $D+16$ is absent from the window set (e.g. $K=3.0$ where $D \equiv 0 \pmod 3$, or $K=2.5, 5.0$ where $15 \equiv 0 \pmod 5$), $D+15$ *is* in the set. Extracting `U = shf_l(D - 1)` yields `U.H0 = D+15` for free, and one full-rate `IADD` (`V = U + U`, `1.000` slot vs `1.887` `SHF`) yields `V.H1 = D`! For $K=1.5$, a single `SHF(d)` + `IADD(U+U)` covers **3 windows** `{d, d+1, d+16}` in `2.887 / 3 = 0.962` slots/window.

| $K$ | `M=32` Aligned / `SHF` / `IADD` | `M=32` Extract Slots/w (excl/incl align) | `M=64` Aligned / `SHF` / `IADD` | `M=64` Extract Slots/w (`o=0` all lanes) |
|---:|---:|---:|---:|---:|
| 1.5 | 2 / 15 / 5 | `1.041` / `1.159` | 4 / 30 / 5 | **`0.963`** |
| 2.0 | 4 / 14 / 0 | **`0.825`** / **`0.825`** | 8 / 28 / 0 | **`0.825`** |
| 2.5 | 2 / 16 / 14 | `1.381` / `1.499` | 4 / 30 / 30 | **`1.353`** |
| 3.0 | 2 / 15 / 15 | **`1.353`** / **`1.353`** | 4 / 30 / 30 | **`1.353`** |
| 3.5 | 2 / 30 / 0 | `1.769` / `1.887` | 4 / 58 / 2 | **`1.741`** |
| 4.0 | 8 / 12 / 0 | **`0.708`** / **`0.708`** | 16 / 24 / 0 | **`0.708`** |
| 5.0 | 2 / 15 / 15 | **`1.353`** / **`1.353`** | 4 / 30 / 30 | **`1.353`** |
| **SAGE K-mix avg** | — | `1.062` / `1.118` | — | **`1.035`** (vs `2.12–2.36` today) |

### 4.2 Eliminating `I2F` (3.571 → 0.000 slots/weight)
1. **C4 — Zero-Centered `IDP4A(-510)` + `XMAD.S16.S16` (1.961 slots/w, BEST OVERALL):**
   - `bs_c = dp4a.u32.u32(P, 0x01010101u, -510)` produces exact `bs - 510 ∈ [-510, +495]` in 1 `IDP4A` (`0.990` slots).
   - `acc_i = x16_pair.H{0|1} * bs_c + acc_i` compiles to **1 single `XMAD.S16.S16`** (`0.971` slots), reading packed int16 pairs directly (halving `x` smem traffic and registers).
   - Because `bs_c` is zero-centered, the hoisted offset constant shrinks by **1900×**: `KOFF + 510 * KINV = -0.0018158` (vs `KOFF = -3.453125`), eliminating the bias-amplification error that degraded `sgemv_bench`'s `MAC=2`. Max product `|32767 * 510| = 16.7M` allows up to 128 terms per int32 flush with zero overflow risk.
   - Accuracy: **`2.02e-5`** (Dist A, `blk=16`) and **`1.25e-5`** (Dist B — **10× better than baseline FP32 `1.16e-4`**!).
2. **C3b — Max-Scale Folded Magic `0x3F800000 + 255*bs` with Per-Tile Cancel (2.053 slots/w, BEST NATIVE FP32):**
   - Naive folding of `0x4B000000 + bs` (C2) fails catastrophically (`2.26e-2` Dist A, `198%` error Dist B) because `8388608 / 147.8 = 56,760` wastes 15.8 bits of mantissa.
   - Using `dp4a.u32.u32(P, 0xFFFFFFFFu, 0x3F800000u)` scales `bs` by **255** at **zero extra instruction cost** (`255 * 1005 = 256,275 < 2^23`), representing exact `1.0 + (255/2^23) * bs` and recovering **8 full bits (255× lower error)**!
   - Initializing per-tile accumulators `t0 = t1 = sxt[a] = -beta * sum_{r=0..15} x_r` (precomputed once in prologue, 1 uniform scalar/tile) and flushing `a0 += t0; a1 += t1` once per 32 weights (`+0.0625` FADD/w) prevents cross-tile drift, achieving **`1.47e-5`** on Hadamard-like `x` with zero `x` quantization.
3. **C1c — Zero-Centered Exact Magic `0x4B000000 - 8389118.f` (2.990 slots/w, BEST BIT-EXACT FP32):**
   - Subtracting `8389118.f = 2^23 + 510` instead of `8388608.f = 2^23` costs the exact same 1 `FADD32I` (`1.000` slot) while zero-centering `bs` to `[-510, +495]`, cutting Distribution B error from `1.16e-4` down to **`1.84e-6`** (63× more accurate than baseline FP32!).

---

## 5. Summary of Negative Results (and Why Each Fails)

- **B2 — Incremental update $P_{i+1} = (P_i \ll k) + v_i C - u_i(C_L \ll 16) \bmod 2^{32}$:** Mathematically 100% exact (verified across all 4,063,232 transitions for $k=1\dots5$), but extracting both outgoing $u_i$ and incoming $v_i$, shifting $P_i$, and multiplying $v_i C - u_i(C_L \ll 16)$ compiles to **6.625 `XMAD` + 2.875 shifter ops = `13.222` core slots/w** (`15.894` full loop) AND creates a serial dependency chain destroying ILP.
- **B3 — Byte-shared decomposition $P = (Q[A] \ll 8) + Q[B]$:** Exact for all 65,536 $h$, but for $K \le 4$ every residue class mod 8 still has 1 unique byte per window (no reduction in `Q` evaluations), and a 1 KB shared-memory table requires `1.25 LDS/w` (`3.91` quarter-rate slots + bank conflicts) plus shift+add = **`16.763` full-loop slots/w**.
- **B4 — 65,536-entry LUT (`uint16_t` 128 KB):** Exceeds 48 KB smem/block and 48 KB L1/SM; random `LDG` costs 3.125 slots/w and at 722 Gw/s would demand 1.44 TB/s of L2 bandwidth (~4× GP106 L2 bandwidth).
- **B5 — Carry-free `bytesum(h * C)`:** Since $\text{bs}(A+B) = \text{bs}(A) + \text{bs}(B) - 255(c_0+c_1+c_2) - 256 c_3$ and carries occur on ~75% of byte sums, exact carries require computing the full 32-bit product $P$.
- **A4 — `XMAD`-based bit-shift extraction:** Exact (`PASS=65536/65536`), but requires 2 `XMAD`s per single window (`1.942` ideal slots/w; `ptxas` didn't emit `.CHI` → `10.008` core slots/w), >2× slower than paired `SHF` (`0.825` slots/w).
- **A5 — Lossless load-time codebook/entropy re-encoding:** Marginal entropy of `bs` (Irwin-Hall sum of 4 uniform bytes) is ~9.24 bits/weight; arbitrary $K=2$ tiles span $2^{512}$ distinct weight matrices, so scalar/block lossless re-encoding bloats memory ≥4× and pre-shifted streams bloat 2×. Only zero-overhead bitstream layout transforms (ring rotation + vector coalescing) preserve ~2.26 bits/weight.
- **C2 — Naive folded `0x4B000000`:** Loses 15.8 bits of mantissa (`2.26e-2` Dist A, `198%` Dist B).
- **C6 — `dp4a-8` (`int8 x`):** Fixing the `sx` mismatch bug + per-16 scale cuts error ~3–4× (from `1.15e-2` to `4.51e-3` Dist A, `2.82e-3` Dist B), still ~300× above `1e-5`.

---

## 6. Best Combined Inner Loop & Implied Throughput Ceiling

**Best Combined Inner Loop:** **A1** (load-time ring rotation, 0% memory bloat) + **A2/A3** (`M=32` or `M=64` paired `(D, D+16)` / `(D-1)+IADD` extraction) + **B1** (direct `.H1/.H0` `mul16`) + **C4** (zero-centered `IDP4A(-510)` + `XMAD.S16.S16`, `blk=16` int16 `x`). *(If native FP32 `x` without quantization is required, swap C4 → **C3b** for `+0.09` slots/w, or **C1c** for `+1.03` slots/w bit-exact FP32.)*

**Implied Ceiling at 10 SMs × 1.5 GHz** (DRAM ceiling @ 204 GB/s, 2.26 b/w = **722.1 G weights/s**):

| Configuration | Slots / weight | Ceiling @ 100 ops/clk/SM (1500 G/s) | Ceiling @ 128 cores/SM (1920 G/s) | % of 722.1 Gw/s DRAM |
|---|---:|---:|---:|---:|
| Today `k_base_k2<0>` (`MAC=0`, `K=2.0`, pure / full) | `10.597` / `15.006` | `141.6` / `100.0` Gw/s | `181.2` / `127.9` Gw/s | 19.6–25.1% / 13.8–17.7% |
| Today `k_base_k2<2>` (`MAC=2`, `K=2.0`, pure) | `7.035` | `213.2` Gw/s | `272.9` Gw/s | 29.5–37.8% |
| **Best stock-`ptxas` (`K=2.0`, C4, pure / full)** | **`5.699` / `6.719`** | **`263.2` / `223.2` Gw/s** | **`336.9` / `285.8` Gw/s** | **36.4–46.7% / 30.9–39.6%** |
| **Best stock-`ptxas` (SAGE K-mix avg `M=64`, C4, pure)** | **`5.908`** | **`253.9` Gw/s** | **`325.0` Gw/s** | **35.2–45.0%** |
| **Best native 2-XMAD (`K=2.0`, C4, pure / full)** | **`4.728` / `5.748`** | **`317.2` / `261.0` Gw/s** | **`406.1` / `334.0` Gw/s** | **43.9–56.2% / 36.1–46.3%** |
| **Best native 2-XMAD (SAGE K-mix avg `M=64`, C4, pure)** | **`4.937`** | **`303.8` Gw/s** | **`388.9` Gw/s** | **42.1–53.8%** |
| Theoretical ALU-only bound (native 2-XMAD, SHF fully hidden) | `3.903` | `384.3` Gw/s | `492.0` Gw/s | 53.2–68.1% |

---

## 7. Reproduction Commands (CPU & Static SASS Only — No GPU Execution)

```bash
# 1. Exhaustive CPU exactness & GEMV accuracy tests
g++ -O3 -std=c++17 opt/kbench/exl3_math/test_cpu.cpp -o /tmp/exl3_test_cpu -lm && /tmp/exl3_test_cpu

# 2. Compile sm_61 device snippets & count SASS opcodes
~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 --use_fast_math \
  -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -c opt/kbench/exl3_math/sass_kernels.cu -o /tmp/exl3_sass.o && \
~/lfm-gtx1060-bench/cuda-12.6/bin/cuobjdump -sass /tmp/exl3_sass.o > /tmp/exl3_sass.txt && \
python3 opt/kbench/exl3_math/count_sass.py /tmp/exl3_sass.txt
```
