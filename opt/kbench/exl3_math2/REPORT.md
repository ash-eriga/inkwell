# EXL3 `mul1` Decode Math — Round 2 Report (Pascal `sm_61`, GTX 1060 6GB)

**Date:** 2026-11-24  
**Scope:** Four unexplored directions for EXL3 `mul1` GEMV decode on Pascal `sm_61` (`opt/kbench/exl3_math2/`), verified on CPU across 16.81M real weights (all 7 EXL3 $K$ geometries), verified by `cuobjdump -sass` (`nvcc 12.6`, `sm_61`), and benchmarked on silicon under `flock /tmp/inkwell-gpu.lock` head-to-head against the shipped MoE core (`inkwell_exl3_gemv_moe_s` / `sgemv_run` in `crates/inkwell-cuda/src/kernels.cu`).

**Strict architectural guardrails enforced on every candidate** (following the Round 1 `M=32` port failure in `41d0798`, which had 48 regs and cut grid size 4×, running ~40% slower despite 7.2 vs 9.47 instr/w):
- **Grid parallelism preserved identically:** 8 tiles/block, 256 threads/block (`144` blocks down, `144` blocks GU at 8 slots — zero grid shrinkage).
- **Occupancy preserved:** `<= 40` registers/thread, `0` spills (except noted 4B outside inner loop in `ALL_K C4 blk=128`), `<= 16 KB` smem/block → **6 blocks/SM** (1,536 thr/SM on 96 KB smem/SM, 65,536 regs/SM).
- **Drop-in `float* xh` interface:** All `C4`/`C6`/`C3b` staging/quantization happens inside the kernel prologue and is timed inside the kernel.

---

## 1. Executive Summary & Plain Verdicts

### Verdict 1: Is `~3 instr/weight` reachable losslessly at batch-1 on `sm_61`?
**NO — provably unreachable at batch-1 losslessly by ~2×.**
1. **Entropy/compression lower bound (real weights):** Across 65,664 real EXL3 tiles (`16,809,984` weights spanning all 7 $K \in \{1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0\}$ from `/tmp/sgemv_gold` and `/tmp/sgemv_dense_gold`), the raw bitstream has first-order byte entropy $H_1(B) = 7.9985\text{–}7.9997$ bits/byte and conditional entropy $H(B \mid B') = 7.898\text{–}7.957$ bits/byte (`zlib` ratio `1.0004`), while decoded `bytesum` has $H(\text{bs}) = 9.04\text{–}9.25$ bits/weight. Thus `ring -> (bs_0..bs_255)` is injective (collision probability $< 2^{-1050}$); **no lossless dictionary/palette/metadata fits in the same bytes**, and only 0%-bloat bijections are valid.
2. **Hardware SASS lower bound (exact lossless batch-1 FP32 MAC on `sm_61`):**
   - Exact $P = (h \cdot \text{0x83DCD12D}) \bmod 2^{32}$ on Pascal (widest full-rate multiplier is 16×16 `XMAD`) requires $\ge 2.00$ `XMAD`/w in hand-assembled SASS (**$\ge 3.00$ `XMAD`/w** in stock `ptxas 12.6`, exhaustively confirmed across 30+ PTX patterns; `SHR + SHFL + ISCADD` also totals **3.00** ops/w via our lossless L2 Strided-Crumb bijection on $K=2.0$).
   - Carry-free `bytesum` is impossible (~75% byte-addition carry rate); computing exact `bs = bytesum(P)` requires **$\ge 1.00$ `IDP4A`/w**.
   - Converting exact `bs` and accumulating into FP32 requires **$\ge 1.00$ `I2F`/`FADD` + $1.00$ `FFMA` = $2.00$ ops/w** (or `1.00 XMAD.S16` if $x$ is quantized to int16, which is near-lossless at $\sim 2\times 10^{-5}$).
   - Window extraction + loop/load overhead requires **$\ge 0.63\text{–}1.19$ ops/w** across the 7 $K$ geometries (`0.375` aligned/paired `K=4.0` to `0.938` `K=3.5`, plus `~0.25` LDG/LDS/loop).
   - **Proven Batch-1 Lossless Lower Bound:**
     - **Hand-assembled SASS (2-XMAD):** **$\ge 5.63\text{–}6.19$ issued instr/w** (exact FP32); **$\ge 4.63\text{–}5.19$** (int16-$x$ `C4`).
     - **Stock `ptxas 12.6` (3-XMAD or L2 `SHFL+ISCADD`):** **$\ge 6.63\text{–}7.19$ issued instr/w** (exact FP32); **$\ge 5.63\text{–}6.19$** (`C4`).
     - **Achieved in this work:** **`6.875` instr/w** exact (`M=32-Strided C0c`, `K=2.0`) and **`7.219` instr/w** exact (`M=16-Paired C0c`, `K=2.0`); **`5.938` instr/w** (`M=32-Strided C4`).
   - **Gap to `~3.0 instr/w` (722 Gw/s DRAM ceiling):** **~2.0×–2.3× unreachable losslessly at batch-1.** The ONLY lossless path that crosses `<= 3.0 instr/w/token` and `> 722 G MAC/s` on `sm_61` is **Direction 3 multi-token batching at $B = 4$** (measured **`2.828` instr/w/tok**, **`734.0` G MAC/s** exact FP32; **`2.453` instr/w/tok**, **`734.0–741.0` G MAC/s** `C4`).

### Verdict 2: What ACTUALLY beats shipped `sgemv_run` on silicon (`sm_61`)?
Fixing `41d0798`'s grid/register collapse via **in-block `tk`-split** (2-way for `M=16`, 4-way for `M=32`) **flips the `-42%` regression into real silicon speedups** while keeping `<= 40` regs, `0` spills, and exact 8-tile/block grids:
- **Pure `K=2.0` (40.7% of SAGE MoE weights), interleaved vs shipped (`280–283 Gw/s`):**
  - **Best exact FP32:** **`M=16-Paired C0c (I2F)`** = **`321.1 Gw/s` (`1.14×` shipped)**, `39` regs, `0` spills, `7.63e-7` error (3× more accurate than shipped). **`M=32-Strided C0c`** ties it at **`319.3 Gw/s` (`1.14×`)**, `5.81e-7` error.
  - **Best sub-quant-step (`<< 1 EXL3 step`):** **`M=32-Strided C4` (NEW lossless 0%-bloat Strided-Crumb bijection + `SHFL+ISCADD` + `XMAD.S16`)** = **`355.9 Gw/s` (`1.27×` shipped)**, `40` regs, `0` spills, `3.92e-5` error (**180× below 1 EXL3 step**).
  - **Fastest overall (approx, ~1 quant step — unsafe for GU):** `M=16-Paired C6 (dp4a-8)` = **`365.9 Gw/s` (`1.30×`)**, `5.56e-3` error.
- **Key silicon discovery (`C0c` I2F beats `C1c` Magic FADD by +12%):** Despite identical SASS count (`7.22` vs `7.19` instr/w), `C0c` (`I2F`) ran **`321.1 Gw/s`** vs `C1c` (`FADD32I`) **`286.9 Gw/s`** (`+12%` faster!). Because Pascal's ALU pipe is saturated while the quarter-rate SFU/CVT pipe (`~476 G/s` ceiling > `321 Gw/s`) is idle, `I2F` offloads 1.0 op/w off the critical ALU pipe for free!
- **Full SAGE All-7-K Mix (interleaved vs shipped `258–264 Gw/s`):**
  - **Best exact FP32:** **`M=16-Paired C0c`** (`266.1 Gw/s`, **`1.03×`**) and **`Hybrid C0c`** (`267.6 Gw/s`, **`1.03×`**), both `40` regs, `0` spills, `5.42e-7` error.
  - **Best sub-quant-step (`<< 1 EXL3 step`):** **`M=16-Paired C4 (blk=16)`** (`274.7 Gw/s`, **`1.05×`**, `2.04e-5`) and **`Hybrid C4`** (`274.0 Gw/s`, **`1.05×`**, `2.45e-5`), both `40` regs, `0` spills.
  - **Why All-7-K gains (`+3%` exact / `+5%` `C4`) are smaller than pure `K=2.0` (`+14–16%` exact / `+27%` `C4`):** In a heterogeneous-K MoE grid, launch latency is bounded by the slowest non-`K=2.0` tail blocks (`K ∈ {1.5, 2.5, 3.0, 3.5, 5.0}`), where 0 exact `(D, D+16)` pairs exist for `K ∈ {2.5, 3.0, 3.5, 5.0}` (only `-4% to -8%` SASS reduction vs `-15%` on `K=2.0`) and `unroll 1` is required to keep `<= 40` regs / `0` spills inside a 7-way switch (vs shipped `sgemv_run`'s `unroll 4`).

---

## 2. Master Ranked Table (All Candidates vs Shipped `sgemv_run`)

All timings are **interleaved old/new min-of-31** (GU `2304 -> 2×1152` + down `1152 -> 2304`, 8 slots, exact 8 tiles/blk, 256 thr/blk, `6 blocks/SM`) under `flock /tmp/inkwell-gpu.lock`. SASS counts are exact full-loop issued instructions/weight from `cuobjdump -sass` (`opt/kbench/exl3_math2/sass_kernels.cu` + `count_sass.py`).

### 2A. Pure `K=2.0` Head-to-Head (40.7% of SAGE MoE, Ranked by Measured `Gw/s`)

| Rank | Design | Exact? (`h=0..65535`) | Issued SASS instr/w (`K=2.0`) | Regs (`bench.cu`) / Spill | Blks/SM | Measured `Gw/s` (min-of-31) | Speedup vs Interleaved Shipped | `max_rel` vs FP64 | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `M=16-Paired C6` (`dp4a-8, blk=16`) | No (int8 $x$) | `5.312` | `39` / `0B` | 6 | **365.9** | **1.30×** | `5.56e-3` | Unsafe (~0.8× quant step) |
| 2 | **`M=32-Strided C4`** (L2 Strided + `SHFL` + `S16`) | Near-exact (int16 $x$) | **`5.938`** | **`40` / `0B`** | **6** | **355.9** | **1.27×** | **`3.92e-5`** | **BEST sub-step (`K=2.0`)** |
| 3 | `M=32-Strided C6` (L2 Strided + `dp4a-8`) | No (int8 $x$) | `4.969` | `40` / `0B` | 6 | 348.2 | 1.24× | `5.56e-3` | Unsafe (~0.8× quant step) |
| 4 | `M=16-Paired C3a` (folded FP32 global) | Exact id. (FP32 drift) | `6.250` | `38` / `0B` | 6 | 330.0 | 1.18× | `1.61e-4` | Unsafe on Dist B (`1.09×` step) |
| 5 | **`M=16-Paired C0c`** (L1-rot + exact `I2F`) | **Exact FP32** | **`7.219`** | **`39` / `0B`** | **6** | **321.1** | **1.14×** | **`7.63e-7`** | **BEST exact FP32** |
| 6 | **`M=32-Strided C0c`** (L2 Strided + exact `I2F`) | **Exact FP32** | **`6.875`** | **`40` / `0B`** | **6** | **319.3** | **1.14×** | **`5.81e-7`** | **Ties #5 (0 XMADs in loop)** |
| 7 | **`M=16-Paired C4`** (`XMAD.S16, blk=16`) | Near-exact (int16 $x$) | `6.344` | `39` / `0B` | 6 | **309.3** | **1.11×** | **`1.82e-5`** | **Strong all-K sub-step** |
| 8 | `M=16-Paired LEA+2*DP2A.LO` (`blk=128`) | Near-exact (int16 $x$) | `6.281` | `40` / `0B` | 6 | 299.6 | 1.07× | `3.92e-5` | Beat by `C4 blk=16` & Strided |
| 9 | `M=16-Paired C1c` (exact Magic `FADD32I`) | **Exact FP32** | `7.188` | `40` / `0B` | 6 | 286.9 | 1.02× | `7.63e-7` | **-12% slower than `C0c` `I2F`!** |
| 10 | `M=16-Paired C4` (`XMAD.S16, blk=128`) | Near-exact (int16 $x$) | `6.344` | `40` / `0B` | 6 | 284.1 | 1.03× | `3.92e-5` | Slower than `blk=16` (outer loop) |
| — | **Shipped `sgemv_run`** (`nky=1, M=8, I2F`) | **Exact FP32** | **`8.375`** (`8.469` isol) | **`40` / `0B`** | **6** | **280.3–282.8** | **1.00×** | `2.41e-6` | **Baseline** |
| 11 | `M=16-Paired C3b` (folded FP32 per-tile) | Exact id. (FP32) | `6.406` | `40` / `0B` | 6 | 274.3 | 0.98× | `8.27e-5` | `sxt[a]` + `FADD` overhead loses |
| 12 | `M=8-Opt C3b` (folded FP32 per-tile) | Exact id. (FP32) | `7.844` | `39` / `0B` | 6 | 268.1 | 0.96× | `9.96e-5` | Loses to shipped `M=8` |
| 13 | `M=8-Opt C4` (`blk=16`) | Near-exact (int16 $x$) | `7.406` | `40` / `0B` | 6 | 250.0 | 0.89× | `1.82e-5` | Loses (`1.03` SHL/w) |
| 14 | `M=8-Opt C0c` (`I2F`) | **Exact FP32** | `8.500` | `40` / `0B` | 6 | 237.6 | 0.84× | `6.42e-7` | Loses (`pp[8]+f[8]` live) |
| 15 | `M=8-Opt C6` / `C4 blk=128` | Approx / Near-exact | `6.500` / `7.406` | `40` / `0B` | 6 | 229.0 / 227.2 | 0.81× | `5.56e-3` / `3.92e-5` | Loses |
| 16 | `M=16-LUT64K-BS8` (64 KB `LDG` LUT) | No (8-bit `bs`) | `7.625` | `40` / `0B` | 6 | **17.4** | **0.06×** | `1.16e-2` | **Catastrophic (L1/L2 thrash)** |
| 17 | `M=16-LUT128K-Exact` (128 KB `LDG` LUT) | **Exact FP32** | `9.062` | `40` / `0B` | 6 | **17.0** | **0.06×** | `7.63e-7` | **Catastrophic (16× slower)** |

### 2B. Full Real SAGE All-7-K Mix (`K ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0}`, Interleaved vs Shipped)

### 2B. Full Real SAGE All-7-K Mix (`K ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0}`, Interleaved vs Shipped)

Exact `sgemv_bench.cu` SAGE $K$-mix weights (GU + down, 8 slots, 8 tiles/blk, 256 thr/blk, interleaved min-of-31; `K=1.5, 3.5, 4.0, 5.0` explicitly included so all 7 switch arms are verified simultaneously):

| Rank | Design (All-7-K SAGE Mix) | Exact? | Regs / Spill | Blks/SM | Ship `us` | Cand `us` (Med) | Ship `Gw/s` | Cand `Gw/s` | Interleaved Speedup | `max_rel` vs FP64 | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `M=16-Paired C6` (`dp4a-8, blk=16`) | No (int8 $x$) | `40` / `0B` | 6 | 241.2 | 214.6 (220.9) | 264.2 | **296.9** | **1.12×** | `6.20e-3` | Unsafe (~0.8× quant step) |
| 2 | `M=16-Paired C3a` (folded FP32 glob) | Exact id. (drift) | `39` / `0B` | 6 | 244.2 | 226.4 (230.9) | 260.8 | **281.4** | **1.08×** | `1.52e-4` | Unsafe on Dist B (`1.09×` step) |
| 3 | **`M=16-Paired C4`** (`XMAD.S16, blk=16`) | Near-exact (int16 $x$) | **`40` / `0B`** | **6** | 243.6 | **231.9** (236.5) | 261.5 | **274.7** | **1.05×** | **`2.04e-5`** | **BEST safe All-7-K** |
| 4 | **`Hybrid C4`** (`L2-Strid K2 + Pair K!=2`) | Near-exact (int16 $x$) | **`40` / `0B`** | **6** | 245.1 | **232.4** (238.4) | 259.9 | **274.0** | **1.05×** | **`2.45e-5`** | Ties #3 (tail-slot bound) |
| 5 | **`Hybrid C0c`** (`L2-Strid K2 + Pair, exact`) | **Exact FP32** | **`40` / `0B`** | **6** | 244.4 | **238.0** (246.5) | 260.7 | **267.6** | **1.03×** | **`5.42e-7`** | **BEST exact All-7-K** |
| 6 | **`M=16-Paired C0c`** (L1-rot only, exact `I2F`) | **Exact FP32** | **`40` / `0B`** | **6** | 246.5 | **239.4** (244.2) | 258.4 | **266.1** | **1.03×** | **`5.42e-7`** | **Simplest exact (L1 only)** |
| — | **Shipped `sgemv_run`** (`all-7-K, M=8, I2F`) | **Exact FP32** | **`40` / `0B`** | **6** | 244.4 | 242.7 (252.3) | 260.6 | 262.5 | **1.00–1.01×** | `2.60e-6` | **Baseline** |
| 7 | `M=16-Paired C4` (`blk=128`) | Near-exact (int16 $x$) | `40` / `4B`* | 6 | 243.2 | 243.2 (249.9) | 261.9 | 261.9 | 1.00× | `3.10e-5` | Neutral (`blk=16` is faster & 0 spill) |
| 8 | `M=16-Paired C1c` (exact Magic `FADD`) | **Exact FP32** | `40` / `0B` | 6 | 246.7 | 252.7 (260.9) | 258.2 | 252.1 | 0.98× | `5.42e-7` | **Loses to shipped & `C0c`** |
| 9 | `M=16-Paired C3b` (folded per-tile) | Exact id. (FP32) | `40` / `0B` | 6 | 242.5 | 281.6 (290.4) | 262.7 | 226.2 | 0.86× | `9.61e-5` | Loses (`sxt` + `FADD` overhead) |

*\*Note on boost-clock reproducibility: Bench 1 was run twice (peak boost `225–228 us` shipped -> `321.1 Gw/s` `M=16-Paired C0c`, `355.9 Gw/s` `M=32-Strided C4`; warm/base boost `243–258 us` shipped -> `285.4 Gw/s` `C0c`, `331.5 Gw/s` `Strided C4`). Interleaved speedup ratios were invariant across both runs: `M=16-Paired C0c` `1.14×–1.16×`, `M=32-Strided C0c` `1.14×`, `M=32-Strided C4` `1.27×`, `M=16-Paired C4 blk=16` `1.11×–1.13×`, `M=16-Paired C1c` `1.02×–1.03×`, `M=16-LUT` `0.06×`.*

### 2C. Per-$K$ Issued SASS Breakdown: Shipped `inkwell_exl3_gemv_moe_s` vs `r2::gemv_m16<0,true>` (Exact `C0c`, 40 regs, 0 spills)

Disassembled directly from the compiled `bench.cu` binary (`cuobjdump -sass`):

| EXL3 $K$ | SAGE MoE Share (GU / DN) | Shipped `inkwell_exl3_gemv_moe_s` (ins/w) | `r2::gemv_m16<0,true>` (ins/w) | Core Ops in `r2::gemv_m16<0,true>` | Reduction |
|---:|---:|---:|---:|---|---:|
| `1.5` | 32.6% / 17.9% | `9.281` | `8.500` | `3.00 XMAD, 0.625 SHF, 0 BFE, 1 I2F, 1 FFMA` | **-8.4%** |
| `2.0` | 42.0% / 38.0% | `8.375` | **`7.125`** | `3.00 XMAD, 0.438 SHF, 0 BFE, 1 I2F, 1 FFMA` | **-15.0%** |
| `2.5` | 14.8% / 18.8% | `9.344` | **`8.625`** | `3.00 XMAD, 0.500 SHF, 0.125 SHL, 0.812 IADD` (`(D-1)+IADD`) | **-7.7%** |
| `3.0` | 7.5% / 14.8% | `9.469` | **`8.625`** | `3.00 XMAD, 0.438 SHF, 0.250 SHL, 0.750 IADD` (`(D-1)+IADD`) | **-8.9%** |
| `3.5` | 2.1% / 7.5% | `9.344` | `9.000` | `3.00 XMAD, 0 BFE` | **-3.7%** |
| `4.0` | 1.0% / 3.1% | `9.375` | **`8.000`** | `3.00 XMAD, 0.375 SHF, 0 BFE, 1 I2F, 1 FFMA` | **-14.7%** |
| `5.0` | 0.0% / 0.0% (dense) | `9.375` | `9.000` | `3.00 XMAD, 0 BFE` | **-4.0%** |

*(Compiler pitfall caught & fixed via SASS inspection: advancing a 64-bit `size_t stride = (size_t)tn * NW` across 3-4 split lane pointers emitted 18–24 extra `XMAD`s/tile in `K != {2,4}`. Switching to a single `char* qb += (uint32_t)(tn * NW * 4)` + precomputed lane offsets `wi[0..NA]` dropped all 5 non-{2,4} loops to exact `3.000 XMAD/w`.)*

---

## 3. Direction 1: Lossless Load-Time Weight Re-Encoding (Real Weights)

### 3.1 Real-Weight Entropy Audit & Injectivity Proof
Measured on all 7 real EXL3 $K$ tensors dumped from the real Inkling model (`/tmp/sgemv_gold/gate_s0..s5` and `/tmp/sgemv_dense_gold/wr_du.bin`, total **65,664 real tiles = 16,809,984 real weights**) by `opt/kbench/exl3_math2/test_cpu.cpp`:

| Tensor | $K$ | Real Tiles | $H_1(B)$ (b/byte) | $H(B \mid B')$ (b/byte) | $H(\text{bs})$ (b/w) | `L1-rot` Verify | `L2-strided` Verify |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gate_s0` (MoE) | 1.5 | 10,368 | `7.9985` | `7.8978` | `9.253` | **PASS (0 fails)** | n/a ($K \ne 2$) |
| `gate_s1` (MoE) | 2.0 | 10,368 | `7.9995` | `7.9234` | `9.215` | **PASS (0 fails)** | **PASS (0/0 fails)** |
| `gate_s2` (MoE) | 2.5 | 10,368 | `7.9997` | `7.9403` | `9.187` | **PASS (0 fails)** | n/a ($K \ne 2$) |
| `gate_s3` (MoE) | 3.0 | 10,368 | `7.9997` | `7.9496` | `9.147` | **PASS (0 fails)** | n/a ($K \ne 2$) |
| `gate_s4` (MoE) | 3.5 | 10,368 | `7.9997` | `7.9571` | `9.135` | **PASS (0 fails)** | n/a ($K \ne 2$) |
| `gate_s5` (MoE) | 4.0 | 10,368 | `7.9993` | `7.9519` | `9.099` | **PASS (0 fails)** | n/a ($K \ne 2$) |
| `wr_du` (dense) | 5.0 | 3,456 | `7.9995` | `7.9097` | `9.036` | **PASS (0 fails)** | n/a ($K \ne 2$) |

`zlib` compression ratio on raw `gate_s1` bytes is **`1.0004`** (incompressible random bits). Any tile has $R = 256K$ bits ($384\text{–}1280$ bits) mapping injectively into $256 \times 9.2 \approx 2350$ bits of `bytesum` output; no auxiliary table, palette, or entropy code can fit losslessly in $\le R$ bits.

### 3.2 Two Valid 0%-Bloat Lossless Bijections
1. **L1 — Load-Time Ring Rotation (All 7 $K$, 0.00% bloat):** Rotating each tile left by $b_0(0) = (K_A - 16) \bmod R$ bits once at model load eliminates per-tile alignment shifts on `K=2.0` and `K=4.0` at `M=16` (`o = 0` for all 16 lanes). Verified 0 fails across all 65,664 real tiles.
2. **L2 — NEW Lossless Strided-Crumb Bijection for `K=2.0` (0.00% bloat, 0 `XMAD`s in inner loop!):**
   - On $K=2.0$, each tile is 256 2-bit crumbs $c_0, \dots, c_{255}$ (64 bytes = 16 `uint32_t`). Window $i$ uses 8 consecutive crumbs $c_i \dots c_{i+7}$; stepping $i \to i+1$ shifts $P_i$ left by 2 bits, ejects 2-bit crumb $u_i = c_i$, and injects 2-bit crumb $v_i = c_{i+8}$:
     $$P_{i+1} = (P_i \ll 2) + D_{\text{rev}}[(v_i \ll 2) \mid u_i] \pmod{2^{32}}, \quad D_{\text{rev}}[(v \ll 2) \mid u] = v \cdot C - ((u \cdot C_L) \ll 16)$$
   - **Why Round 1's B2 failed (13.2 sl/w) and why L2 succeeds (`5.94–6.88` instr/w, `1.27×` silicon speedup):**
     - **0%-bloat crumb permutation:** Group the 32 crumbs of each lane $l \in [0..7]$ into 8 stride-8 chains (`ch = 0..7`), storing each 4-crumb chain `(c[32l+ch], c[32l+ch+8], c[32l+ch+16], c[32l+ch+24])` into **one contiguous byte** across two words `(W0, W1)`. Then EVERY transition `(u_i, v_i)` is **4 contiguous bits** (no cross-word masking!), and the seed window $h_0$ is reconstructed in just `2× IDP4A(W & 0x03030303, 0x01041040, 0)` + 1 `mul16_h0`!
     - **Replicated 16-entry `SHFL` table:** Store `D_rev[lane & 15]` in **one live register** replicated across both halves of the 32-lane warp. Because `SHFL.IDX` only reads the low 5 bits of `idx` and both halves hold the same 16 entries, **`SHFL` ignores bit 4 and all upper bits `[31:4]`**, eliminating ALL masking (`LOP32I`)!
     - **Resulting per-weight decode:** `SHR` + `SHFL.IDX` + `ISCADD(P, D, 2)` = **3.00 instructions/weight for extract + multiply combined** (**0 `XMAD`s in the 31-step loop!**).
   - **Verified:** Exhaustive `262,144/262,144` transition check with dirty upper bits = `0` fails; all `10,368` real `K=2.0` tiles (`2,654,208` real weights) in both `M=32` and `M=64` = `0/0` fails; **measured `355.9 Gw/s` (`1.27×` shipped)** for `C4` and **`319.3 Gw/s` (`1.14×`)** for exact `C0c`!
   - **Why unique to `K=2.0`:** Requires $K \mid 16$ (so `u_i` and `v_i` lie on clean stride-$16/K$ chains) AND $2K \le 5$ bits (so the $2^{2K}$-entry table fits in a 32-lane warp `SHFL`). $K=2.0$ gives $2K=4 \le 5$ (16 entries, fits twice in 32 lanes!); $K=4.0$ would need $2^8=256$ entries (> 32 lanes), and half-integer/odd $K$ break uniform crumb alignment.

---

## 4. Direction 2: Bit-Parallel / Architectural Exploration (`sm_61`)

1. **Fixing Round 1's `41d0798` Regression via In-Block `tk`-Split:**
   - `41d0798` put 4 tiles/warp (`M=32`) → grid shrank `144 -> 80` (down) and `288 -> 96` (GU), 8 accs + 7 stash regs → 48 regs (5 blk/SM) → `-42%` slower (`370–375 us` vs `254–263 us`).
   - Our **2-way (`M=16`)** and **4-way (`M=32`) in-block `tk`-split** keep **exact 8 tiles/block** (`144` blocks down, `144` GU) and reduce per-lane live state to **2 accumulators, 0 stash regs** → **`37–40` regs, 0 spills, 6 blocks/SM** → flips `-42%` into **`+14%` (exact) to `+27%` (`C4`)** on `K=2.0`!
2. **Silicon Discovery: `I2F` (`C0c`) beats Magic `FADD32I` (`C1c`) by +12%:**
   - Round 1's single-pipe slot model predicted `C1c` (`FADD32I`, `7.19` instr/w) would beat `C0c` (`I2F`, `7.22` instr/w) by ~30% because `I2F` is quarter-rate (28 lane-ops/clk/SM).
   - **Measured on silicon (`K=2.0`):** `C0c` = **`321.1 Gw/s` (`198.4 us`)** vs `C1c` = **`286.9 Gw/s` (`222.0 us`)** — `I2F` wins by **+12%**! (Same on `All-7-K`: `C0c` beats `C1c` by +7%.)
   - **Why:** On Pascal GP106, `I2F` executes on the dedicated SFU/conversion pipe (~28 lane-ops/clk/SM × 10 SMs × ~1.7–1.84 GHz ≈ **476–515 G/s** ceiling), which has plenty of slack at `321 Gw/s`. Replacing `I2F` with `FADD32I` moves 1.0 op/w **onto the already-saturated main ALU pipe**, slowing the kernel down!
3. **Stock `ptxas 12.6` 2-XMAD Impossibility & `LEA.HI + 2*DP2A.LO`:**
   - Exhaustively tested 30+ PTX patterns (`mul.wide.u16`, `mad.lo.u32` with `0x83DC0000`, `vmad`, `mad24`, `shf.r.clamp+add`). Stock `ptxas 12.6` cannot emit 2-XMAD `h*C`; minimum is 3 `XMAD`s (passing opaque `uint32_t CM = 0x83DCD12D` parameter to prevent re-folding to 4-XMAD+BFE).
   - Splitting $P$ via `mul.wide.u16` + `shf.r.clamp(16)` + `add` DOES emit **`2 XMAD + 1 LEA.HI`** (avoiding the 3rd XMAD!) because `rt.H0 | rhi.H0 == P` (verified exact on all 65,536 windows!), pairing directly with **`2*DP2A.LO`** (`32` regs, `6.281` instr/w). However, total ops/w = `2+1+2 = 5.0` ties `C4`'s `3 XMAD + 1 IDP4A + 1 XMAD.S16 = 5.0`, and measured **`299.6 Gw/s`** (`1.07×`), beaten by `C4 blk=16` (`309.3 Gw/s`) and `M=32-Strided C4` (`355.9 Gw/s`).
4. **Number-Theoretic Pairing Theorem Across All 7 $K$:**
   - Exact `(D, D+16)` halfword pairing exists iff $\exists i, j$ with $D(j) - D(i) = 16$. This holds **only for $K \in \{1.5, 2.0, 4.0\}$** (~67% of SAGE MoE). For $K \in \{2.5, 3.0, 3.5, 5.0\}$, **0 exact `(D, D+16)` pairs exist**; `(D-1) SHF + IADD(U+U)` pairs `(D, D+15)` on $K \in \{2.5, 3.0, 5.0\}$ (halving half-rate shifter ops), while $K=3.5$ has 0 pairs of either kind.
5. **Empirical Disproof of 64 KB / 128 KB `LDG` LUTs:**
   - `M=16-LUT128K-Exact` (`9.062` instr/w) measured **`17.0 Gw/s` (`0.06×`, 3742 us)** and `M=16-LUT64K-BS8` (`7.625` instr/w) measured **`17.4 Gw/s` (`0.06×`, 3654 us)** — **16× slower than shipped**, killed simultaneously by 64-bit address math (`+3–4` ops/w) AND L1/L2 thrashing (64–128 KB random access > 48 KB L1/SM).

---

## 5. Direction 3: Multi-Token Batching (`B = 1, 2, 3, 4`)

### 5.1 Roofline Correction
Increasing batch size $B$ reads the **exact same 2.26 bits/weight** from DRAM while executing $B\times$ MACs per weight. Arithmetic intensity therefore grows $B\times$, making the kernel **more compute/issue-bound per loaded weight**, NOT DRAM-BW-bound! Measured `Rd_Gw/s` (weights read/s) *decreases* from `282.5 -> 182.2 Gw/s` as $B$ goes `1 -> 4`, while **effective throughput per token (`Eff_GMAC/s = B × Rd_Gw/s`) scales super-linearly up to `2.57×`**, crossing the **`722.1 G MAC/s`** (batch-1 DRAM-BW equivalent, `<= 3.0` instr/w/tok) mark at **`B = 4`**!

### 5.2 Kernel Shape & Measured Results (`K=2.0`, GU `2304 -> 2×1152`, 8 slots, interleaved vs shipped `B=1`)
- **Shape:** 2-way `tk`-split `M=16-Paired` (`8 tiles/blk`, `40` regs, **`0` spills**, `6 blocks/SM`). Staged in 32-tile smem slabs `xs[32][16][B]` (`2048 B × B`: `8,240 B` at `B=4` exact FP32; `12,672 B` at `B=4` `C4` incl. outer float acc staging in smem — both `<= 16 KB/blk` for 6 blk/SM!). `+2` accumulator regs per extra vector; `1× LDS.128` loads all 4 tokens for a row simultaneously, shared across columns `c` and `c+8`.

| Kernel | $B$ | Issued SASS (ins/w) | **SASS ins/w/tok** | Regs / Spill | Smem (B) | Ship `B=1` (us) | Cand (us) | **Rd_Gw/s** (per weight) | **Eff_GMAC/s** (per token) | vs Ship `B=1` | `max_rel` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M=16-Batch C0c` (exact) | 1 | `7.812` | `7.812` | `40` / `0B` | 2,096 | 147.1 | 147.5 | 288.0 | 288.0 | `1.00×` | `5.80e-7` |
| `M=16-Batch C0c` (exact) | 2 | `9.000` | `4.500` | `40` / `0B` | 4,144 | 146.3 | 174.1 | 244.0 | **487.9** | **`1.68×`** | `6.75e-7` |
| `M=16-Batch C0c` (exact) | 3 | `10.375` | `3.458` | `40` / `0B` | 6,192 | 147.0 | 209.5 | 202.7 | **608.2** | **`2.11×`** | `6.39e-7` |
| **`M=16-Batch C0c` (exact)** | **4** | `11.312` | **`2.828`** | **`40` / `0B`** | **8,240** | 146.4 | 231.4 | 183.5 | **734.0** | **`2.53×`** | **`6.34e-7`** |
| `M=16-Batch C4` (`int16 x`) | 2 | `7.875` | `3.938` | `40` / `0B` | 6,336 | 146.1 | 172.0 | 246.9 | **493.7** | **`1.70×`** | `2.22e-5` |
| **`M=16-Batch C4` (`int16 x`)** | **4** | `9.812` | **`2.453`** | **`40` / `0B`** | **12,672** | 146.0 | 231.4 | 183.5 | **734.0** | **`2.52×`** | **`2.03e-5`** |

*(For Inkling's speculative decode / dense `B=2..4` paths where multiple tokens share the same matrix, `B=4` exact `C0c` delivers **2.51× higher MAC throughput** at `6.3e-7` accuracy with zero spills and 6 blk/SM!)*

---

## 6. Direction 4: Approximate Pareto Curve vs EXL3 Quantization Step

On real EXL3 weights (`gate_s1` $K=2.0$, $w_{\text{std}} = 0.9532$), one EXL3 quantization step (`±KINV`, where `KINV = fp16 0x1eee = 0.006767273`) produces `rms_rel ≈ 7.03e-3` (`max_rel ≈ 6.94e-3`), and half a step (`U[-0.5, +0.5]`) produces `rms_rel ≈ 2.03e-3`. Tested across `gate_s0` ($K=1.5$), `gate_s1` ($K=2.0$), and `gate_s3` ($K=3.0$) on 3 `x` distributions (`Real H128(x_with_10x_outliers)`, `Dist A` zero-mean + 6× outliers, `Dist B` adversarial all-positive `[0.5, 1.5]`):

| Method | Issued SASS (`K=2.0`) | Meas `Gw/s` (`K=2.0`) | `gate_s1` Real `H128` `rms_rel` | `rms / 1-step` (H128 / Dist A / Dist B) | Quant-Step Safety |
|---|---:|---:|---:|---:|---|
| **[REF] ±1 EXL3 quant step (`KINV/w`)** | — | — | `7.03e-3` | `1.02× / 1.03× / 1.01×` | Reference (`1.0×` step) |
| **[REF] ±0.5 EXL3 quant step (`U[-.5,.5]`)** | — | — | `2.03e-3` | `0.29× / 0.29× / 0.30×` | Reference (`0.5×` step) |
| `C0` Shipped FP32 `I2F` (uncentered) | `8.375` | `282.5` | `1.88e-6` | `0.0003× / 0.0006× / 0.0166×` | **Exact (`<< 1 step`)** |
| **`C0c` Zero-Centered `I2F` (`M=16` / `M=32-Str`)** | **`7.219` / `6.875`** | **`321.1` / `319.3`** | **`8.57e-7`** | **`0.0001× / 0.0001× / 0.0001×`** | **Exact (`<< 1 step`, 100× safer on Dist B)** |
| `C1c` Zero-Centered Magic `FADD` | `7.188` | `286.9` | `8.57e-7` | `0.0001× / 0.0001× / 0.0001×` | Exact (slower than `C0c`!) |
| **`C4` `IDP4A(-510)+XMAD.S16` (`blk=16`)** | **`6.344`** | **`309.3`** | **`1.65e-5`** | **`0.0024× / 0.0031× / 0.0017×`** | **Safe (`~350×` below 1 step!)** |
| **`C4` `M=32-Strided` (`blk=128`)** | **`5.938`** | **`355.9`** | **`2.18e-5`** | **`0.0032× / 0.0072× / 0.0018×`** | **Safe (`~140–300×` below 1 step!)** |
| `C3b` Folded `0x3F800000+255*bs` (per-tile) | `6.406` | `274.3` | `1.60e-5` | `0.0023× / 0.0031× / 0.0765×` | Safe, but slower than `C0c` on GPU |
| `A-FP16x` (fp16 $x$, exact `bs`) | — | — | `2.06e-4` | `0.030× / 0.032× / 0.032×` | Safe (`~33×` below step), dominated by `C4` |
| `C3a` Folded `0x3F800000+255*bs` (global) | `6.250` | `330.0` | `1.16e-4` | `0.017× / 0.040× /` **`1.092×`** | **FAILS Dist B (`> 1 step`)** |
| `C6-o1` `dp4a-8` (`blk=16` + 1 FP32 outlier) | — | — | `3.95e-3` | `0.57× / 0.56× / 0.44×` | Borderline (~0.6× step, > 0.5-step floor) |
| `C6` `dp4a-8` matched `sx_q` (`blk=16`) | `5.312` | `365.9` | `4.23e-3` | `0.61× / 0.82× / 0.44×` | **UNSAFE** (`0.61–0.82×` step; broke GU in ledger) |
| `C6` `dp4a-8` matched `sx_q` (`blk=128`) | — | — | `5.49e-3` | `0.80× / 1.81× / 0.46×` | **UNSAFE** (`1.81×` step) |
| `A-BS8` 8-bit `bs` (`round(bs/4)*4`, 64KB LUT) | `7.625` | `17.4` | `7.68e-3` | `1.11× / 1.24× / 23.1×` | **UNSAFE & 16× slower** |
| `A-2BYTE` (low 2 bytes ×2, 1 XMAD) | — | — | `1.01e+0` | `146× / 150× / 148×` | **CATASTROPHIC (garbage)** |

**Sharp Pareto cliff:** Everything at `int16 x` (`C4`, `0.002–0.007×` quant step) or exact FP32 (`C0c`, `0.0001×` step) sits **140×–10,000× below a single EXL3 quantization step**. Every cheaper approximation (`C6 dp4a-8`, `A-BS8`, `A-2BYTE`) jumps **two orders of magnitude** right to or past 1 full quantization step (`0.6×–150×`). There is **no useful middle ground** between `C4` (`5.94–6.34` instr/w) and `C6` (`4.97–5.31` instr/w).

---

## 7. Porting Recommendation (What to Port First, and Why)

**Honest engineering recommendation grounded in measured silicon `Gw/s`:**

1. **For homogeneous / dense `K=2.0` (and `K=4.0`) GEMVs — or if porting a batch-1 decode core first:**
   - **First choice (exact FP32, lowest risk, zero accuracy risk):** **`M=16-Paired C0c` (L1 load-time ring rotation + 2-way in-block `tk`-split + zero-centered `I2F`)**.
     - Measured **`+14–16%`** (`321.1` vs `280.8 Gw/s` on `K=2.0`), **`+3%`** on heterogeneous All-7-K MoE, **39–40 regs, 0 spills, 6 blk/SM**, exact same 8-tiles/blk grid, **3× more accurate than shipped** (`5.4e-7` vs `2.6e-6`), and requires only a trivial 0%-bloat load-time bit rotation (or compile-time offset folding as in `plan_pw.cpp` for zero re-layout on integer $K$).
     - **Do NOT port `C1c` (Magic `FADD32I`) or `C3b` (Folded per-tile):** both were empirically **slower** than `C0c` `I2F` on `sm_61` (`C1c` by `-10–12%`, `C3b` by `-15%`) because `I2F` utilizes Pascal's idle SFU/CVT pipe.
   - **Second choice (maximum batch-1 speed on `K=2.0`, 180× below 1 EXL3 quant step):** **`M=32-Strided C4`** (L2 0%-bloat Strided-Crumb bijection + replicated 16-entry `SHFL` + `ISCADD` + `XMAD.S16`, 4-way in-block `tk`-split) delivers **`+27%`** (`355.9` vs `280.3 Gw/s`, `40` regs, `0` spills, `3.92e-5` error) on `K=2.0` (`40.7%` of SAGE MoE). For uniform All-7-K `C4` without strided re-encoding, **`M=16-Paired C4 (blk=16)`** delivers **`+11–13%`** on `K=2.0` and **`+5%`** on heterogeneous All-7-K (`2.04e-5` error, `40` regs, `0` spills).

2. **Expected End-to-End Impact on Serial Inkling (`56.4 ms/tok`, `19.84 ms` MoE GEMV + `16.00 ms` dense GEMV+had):**
   - Because heterogeneous All-7-K `gemv_moe_s` only gains **`+3%` (exact `C0c`) to `+5%` (`C4`)** due to non-`K=2.0` wave tail stragglers (saving **`~0.6–1.0 ms/tok`** on MoE GEMV, plus ~`1–2 ms/tok` on any `K=2.0`/`4.0` dense GEMVs), **batch-1 decode math is NOT a big e2e lever on serial decode (~1–3% total e2e ceiling)**. Do not invest heavy porting effort expecting a 20%+ serial TPS jump — the hardware lower bound (`>= 6.6–7.2` stock-`ptxas` instr/w exact) confirms serial decode math on Pascal `sm_61` is near its floor.

3. **Biggest Real Throughput Lever Found: Direction 3 (`M=16-Batch C0c` at $B = 2\text{–}4$):**
   - For any multi-token path (speculative verification with compact `NT=2..4` blocks as identified in `opt/ledger.md:130–139`, or multi-user/batched dense decode), **`M=16-Batch C0c (exact)`** achieves **`1.68×` at `B=2` (`487.9 G MAC/s`), `2.11×` at `B=3` (`608.2 G MAC/s`), and `2.53×` at `B=4` (`734.0 G MAC/s`)** at **`6.3e-7` exact FP32 accuracy, 40 regs, 0 spills, 8.2 KB smem (6 blk/SM)** — the **only lossless design** that crosses the `<= 3.0 instr/w/tok` (`> 722 G MAC/s`) threshold on `sm_61`.

---

## 8. Reproduction Commands

```bash
# 1. Exhaustive CPU exactness, real-weight entropy/losslessness (16.81M weights), & approx Pareto curve
g++ -O3 -std=c++17 opt/kbench/exl3_math2/test_cpu.cpp -o /tmp/exl3_math2_cpu -lm && /tmp/exl3_math2_cpu

# 2. Exact sm_61 SASS instruction counts across all 23 kernels (CPU-only)
~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 --use_fast_math -Xptxas -v \
  -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -c opt/kbench/exl3_math2/sass_kernels.cu -o /tmp/exl3_math2_sass.o \
  2> /tmp/exl3_math2_ptxas.log && \
~/lfm-gtx1060-bench/cuda-12.6/bin/cuobjdump -sass /tmp/exl3_math2_sass.o > /tmp/exl3_math2_sass.txt && \
python3 opt/kbench/exl3_math2/count_sass.py /tmp/exl3_math2_sass.txt /tmp/exl3_math2_ptxas.log

# 3. Full GPU microbenchmark vs shipped inkwell_exl3_gemv_moe_s (ALWAYS under flock)
~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 --use_fast_math -Xptxas -v \
  -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
  -gencode arch=compute_61,code=sm_61 -Icrates/inkwell-cuda/src \
  opt/kbench/exl3_math2/bench.cu -o /tmp/exl3_math2_bench && \
flock /tmp/inkwell-gpu.lock /tmp/exl3_math2_bench 7
```
