# Group-Wise W4 / W5 / W6 Dense GEMV Kernel Study (`sm_61`, GTX 1060 6 GB)

## 1. Executive Verdict

**Group-wise W4 with exact F32 activations does not reach 2× over shipped EXL3, because the GTX 1060's 192.2 GB/s DRAM bus sets a hard zero-compute floor of 5.80 ms/token at 4.125 bpw.** However, our tuned `N32`-tiled `W4_g128_sym` F32 kernel reaches **6.98 ms/token** (within 4% of raw `LDG.128` STREAM on `unembed` at **172.3 GB/s**, and 83% of raw STREAM across all 7 shapes):

- **Vs bare `inkwell_exl3_gemv_s` (strictly same-clock interleaved, peak boost):** `8.85 ms → 6.98 ms` (**1.27×**, saves **−1.87 ms/token**).
- **Vs shipped `inkwell_exl3_had_in + inkwell_exl3_gemv_sf` (production dense chain):** `~10.24 ms → 6.98 ms` (**1.47×**, saves **−3.26 ms/token** at peak boost; **1.50–1.74× / −3.3 to −5.2 ms/token** under sustained warm SM clocks where compute-bound EXL3 slows to `10.5–12.2 ms` while DRAM-bound W4 stays flat).
- **VRAM cost:** **+43.43 MiB** total for `W4_g128_sym` (`+38.07 MiB` attention + `+5.36 MiB` unembed), or **+59.89 MiB** for `W4_g128_asym` / `W4_g64_sym` (`+47.67 MiB` attention + `+12.22 MiB` unembed).
- **Exact F32 accumulation is sufficient:** `int16` activation `dp2a` (`1.21` SASS ins/w) only saves `0.27 ms/token` over exact F32 (`6.71` vs `6.98 ms`) while raising max relative error 100× (`4.2e-5` vs `3.8e-7`). Keep exact F32 activations.

### Verdict by Format (Quality Gate: Mean KLD ≤ `0.0475`, Top-1 ≥ `94.66%`)

| Format | Eff. bpw | Modeled ms/tok (fused argmax) | vs bare `gemv_s` (same-clock) | vs shipped `had_in+gemv_sf` | VRAM Δ (Attn / Unemb / Total) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **EXL3 shipped** | ~3.79 | 8.85 bare / ~10.24 full | 1.00× | 1.00× | 0 / 0 / 0 MiB (499.96 MiB) | Baseline (`KLD 0.0475`, `94.66%`) |
| **`W4_g128_sym`** | **4.125** | **6.98 ms** | **1.27× (−1.87 ms)** | **1.47× (−3.26 ms)** | **+38.07 / +5.36 / +43.43 MiB** | **BEST SPEED & VRAM.** Adopt if GPTQ/AWQ hits KLD ≤ 0.0475, top-1 ≥ 94.66%. |
| **`W4_g128_asym`** | **4.250** | **7.09 ms** | **1.25× (−1.78 ms)** | **1.45× (−3.18 ms)** | **+47.67 / +12.22 / +59.89 MiB** | **PRIMARY CANDIDATE** if sym loses quality: +0.11 ms and +16.5 MiB for asymmetric zero-point. |
| **`W4_g64_sym`** | **4.250** | **7.24 ms** | **1.25× (−1.78 ms)** | **1.44× (−3.19 ms)** | **+47.67 / +12.22 / +59.89 MiB** | **CO-PRIMARY** (same VRAM as `g128_asym`, +0.15 ms slower; test against `g128_asym` on KLD). |
| `W4_g64_asym` | 4.500 | 7.37 ms | 1.20× (−1.47 ms) | 1.39× (−2.85 ms) | +66.87 / +25.96 / +92.83 MiB | **FALLBACK.** Eats ~93 MiB (~45 expert cache slots on 6 GB card); adopt only if `g128_asym`/`g64_sym` miss KLD. |
| `W5_g128_sym/asym` | 5.125–5.250 | 8.12 ms | 1.10× (−0.79 ms) | 1.27× (−2.20 ms) | +114.85 / +60.30 / +175.16 MiB | **REJECT GLOBALLY; SELECTIVE ONLY.** +175 MiB evicts too many MoE experts on 6 GB. Use only on 1–2 sensitive matrices if W4 fails KLD. |
| `W6_g128_sym/asym` | 6.125–6.250 | 9.28–9.52 ms | 0.95–0.96× (+0.4 ms) | 1.10–1.11× (−1.0 ms) | +191.64 / +115.25 / +306.89 MiB | **REJECT.** Slower than bare EXL3, +307 MiB VRAM. Only viable on tiny `wr_du` (+5.1 MiB across 24 layers) if ever needed. |

**Unembed alone (`W4_g128_sym`):** Because EXL3 pads unembed rows to `201,088` (`221.31 MiB`), `W4_g128_sym` at real vocab `200,058` costs only **+5.36 MiB** and saves **−0.32 to −0.65 ms/token** with fused argmax (`171.9 GB/s`). Attention (`48 × 6` shapes) delivers **~88% of the total latency win** (`−1.55 ms` bare / `−2.88 ms` shipped) because it also eliminates `288` `inkwell_exl3_had_in` launches and fused `had_out` epilogues per token.

---

## 2. Roofline, Hardware Floor & EXL3 Clock Sensitivity

All measurements ran on the target **NVIDIA GeForce GTX 1060 6 GB** (`sm_61`, 10 SMs, `192.2 GB/s` theoretical DRAM peak, `nvcc 12.6`, `-arch=sm_61 --use_fast_math`) using real dumped EXL3 tensors in `/tmp/sgemv_dense_gold` and `flock /tmp/inkwell-gpu.lock`.

### Phase 0: Raw `LDG.128` STREAM Ceiling vs Shipped EXL3 (31 interleaved reps)

| Shape (`K → N`) | EXL3 Bytes (KB) | Bare `gemv_s` (µs) | Shipped `had_in+gemv_sf` (µs) | Hadamard Overhead (µs / %) | Raw `LDG.128` Stream at W4-g128-sym Bytes (µs / GB/s) |
|---|---:|---:|---:|---:|---:|
| `wq_du` (`2304 → 3072`) | 2,592.0 | 41.70 | 46.85 | +5.15 (+12.4%) | 23.55 (155.0 GB/s) |
| `gate_attn` (`2304 → 3072`) | 2,592.0 | 41.98 | 47.10 | +5.12 (+12.2%) | 23.55 (155.0 GB/s) |
| `wr_du` (`2304 → 384`) | 540.0 | 10.11 | 16.32 | +6.21 (+61.4%) | 3.07 (148.5 GB/s) |
| `wk_dv` (`2304 → 1024`) | 1,008.0 | 19.01 | 24.51 | +5.50 (+28.9%) | 9.22 (132.0 GB/s) |
| `wv_dv` (`2304 → 1024`) | 1,008.0 | 19.42 | 25.25 | +5.82 (+30.0%) | 9.22 (132.0 GB/s) |
| `wo_ud` (`3072 → 2304`) | 3,024.0 | 45.06 | 50.18 | +5.12 (+11.4%) | 24.58 (148.5 GB/s) |
| `unembed` (`2304 → 200058`) | 226,224.0 | 2,031.23 | 2,094.08 | +62.85 (+3.1%) | 1,329.15 (178.8 GB/s) |
| **Modeled `48×attn6 + unemb`** | — | **10.541 ms** | **12.184 ms** | **+1.643 ms (+15.6%)** | **5.802 ms (zero-compute floor)** |

**Two key findings from Phase 0 vs Phase 2:**
1. **Why 2× is physically impossible at ≥4 bpw:** Zero-compute `LDG.128` streaming of 4.125 bpw weights takes **5.802 ms/token** (`48 × 93.19 µs + 1.329 ms`). Against peak-boost bare `inkwell_exl3_gemv_s` (`8.85 ms`), even zero-compute streaming caps out at `1.53×` (`1.77×` vs peak-boost `had_in+gemv_sf` `10.24 ms`).
2. **EXL3 is SM-clock sensitive; W4 is DRAM-pinned:** Phase 0 ran warm right after back-to-back `inkwell` runs (~1569 MHz SM clock, full 4004 MHz DRAM clock -> `178.8 GB/s` STREAM). Compute-bound EXL3 (`~8.4–9.5` ins/w) took `10.54 ms` bare / `12.18 ms` full in Phase 0, then sped up 1.19× to `8.85 ms` bare in Phase 2 as SM clocks boosted to ~1885 MHz. Memory-bound `N32` W4 (`2.65` ins/w) tracks the unthrottled DRAM clock. All speedups below compare every format against **its own strictly interleaved same-clock `inkwell_exl3_gemv_s` control** from Phase 2, with `had_in+gemv_sf` scaled by the exact interleaved Phase 0 overhead ratio (`+15.6%`).

---

## 3. Pascal `sm_61` Kernel Design, Unpack Tricks & SASS Analysis

### 3.1 Unpack Tricks on `sm_61` (Phase 1 Ablation on `wq_du` & `unembed`)

Pascal `sm_61` has 2×32 FP32 ALUs and 32 SFU/CVT pipes per SM partition, with dual-issue between `FFMA` and `I2F.F32.U8`. Four unpack strategies were tested in `w4dense_kernels.cuh`:

1. **`DUAL_I2F` (Winner, 0 shifts):** Pack 8 consecutive weights into one `uint32_t p` with bytes `[u0..u3]` in low nibbles and `[u4..u7]` in high nibbles. An opaque PTX `and.b32` (`asm("and.b32 %0, %1, %2;")`) produces `lo = p & 0x0F0F0F0F` and `hi = p & 0xF0F0F0F0`, stopping `ptxas` from folding `(p >> 8) & 0xF` into `BFE + I2F.F32.U32`. This unlocks native `I2F.F32.U8.B{0..3}` on both `lo` and `hi` with **zero shifts**, combining once via `fmaf(0.0625f, a_hi, a_lo)`.
2. **`BFE_I2F`:** Standard shift+mask per nibble → `0.88` shift/BFE ins/w, `4.41` ins/w.
3. **`LOP3_MAG`:** `lop3.b32` into `0x4B000000` (`2^23`) + `FADD(-8388608.f)` → zero `I2F`, but `FADD` + shifts burn FP32 ALU pipe (`6.25` ins/w; 1.19× slower on `unembed`).
4. **`LOP3_FLD`:** `lop3.b32` into `0x3F800000` (`1.0 + u/16`, fold `-1.5·Σx` and `16·scale` outside inner loop) → eliminates `FADD` and `I2F`, still costs `1.00` shift/w (`4.47` ins/w; matches row-major `DUAL_I2F` but loses precision slightly: `1.16e-6` vs `3.5e-7`).
5. **W5 / W6 split-plane merge (`0x10101010` / `0x30303030`):** Storing the upper 1 bit (W5) or 2 bits (W6) aligned to byte positions lets `lop3.b32` (`0xF8: a | (b & c)`) inject bits `[4]` or `[5:4]` directly into the `I2F.F32.U8` byte operand (`0..31` / `0..63`) with **zero extra `I2F` and zero extra `FFMA`**.

### 3.2 Why `N32` Tiled K-Outer Beats Row-Major by −1.0 ms/token

Row-major W4 (`1` warp = `2` rows, `32` lanes along `K`) stalled at `107–112 GB/s` on `wq_du`/`wo_ud` and `150–157 GB/s` on `unembed` due to: (a) 32-bit `LDG.32` weight loads, (b) 2-way `LDS.128` bank conflicts on stride-8 `x` loads, (c) redundant per-lane scale loads (`16` lanes loading the same `F16` scale), (d) wave-quantization tails (`wo_ud` = 144 blocks vs 70 blks/wave → 3 waves for 2.06 waves of work), and (e) cross-block `split_k > 1` atomic + `__threadfence()` overhead (+5–8 µs on `wr_du`/`wk_dv`, making cross-block split-K 1.6× slower than `sk1`).

**`N32` tiled K-outer layout** (`[ceil(N/32), K/32, 32]` of `uint4`, deterministic load-time transpose of row-major with **0 extra bytes on all 6 attention shapes** and **+6.9 KB** on `unembed`) fixes all five simultaneously:
- Each lane `l ∈ [0..31]` owns one complete output row `tile * 32 + l`.
- Weights load via coalesced **`LDG.128`** (`uint4` = 32 weights/lane = 512 B/warp).
- All 32 lanes read the **exact same** `float4 x` from shared memory → **1-cycle hardware broadcast, 0 bank conflicts** (`STAGE_X=true` beat `STAGE_X=false` `__ldg(x)` by 1.4–1.7× across all shapes because L1 tex cache thrashes with `LDG.128` weight traffic).
- **Zero `__shfl_xor_sync`** inside the `K` loop; **1 scale load per 128 weights** (16× fewer than row-major); coalesced 128 B `STG.32` output writes.
- For small `N` (`wr_du=384`, `wk_dv=1024`, `wq_du=3072`, `wo_ud=2304`), **`WPT` warps collaborate in-block** on the same 32-row tile (`WPT=16` on `wr_du`, `WPT=8` on `wk_dv`/`wv_dv`, `WPT=4` on `wq_du`/`gate_attn`/`wo_ud`, `WPT=1` persistent `grid=160` on `unembed`) and reduce in shared memory with **zero global atomics and zero `__threadfence()`**.

### 3.3 Phase 1 Ablation Summary (`W4_g128_sym`, 25 interleaved reps)

| Shape | Variant | Max Rel Err | Med (µs) | Min (µs) | GB/s | vs `gemv_s` |
|---|---|---:|---:|---:|---:|---:|
| `wq_du` | `RM_DUAL_I2F w8 r2 sk1 smem` | 3.16e-7 | 32.77 | 31.74 | 112.0 | 1.08× |
| `wq_du` | `RM_BFE_I2F w8 r2 sk1 smem` | 3.52e-7 | 33.79 | 32.70 | 108.6 | 1.03× |
| `wq_du` | `RM_LOP3_MAG w8 r2 sk1 smem` | 3.52e-7 | 33.76 | 32.77 | 108.7 | 1.03× |
| `wq_du` | `RM_LOP3_FLD w8 r2 sk1 smem` | 9.40e-7 | 32.26 | 31.42 | 113.8 | 1.10× |
| `wq_du` | **`N32_F32 w8 wpt4 smem`** | **3.25e-7** | **27.20** | **26.59** | **135.0** | **1.28×** |
| `wq_du` | `N32_F32 w8 wpt4 ldg` (no smem x) | 3.25e-7 | 38.91 | 37.76 | 94.3 | 0.92× |
| `wq_du` | `N32_DP2A w8 wpt4 smem` (int16 x) | 2.46e-5 | 26.62 | 25.60 | 137.9 | 1.34× |
| `wo_ud` | `RM_DUAL_I2F w8 r2 sk1 smem` | 2.08e-7 | 36.54 | 34.82 | 100.5 | 1.04× |
| `wo_ud` | **`N32_F32 w8 wpt4 smem`** | **2.94e-7** | **27.36** | **26.53** | **134.2** | **1.42×** |
| `wk_dv` | `RM_DUAL_I2F w8 r2 sk1 smem` | 3.32e-7 | 13.12 | 10.24 | 93.7 | 1.33× |
| `wk_dv` | **`N32_F32 w8 wpt8 smem`** | **3.24e-7** | **11.26** | **10.02** | **109.2** | **1.50×** |
| `wr_du` | `RM_DUAL_I2F w8 r2 sk1 smem` | 3.90e-7 | 7.17 | 6.14 | 65.1 | 1.23× |
| `wr_du` | **`N32_F32 w16 wpt16 smem`** | **3.04e-7** | **7.07** | **6.14** | **66.0** | **1.26×** |
| `wr_du` | `N32_DP2A w16 wpt16 smem` (int16 x) | 2.85e-5 | 5.70 | 4.83 | 82.0 | 1.44× |
| `unembed` | `RM_DUAL_I2F w8 r2 sk1 smem` | 3.53e-7 | 1,573.60 | 1,544.10 | 151.5 | 1.10× |
| `unembed` | `RM_BFE_I2F w8 r2 sk1 smem` | 4.01e-7 | 1,661.95 | 1,647.62 | 143.5 | 1.08× |
| `unembed` | `RM_LOP3_MAG w8 r2 sk1 smem` | 4.01e-7 | 1,876.99 | 1,857.54 | 127.1 | 0.97× |
| `unembed` | **`N32_F32 w4 wpt1 g160 smem`** | **3.78e-7** | **1,386.50** | **1,379.81** | **172.0** | **1.22×** |
| `unembed` | `N32_F32 w8 wpt1 g160 smem` | 3.78e-7 | 1,399.39 | 1,379.78 | 170.4 | 1.25× |
| `unembed` | `N32_DP2A w8 wpt1 g160 smem` | 2.70e-5 | 1,358.85 | 1,348.29 | 175.5 | 1.29× |

### 3.4 `cuobjdump -sass` & `ptxas` Resource Table (`sm_61`, Zero Spills Everywhere)

| Kernel | Thr | Regs | Spills | Blks/SM | Occ% | Inner `ins/w` | Core/w | `I2F`/w | `FFMA`/`DP2A`/w | `LDG`/w | `LDS`/w | `SHF`/w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shipped:inkwell_exl3_gemv_s` | 256 | 40 | 0 | 3 | 37.5% | ~8.4–9.5 | — | — | — | — | — | — |
| `shipped:inkwell_exl3_gemv_sf` | 256 | 48 | 0 | 3 | 37.5% | ~8.4–9.5 | — | — | — | — | — | — |
| `rm_w4_g128_sym_DUAL_I2F_w8_r2` | 256 | 32 | 0 | 8 | 100.0% | 3.812 | 3.22 | 1.00 | 1.12 | 0.25 | 0.19 | 0.00 |
| `rm_w4_g128_sym_BFE_I2F_w8_r2` | 256 | 40 | 0 | 6 | 75.0% | 4.406 | 3.84 | 1.00 | 1.12 | 0.25 | 0.19 | 0.88 |
| `rm_w4_g128_sym_LOP3_FLD_w8_r2` | 256 | 39 | 0 | 6 | 75.0% | 4.469 | 3.94 | 0.00 | 1.12 | 0.25 | 0.19 | 1.00 |
| `rm_w4_g128_sym_LOP3_MAG_w8_r2` | 256 | 39 | 0 | 6 | 75.0% | 6.250 | 5.56 | 0.00 | 1.12 | 0.25 | 0.19 | 0.88 |
| **`n32_w4_g128_sym_w8_k4_sm`** | **256** | **40** | **0** | **6** | **75.0%** | **2.648** | **2.34** | **1.00** | **1.02** | **0.04** | **0.26** | **0.00** |
| **`n32_w4_g128_asym_w8_k4_sm`** | **256** | **40** | **0** | **6** | **75.0%** | **2.672** | **2.34** | **1.00** | **1.02** | **0.04** | **0.26** | **0.00** |
| **`n32_w4_g64_sym_w8_k4_sm`** | **256** | **40** | **0** | **6** | **75.0%** | **2.766** | **2.42** | **1.00** | **1.05** | **0.05** | **0.27** | **0.00** |
| **`n32_w4_g64_asym_w8_k4_sm`** | **256** | **40** | **0** | **6** | **75.0%** | **2.797** | **2.44** | **1.00** | **1.05** | **0.05** | **0.27** | **0.00** |
| **`n32_w5_g128_sym_w8_k4_sm`** | **256** | **40** | **0** | **6** | **75.0%** | **3.305** | **2.94** | **1.00** | **1.02** | **0.07** | **0.26** | **0.34** |
| **`n32_w6_g128_sym_w8_k4_sm`** | **256** | **40** | **0** | **6** | **75.0%** | **3.281** | **2.92** | **1.00** | **1.02** | **0.07** | **0.26** | **0.31** |
| **`n32_argm_w4_g128_sym`** | **256** | **40** | **0** | **6** | **75.0%** | **2.695** | **2.35** | **1.00** | **1.02** | **0.04** | **0.26** | **0.00** |
| **`n32_argm_w4_g128_asym`** | **256** | **40** | **0** | **6** | **75.0%** | **2.703** | **2.36** | **1.00** | **1.02** | **0.04** | **0.26** | **0.00** |
| `n32_argm_w5_g128_sym` | 256 | 40 | 0 | 6 | 75.0% | 3.328 | 2.94 | 1.00 | 1.02 | 0.07 | 0.26 | 0.34 |
| `n32_argm_w6_g128_sym` | 256 | 40 | 0 | 6 | 75.0% | 3.289 | 2.91 | 1.00 | 1.02 | 0.07 | 0.26 | 0.31 |
| `n32_dp2a_g128_sym_w8_k1` | 256 | 40 | 0 | 6 | 75.0% | 1.211 | 0.98 | 0.01 | 0.51 | 0.04 | 0.14 | 0.12 |

`n32_w4_g128_sym` sits within **4.6% of the Pascal F32 hardware instruction floor** (`2.531` = `1.00 I2F + 1.00 FFMA + 0.25 LOP32I + 0.25 LDS.128 + 0.031 LDG.128`).

---

## 4. Full 7-Shape Interleaved GPU Benchmark Results (Phase 2, 31 Reps)

Every kernel passed CPU FP64 verification (`fails=0`, max relative error `1.36e-7` to `7.47e-7` for F32, `2.39e-5` to `4.18e-5` for `dp2a_i16`, and **100% exact argmax token match** across all 200,058 vocabulary rows).

### 4.1 Per-Shape Timings (Median µs / GB/s) for Top Formats vs Interleaved `inkwell_exl3_gemv_s`

| Shape | Interleaved `EXL3 gemv_s` (µs) | `W4_g128_sym` (µs / GB/s / vs S) | `W4_g128_asym` (µs / GB/s / vs S) | `W4_g64_sym` (µs / GB/s / vs S) | `W4_g64_asym` (µs / GB/s / vs S) | `W4_g128_dp2a` (µs / GB/s / vs S) | `W5_g128_sym` (µs / GB/s / vs S) | `W6_g128_sym` (µs / GB/s / vs S) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wq_du` | 34.72 | **29.18** / 125.8 / **1.19×** | 29.70 / 127.3 / 1.16× | 29.70 / 127.3 / 1.16× | 30.72 / 130.3 / 1.13× | 27.58 / 133.1 / 1.28× | 34.34 / 132.7 / 1.01× | 36.86 / 147.6 / 0.94× |
| `gate_attn` | 33.66 | **28.67** / 128.0 / **1.17×** | 29.70 / 127.3 / 1.14× | 30.21 / 125.2 / 1.15× | 31.20 / 128.3 / 1.12× | 27.58 / 133.1 / 1.28× | 32.48 / 140.3 / 1.07× | 36.86 / 147.6 / 0.96× |
| `wr_du` | 8.19 | **7.01** / 66.6 / **1.17×** | 6.78 / 70.9 / 1.28× | 6.62 / 72.6 / 1.24× | 6.69 / 76.0 / 1.22× | 5.12 / 91.2 / 1.60× | 7.87 / 73.4 / 1.12× | 7.07 / 97.3 / 1.16× |
| `wk_dv` | 17.15 | **12.19** / 100.9 / **1.41×** | 12.00 / 105.6 / 1.37× | 12.29 / 103.1 / 1.42× | 12.10 / 110.8 / 1.35× | 12.16 / 101.1 / 1.47× | 13.31 / 114.5 / 1.30× | 17.31 / 105.1 / 1.04× |
| `wv_dv` | 16.38 | **11.26** / 109.2 / **1.45×** | 12.29 / 103.1 / 1.42× | 13.22 / 95.8 / 1.36× | 11.87 / 112.9 / 1.38× | 11.97 / 102.8 / 1.37× | 13.31 / 114.5 / 1.29× | 15.87 / 114.6 / 1.07× |
| `wo_ud` | 38.78 | **28.29** / 129.8 / **1.37×** | 27.65 / 136.8 / 1.41× | 28.67 / 131.9 / 1.38× | 29.70 / 134.8 / 1.31× | 26.62 / 137.9 / 1.38× | 32.48 / 140.3 / 1.16× | 37.38 / 145.6 / 1.01× |
| `unembed` | 1,676.29 | **1,384.45** / 172.3 / **1.21×** | 1,419.26 / 173.1 / 1.19× | 1,507.81 / 162.9 / 1.12× | 1,562.62 / 166.4 / 1.08× | 1,377.02 / 173.2 / 1.22× | 1,746.94 / 169.5 / 0.96× | 2,057.12 / 172.0 / 0.82× |
| `unemb+argm` | 1,705.98 | **1,382.40** / **171.9** / **1.23×** | **1,417.22** / **172.8** / **1.19×** | **1,440.77** / **170.0** / **1.18×** | **1,504.74** / **172.3** / **1.12×** | — | **1,703.94** / **173.3** / **0.99×** | **2,018.82** / **174.8** / **0.83×** |

*Fused `unembed + argmax` saves an additional `2–67 µs` over separate `unembed` (by eliminating the 800 KB FP32 logit writeback and separate argmax pass) and hits `170.0–174.8 GB/s` (95.1–97.8% of raw `LDG.128` STREAM).*

### 4.2 Modeled Dense ms/token (`48 × sum(6 attn) + unembed`, Strictly Same-Clock Interleaved)

| Format | Eff. bpw | `attn6` (µs) | `48×attn6` (ms) | `unemb+argm` (ms) | Total (ms) | Interleaved `ctrl gemv_s` (ms) | vs `gemv_s` (Δ ms) | Scaled `EXL3 full` (~ms) | vs `EXL3 full` (Δ ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`W4_g128_sym`** | 4.125 | **116.6** | **5.597** | **1.382** | **6.979** | 8.852 | **1.27× (−1.87)** | 10.235 | **1.47× (−3.26)** |
| `W4_g128_sym_RM` | 4.125 | 134.0 | 6.431 | 1.514 | 7.945 | 9.021 | 1.14× (−1.08) | 10.415 | 1.31× (−2.47) |
| **`W4_g128_asym`** | 4.250 | **118.1** | **5.670** | **1.417** | **7.087** | 8.868 | **1.25× (−1.78)** | 10.269 | **1.45× (−3.18)** |
| **`W4_g64_sym`** | 4.250 | **120.7** | **5.794** | **1.441** | **7.235** | 9.011 | **1.25× (−1.78)** | 10.428 | **1.44× (−3.19)** |
| `W4_g64_asym` | 4.500 | 122.3 | 5.869 | 1.505 | 7.374 | 8.849 | 1.20× (−1.47) | 10.226 | 1.39× (−2.85) |
| `W4_g128_dp2a` | 4.125 | 111.0 | 5.329 | 1.377 | 6.706 | 8.869 | 1.32× (−2.16) | 10.264 | 1.53× (−3.56) |
| `W5_g128_sym` | 5.125 | 133.8 | 6.422 | 1.704 | 8.126 | 8.915 | 1.10× (−0.79) | 10.329 | 1.27× (−2.20) |
| `W5_g128_asym` | 5.250 | 132.8 | 6.376 | 1.746 | 8.122 | 8.900 | 1.10× (−0.78) | 10.301 | 1.27× (−2.18) |
| `W5_g64_sym` | 5.250 | 132.3 | 6.348 | 1.762 | 8.111 | 8.858 | 1.09× (−0.75) | 10.240 | 1.26× (−2.13) |
| `W6_g128_sym` | 6.125 | 151.3 | 7.265 | 2.019 | 9.284 | 8.930 | 0.96× (+0.35) | 10.337 | 1.11× (−1.05) |
| `W6_g128_asym` | 6.250 | 155.5 | 7.462 | 2.060 | 9.522 | 9.069 | 0.95× (+0.45) | 10.512 | 1.10× (−0.99) |
| `W6_g64_sym` | 6.250 | 151.9 | 7.291 | 2.065 | 9.356 | 8.811 | 0.94× (+0.54) | 10.191 | 1.09× (−0.83) |

---

## 5. Exact VRAM Delta Table

Exact parameter census from `~/models/inkling/rexl3-*.safetensors`:
- **Attention (24 layers × 6 matrices):** `644,087,808` params (**644.09M**). Shipped EXL3: **278.65 MiB** (`277.49 MiB` trellis + `1.17 MiB` `suh`/`svh`, ~3.63 bpw).
- **Unembed (`200,058 × 2,304`):** `460,933,632` real params (**460.93M**); `N32` padding rounds `200,058 → 200,064` (+6 rows = `+13,824` params = **+6.9 KB**). Shipped EXL3 pads to `201,088 × 2,304` (`463.31M`): **221.31 MiB** (`220.92 MiB` trellis + `0.39 MiB` `suh`/`svh`, ~4.03 eff bpw on real params).
- **Shipped dense total:** **499.96 MiB** (`278.65 + 221.31 MiB`).

| Format | Eff. bpw | Attn (MiB) | Δ Attn (MiB) | Unembed `N32` (MiB) | Δ Unembed (MiB) | Total (MiB) | **Δ Total (MiB)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EXL3 shipped** | ~3.79 | 278.65 | 0.00 | 221.31 | 0.00 | 499.96 | **0.00** |
| **`W4_g128_sym`** | **4.125** | **316.72** | **+38.07** | **226.67** | **+5.36** | **543.39** | **+43.43** |
| **`W4_g128_asym`** | **4.250** | **326.32** | **+47.67** | **233.53** | **+12.22** | **559.85** | **+59.89** |
| **`W4_g64_sym`** | **4.250** | **326.32** | **+47.67** | **233.53** | **+12.22** | **559.85** | **+59.89** |
| `W4_g64_asym` | 4.500 | 345.52 | +66.87 | 247.27 | +25.96 | 592.79 | +92.83 |
| `W5_g128_sym` | 5.125 | 393.50 | +114.85 | 281.61 | +60.30 | 675.12 | +175.16 |
| `W5_g128_asym` | 5.250 | 403.10 | +124.45 | 288.48 | +67.17 | 691.58 | +191.62 |
| `W5_g64_sym` | 5.250 | 403.10 | +124.45 | 288.48 | +67.17 | 691.58 | +191.62 |
| `W6_g128_sym` | 6.125 | 470.29 | +191.64 | 336.56 | +115.25 | 806.85 | +306.89 |
| `W6_g128_asym` | 6.250 | 479.88 | +201.23 | 343.43 | +122.12 | 823.32 | +323.36 |
| `W6_g64_sym` | 6.250 | 479.88 | +201.23 | 343.43 | +122.12 | 823.32 | +323.36 |

*VRAM fit check:* With BF16 ctrl/router already accepted on master (`4,934 MiB` peak vs `5,120 MiB` cap → `186 MiB` spare headroom), `W4_g128_sym` (`+43.4 MiB`) and `W4_g128_asym` / `W4_g64_sym` (`+59.9 MiB`) fit inside existing headroom **without shrinking the expert cache**. `W5` (`+175 MiB`) and `W6` (`+307 MiB`) would blow the 5,120 MiB cap or evict 80–150 cached experts.

---

## 6. Format Contract for the Quantization Agent

To keep safetensors portable and easy to produce, **store weights row-major `[N_out, K_in]` on disk** (no Hadamard `suh`/`svh` tensors needed). The Inkwell loader performs the deterministic 32-row `N32` tile transpose once at load time (takes <15 ms on CPU for 550 MiB, zero extra VRAM on attention, +6.9 KB on unembed).

### 6.1 Tensor Names (Safetensors)

Keep the standard Inkling module prefix `{prefix}` for each of the 6 attention projections per layer (`model.layers.{l}.self_attn.{q_proj,gate_proj,r_proj,k_proj,v_proj,o_proj}` or matching the current EXL3 stem) and `lm_head`:

| Tensor Suffix | Dtype | Shape (on disk, row-major) | Present in |
|---|---|---|---|
| `{prefix}.qweight` | `U8` (or `U32` LE `[N, K/8]`) | `[N_out, K_in / 2]` | W4, W5, W6 |
| `{prefix}.qweight_hi1` | `U8` (or `U32` LE `[N, K/32]`) | `[N_out, K_in / 8]` | W5 only |
| `{prefix}.qweight_hi2` | `U8` (or `U32` LE `[N, K/16]`) | `[N_out, K_in / 4]` | W6 only |
| `{prefix}.scales` | `F16` | `[N_out, K_in / G]` (sym) or `[N_out, K_in / G, 2]` (asym: `[..., 0]=scale, [..., 1]=min`) | W4, W5, W6 |

Metadata per safetensors header (or per-tensor json sidecar): `quant_type ∈ {"w4_g128_sym", "w4_g128_asym", "w4_g64_sym", "w4_g64_asym", "w5_g128_sym", "w5_g128_asym", "w6_g128_sym"}`. Mixed bitwidth across tensors is supported automatically from tensor suffixes + `scales.shape`.

### 6.2 Group Axis & Dequantization Equation

- **Group axis:** Along `K_in` (input features, contiguous in memory). Group `g ∈ [0, K_in/G)` covers input columns `[g·G, (g+1)·G)`, with `G ∈ {128, 64}` and `K_in % 256 == 0` (`2304` and `3072` both satisfy this).
- **Symmetric (`scales` shape `[N_out, K_in/G]` F16):**
  - Quantized integer `u ∈ [0, 2^b - 1]` stored unsigned with bias `Z = 2^(b-1)` (`Z = 8` for W4, `16` for W5, `32` for W6).
  - `w_hat = scale * (u - Z)` (both `[-8, 7]` and `[-7, 7]` work identically).
- **Asymmetric (`scales` shape `[N_out, K_in/G, 2]` F16 interleaved `{scale, min}`):**
  - Quantized integer `u ∈ [0, 2^b - 1]` unsigned; `min` is the F16 minimum weight of the group (typically negative).
  - `w_hat = scale * u + min` (so `y += scale * Σ(u·x) + min * Σ(x)`).

### 6.3 Exact Bit Packing (Interleaved Nibbles for Zero-Shift `I2F.F32.U8`)

For every 8 consecutive weights `u[0..7]` in row `r`, columns `col0 .. col0+7` (one 32-bit little-endian word `p`):
- **Low 4 bits (`qweight`):** Byte `b ∈ {0, 1, 2, 3}` holds:
  - low nibble (bits `0..3`): `u[b] & 0xF`
  - high nibble (bits `4..7`): `u[4 + b] & 0xF`
  - Equivalently in Python/NumPy for `u` shaped `[N, K/8, 8]`:
    ```python
    lo4 = u & 0x0F
    # 4 bytes per 8 weights: byte b = lo4[..., b] | (lo4[..., 4 + b] << 4)
    qweight = (lo4[..., 0:4] | (lo4[..., 4:8] << 4)).astype(np.uint8).reshape(N, K // 2)
    ```
- **W5 upper 1 bit (`qweight_hi1`, 4 low words = 32 weights → 1 `uint32` `H`):**
  - For `u` shaped `[N, K/32, 4, 8]` (indices `[row, hidx, k in 0..3, t in 0..7]`):
    - Byte `b ∈ {0..3}` of `H` holds bit `k` = `(u[..., k, b] >> 4) & 1` and bit `4 + k` = `(u[..., k, 4 + b] >> 4) & 1`.
- **W6 upper 2 bits (`qweight_hi2`, 2 low words = 16 weights → 1 `uint32` `H`):**
  - For `u` shaped `[N, K/16, 2, 8]` (indices `[row, hidx, k in 0..1, t in 0..7]`):
    - Byte `b ∈ {0..3}` of `H` holds bits `[2k+1 : 2k]` = `(u[..., k, b] >> 4) & 3` and bits `[4+2k+1 : 4+2k]` = `(u[..., k, 4 + b] >> 4) & 3`.

### 6.4 Load-Time `N32` Tile Transpose (Runtime Loader Responsibility)

At model load, the runtime transposes each row-major matrix into 32-row tiles (`nt = ceil(N_out / 32)`, zero-padding tail rows `N_out .. 32*nt - 1` so `unembed` adds only 6 rows = 6.9 KB):
- `Wlo_t[tile, k_u4, lane]` = `uint4` of row `32*tile + lane`'s 4 consecutive `uint32` words `[4*k_u4 .. 4*k_u4 + 3]`
- `Whi1_t[tile, k_u4, lane]` = `uint32` (`W5`), `Whi2_t[tile, k_u4, lane]` = `uint2` (`W6`)
- `S_t[tile, g, lane]` = `F16` (`sym`) or `F16x2` `{scale, min}` (`asym`)

---

## 7. Recommendation & Next Steps

1. **Quantization agent test order:** Start with **`W4_g128_sym`** (GPTQ / AWQ) on attention + unembed (`+43.4 MiB`, `−3.26 ms/tok` shipped). If KLD > `0.0475` or top-1 < `94.66%`, test **`W4_g128_asym`** and **`W4_g64_sym`** (`+59.9 MiB`, `−3.18 ms/tok` shipped).
2. **Mixed fallback if only a few matrices regress KLD:** Keep `W4_g128_sym/asym` everywhere else and bump only the sensitive matrices (`wo_ud` or `wr_du`, or keep `unembed` in EXL3 / `W4_g64_asym`) rather than moving all dense weights to `W5`/`W6`.
3. **Reproducing this study:** Run `opt/kbench/w4dense/build_and_run.sh /tmp/sgemv_dense_gold` (compiles `w4dense_kernels.cuh` + `bench_w4dense.cu` with `nvcc 12.6 -arch=sm_61`, runs `sass_inspect.py` on `cuobjdump -sass`, and executes the 31-rep interleaved benchmark under `flock /tmp/inkwell-gpu.lock`).
