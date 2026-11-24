# `mul_mat_vec_f` & MoE Hadamard Kernel Study (`sm_61`, GTX 1060 6 GB)

## Headline Verdict

**Strictly bit-identical fixes recover ~4.6–5.25 ms/tok** (bench: **−5.25 ms/tok**; nsys-adjusted: **−4.6 to −5.4 ms/tok**), moving Inkling-2-Mini decode from ~43 ms/tok to **~37.7–38.4 ms/tok** (meeting the ≤ 38 ms/tok target) for **+120 KiB** of VRAM and zero accuracy loss. Adding one optional fused BF16 `sink + relbias` kernel (`max_abs = 4.77e-7`, flagged reordered vs opaque cuBLAS SGEMM) adds another **−0.29 ms/tok** (bench) / **~−0.55 ms/tok** (nsys) while freeing **512 KiB** VRAM.

The single largest finding is a **bit-identical descriptor-placement bug** in `crates/inkwell-run/src/runner.rs:1129–1139`: even when a layer's expert weights are resident in VRAM (`on_dev == true`, 10 loop layers = 30 visits/tok), its descriptor tables (`descs_g/u/d`, `128 × 32 B = 4 KiB` each) are allocated via `host_alloc_mapped` (zero-copy PCIe host memory). On Pascal (`sm_61`), uncached PCIe descriptor fetches stall all 480 CTAs per MoE visit (**14,400 CTAs/tok**), adding **+140.1 µs/visit** (**+4.20 ms/tok** in microbench; corroborated within 4% by `nsys_mix_on.sqlite` where `ON_DEV` `gemv_moe_s(DN)` is 131.03 µs vs `CACHED` VRAM-desc `85.35 µs` = +45.68 µs vs bench +47.74 µs). Moving `on_dev` `descs_g/u/d` to `DevBuf` costs **120 KiB** VRAM (or 288 KiB for all 24 layers) and fixes both `had_in/out_moe` (−0.93 to −1.31 ms/tok) and `gemv_moe_s` (−2.8 to −3.05 ms/tok) in one shot.

A second key empirical finding: **do not fuse `had_in`/`had_out` into `inkwell_exl3_gemv_moe_s`**. Even when constrained via `__launch_bounds__(256, 6)` to 40 registers (matching shipped `gemv_moe_s`), epilogue fusion introduces 16 bytes of local stack traffic, two `__syncthreads()` barriers, and a 1-warp tail, regressing the full MoE chain by **+18–20 µs/visit** (`300.9–302.3 µs` vs `282.6 µs`). Instead, leaving `gemv_moe_s` untouched and vectorizing/fusing the surrounding Hadamard + SITU + wsum kernels (`H3e`, 8 → 6 launches/visit, 0 spills, bit-identical) saves **−4.90 µs/visit (−0.235 ms/tok)** cleanly.

---

## 1. Exact Attribution (`nsys` Trace Comparison)

All timings below come from 50 steady-state decode steps (steps 10–59 of 68) in `/tmp/nsys_mix_on.sqlite` (current master, `--cuda-graph-trace=node`) and `/tmp/nsys_pw16_ctl.sqlite` (prior master before BF16 controller weights + `rms2down`/`gf_apply` fusions).

### 1.1 Why `mul_mat_vec_f` Dropped from 3.36 ms/tok to 0.67 ms/tok

In prior master (`/tmp/nsys_pw16_ctl.sqlite`), `mul_mat_vec_f<float, float>` accounted for **3.357 ms/tok** across 304 launches/tok:
- 96× `ctrl_gf_cat` `[4608, 64]` F32: `1.526 ms/tok` → now fused into `inkwell_gf_apply<bf16>` (`0.701 ms/tok`)
- 96× `ctrl_down` `[64, 4608]` F32: `0.993 ms/tok` → now fused into `inkwell_rms2down<bf16>` (`1.030 ms/tok`)
- 48× `mlp.gate` `[128, 2304]`: `0.507 ms/tok` → now BF16 `mul_mat_vec_f` (`0.396 ms/tok`)
- 48× `attn_sink_proj` `[24, 2304]`: `0.253 ms/tok` → now BF16 `mul_mat_vec_f` (`0.209 ms/tok`)
- 16× `k/v_sconv.weight_proj` `[32, 2304]`: `0.078 ms/tok` → now BF16 `mul_mat_vec_f` (`0.063 ms/tok`)

### 1.2 Current Master Kernel Attribution (`/tmp/nsys_mix_on.sqlite`)

| Group / Kernel | Model Op(s) | Dtype & Shape | Launches/tok | Avg (µs) | Total (ms/tok) | Call Site |
|---|---|---|---:|---:|---:|---|
| **`mul_mat_vec_f<bf16,f32>`** | `mlp.gate.weight` | BF16 `[128, 2304]` | 48 | 8.26 | **0.396** | `runner.rs:1787` |
| `inkwell_router` | top-8 sigmoid + bias | F32 `[128]` | 48 | 17.44 | **0.837** | `runner.rs:1796`, `kernels.cu:4478` |
| **`mul_mat_vec_f<bf16,f32>`** | `attn_sink_proj.weight` | BF16 `[24, 2304]` | 48 | 4.35 | **0.209** | `runner.rs:1740` |
| `sgemm_32x32x32_NT_vec` | `attn_rel_bias_pad × r` | F32 `[1025, 16]×[16, 24]` | 48 | 9.66 | **0.464** | `runner.rs:1741` (widened `:963`) |
| `inkwell_relbias_gather` | `table[rel_pos] - sinks` | F32 `[n_kv, 24]` | 48 | 1.85 | **0.089** | `runner.rs:1748` |
| **`mul_mat_vec_f<bf16,f32>`** | `k/v_sconv.weight_proj` | BF16 `[32, 2304]` | 16 | 3.95 | **0.063** | `runner.rs:1693–1694` |
| `inkwell_rms2down<bf16>` | controller `rms2cat + down` | BF16 `[64, 4608]` | 96 | 10.73 | **1.030** | `runner.rs:1559` (72 in 24×3 triads `:1621`) |
| `inkwell_gf_apply<bf16>` | controller `gf_cat + apply` | BF16 `[4608, 64]` | 96 | 7.31 | **0.701** | `runner.rs:1579` (72 in 24×3 triads `:1621`) |
| **`had_in_moe` (ON_DEV)** | GU (30) + DN (30) | F32/F16, **PINNED descs** | 60 | 14.20 / 6.67 | **0.626** | `runner.rs:1129`, `moe.rs:472` |
| **`had_out_moe` (ON_DEV)** | GU (30) + DN (30) | F32/F16, **PINNED descs** | 60 | 9.63 / 11.81 | **0.643** | `runner.rs:1129`, `moe.rs:472` |
| **`had_in_moe` (CACHED)** | GU hit/miss (36) + DN (18) | F32/F16, VRAM `sdescs` | 54 | 11.09 / 2.78 | **0.450** | `moe.rs:451`, `kernels.cu:3655–3705` |
| **`had_out_moe` (CACHED)** | GU (18) + DN (18) | F32/F16, VRAM `sdescs` | 36 | 3.48 / 3.47 | **0.126** | `moe.rs:451`, `kernels.cu:3702–3721` |
| `situ_pair` + `expert_wsum` | MoE activation + reduce | F32 `[8, 1152]` / `[8, 2304]` | 96 | 2.73 / 2.28 | **0.240** | `kernels.cu:3550,3570,3703,3722` |
| *(Coupled)* `gemv_moe_s` (ON_DEV) | GU (30) + DN (30) | EXL3, **PINNED descs** | 60 | 206.55 / 131.03 | **10.127** | `runner.rs:1129`, `moe.rs:472` |
| *(Ref)* `gemv_moe_s` (CACHED) | GU (18) + DN (18) | EXL3, VRAM `sdescs` | 54 | 304.64* / 85.35 | **7.020** | `moe.rs:451` (*split hit+miss) |

*Note: Remaining `mul_mat_vec_f<bf16,f32>` sums to `0.396 + 0.209 + 0.063 = 0.668 ms/tok` (112 launches/tok). MoE Hadamard (`had_in_moe` + `had_out_moe`) sums to `1.845 ms/tok` (210 launches/tok), of which `1.269 ms/tok` (69%) is on the 30 `on_dev` visits alone.*

---

## 2. Hardware Calibration, Floors & Root-Cause Gaps

Measured on the target **NVIDIA GeForce GTX 1060 6 GB** (`sm_61`, 10 SMs, L2 = 1536 KiB, 192-bit GDDR5 @ 4004 MHz, peak 192.2 GB/s) inside CUDA Graphs (`bench_mmvf_had.cu`, Suite 0):
- **CUDA Graph empty-kernel launch floor:** `med = 1.000 µs` (`min = 0.960 µs`)
- **Raw `LDG.128` DRAM streaming bandwidth (16 MiB):** `med = 94.94 µs` → **`176.7 GB/s`**

| Op | Bytes / call | BW floor @ 176.7 GB/s | Launch floor | Bench Base (µs) | nsys Avg (µs) | Primary Bottleneck / Root Cause |
|---|---:|---:|---:|---:|---:|---|
| `mlp.gate` `[128, 2304]` BF16 | 599 KiB | 3.39 µs | 1.0 µs | 6.08 | 8.26 | `mmvf-256` 1 row/blk (`grid=128`), 32-bit `bf16x2` loads, re-reads `x[2304]` 128× |
| `inkwell_router` `[128]` F32 | 1.1 KiB | 0.01 µs | 1.0 µs | 14.66 | 17.44 | **`grid=(1,1,1)` serial thread-0 top-8**: 128 serial `expf` + 8×128 serial comparisons |
| `rms2down<bf16>` `[64, 4608]` | 608 KiB | 3.44 µs | 1.0 µs | 9.24 | 10.73 | **`18,688 B` smem** (`sg[4608]`) limits occupancy to 2–3 blks/SM; redundant `rms2cat` across 64 CTAs |
| `gf_apply<bf16>` `[4608, 64]` | 617 KiB | 3.49 µs | 1.0 µs | 6.38 | 7.31 | 32-bit `bf16x2` loads + 3× separate launches on 24 attention triads |
| `sink` + `relbias` + `gather` | 183–243 KiB | 1.0–1.4 µs | 3.0 µs | 9.15–9.50 | 16.0 | Computes all 1025 rows in F32 cuBLAS SGEMM (`[1025,24,16]`) then discards `1025 - n_kv` rows; 3 launches |
| `k/v_sconv` `[32, 2304]` ×2 | 313 KiB | 1.77 µs | 2.0 µs | 5.50 | 7.90 | 2 tiny `grid=32` launches back-to-back |
| `had_in/out_moe` (ON_DEV, 4k) | 558 KiB | 3.16 µs | 4.0 µs | **55.04** | **42.31** | **BUG: `descs_g/u/d` in `host_alloc_mapped` (PCIe)** → 192 CTAs stall on uncached PCIe reads; scalar loads + idle tail warps |
| `had_in/out_moe` (VRAM, 4k) | 558 KiB | 3.16 µs | 4.0 µs | 11.32 | ~12.5 | 4 launches + scalar loads + idle tail warps (`k=2304` = 18 warps on `5×4=20` warps); `situ_pair`/`wsum` add 2 more launches |
| `gemv_moe_s` GU+DN (ON_DEV) | ~8.8 MiB* | ~50 µs* | 2.0 µs | **363.7** | **337.6** | **Same PCIe descriptor bug**: 288 CTAs read `descs_g/u/d` over PCIe (+101.7 µs/visit vs VRAM descs!) |

*\*EXL3 GEMV is trellis-decode latency/compute-bound, not pure DRAM-bound; the +101.7 µs gap between `PINNED` and `VRAM` descriptors is pure PCIe stall.*

---

## 3. Microbenchmark Results & Exactness Verification

All measurements used **CUDA Graph interleaved round-robin replay** (25 interleaved reps per variant, rotating across weight buffers exceeding L2 to avoid L2 aliasing, reporting median & min µs/call), compiled with `nvcc 12.6 -O3 -std=c++17 --use_fast_math -arch=sm_61` and executed under `flock /tmp/inkwell-gpu.lock` (`opt/kbench/mmvf_had/build_and_run.sh`). Suite 5 uses **real EXL3 weights** from `/tmp/sgemv_gold/`.

### Suite 1 — `mlp.gate [128, 2304] BF16` + `inkwell_router` (48×/tok)

| Variant | Exactness Status | Bit Diffs / Max Abs | Regs / Smem / Spills | Med (µs) | Min (µs) | ms/tok | Speedup | Save (ms/tok) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `BASE`: `mmvf_bf16` + `inkwell_router` | reference | — | 31+32 / 2.3K / 0 | 20.70 | 20.51 | 0.994 | 1.00× | +0.000 |
| `(alone) mmvf_bf16_ref [128, 2304]` | reference | — | 31 / 0.1K / 0 | 6.08 | 5.91 | 0.292 | — | — |
| `(alone) inkwell_router` (serial) | reference | — | 32 / 2.3K / 0 | 14.66 | 14.53 | 0.703 | — | — |
| `(alone) router_par_128` (4-warp) | **BIT-EXACT** | `ids=8/8, 0/8, 0.0` | 32 / 0.6K / 0 | 5.76 | 5.60 | 0.276 | — | — |
| `M1a`: `mmvf_mr<4>` + `router_par_128` | **BIT-EXACT** | `ids=8/8, 0/8, 0.0` | 40+32 / 0.5K / 0 | 11.58 | 11.42 | 0.556 | 1.79× | **−0.438** |
| **`M1b-4`: `gate_router_fused_exact<4>`** | **BIT-EXACT** | **`ids=8/8, 0/8, 0.0`** | **40 / 1.6K / 0** | **11.12** | **10.93** | **0.534** | **1.86×** | **−0.460** |
| `M1b-8`: `gate_router_fused_exact<8>` | **BIT-EXACT** | `ids=8/8, 0/8, 0.0` | 48 / 2.2K / 0 | 11.62 | 11.48 | 0.558 | 1.78× | −0.436 |
| `M1c`: `gate_router_fused_ldg128` | **REORDERED** | `ids=8/8, 2/8, 5.96e-8` (logits `1.07e-6`) | 38 / 10.4K / 0 | 11.14 | 10.98 | 0.535 | 1.86× | −0.459 |

**Takeaway:** Bit-exact `M1b-4` (`11.12 µs`) matches reordered `LDG.128` (`11.14 µs`) to the hundredth of a microsecond. Because `inkwell_router` averages 17.44 µs in `nsys` vs 14.66 µs in bench, real model savings are **~−0.55 ms/tok** (100% bit-exact, 48 fewer launches/tok).

### Suite 2 — Controllers `rms2down<bf16>` + `gf_apply<bf16>` (96×/tok = 24×3 attn triads + 24 singles)

**Suite 2A — Single controller (24 single sites/tok; table shows full 96× scale for comparison):**

| Variant | Exactness Status | Bit Diffs / Max Abs | Regs / Smem / Spills | Med (µs) | ms/tok (×96) | Save (×24 singles) |
|---|---|---|---:|---:|---:|---:|
| `BASE`: shipped `rms2down` + `gf_apply` | reference | — | 32+11 / 18.7K / 0 | 15.51 | 1.489 | +0.000 |
| `(alone)` shipped `rms2down` (`grid=64`) | reference | — | 32 / 18.7K / 0 | 9.24 | 0.887 | — |
| `(alone)` `rms2down_nosmem_mr<2>` | **BIT-EXACT** | `0 / 0.0` | 40 / 0.4K / 0 | 8.35 | 0.802 | −0.021 |
| **`(alone)` `rms2down_nosmem_mr<4>`** | **BIT-EXACT** | **`0 / 0.0`** | **48 / 0.6K / 0** | **7.32** | **0.703** | **−0.046** |
| `(alone)` `rms2down_ldg128` | **REORDERED** | `55/64, 1.91e-6` | 44 / 0.1K / 0 | 7.55 | 0.725 | −0.041 |
| `(alone)` shipped `gf_apply` | reference | — | 11 / 0 / 0 | 6.38 | 0.612 | — |
| `(alone)` `gf_apply_opt<1>` (ptr-array arg) | **BIT-EXACT** | `0 / 0.0` | 14 / 0 / 0 | 7.72 | 0.741 | *(+1.34 µs ptr-chase)* |
| `(alone)` `gf_apply_ldg128<1>` | **REORDERED** | `1056/2304, 4.17e-7` | 28 / 0.4K / 0 | 6.19 | 0.594 | −0.005 |
| `M2a-4`: `nosmem<4>` + shipped `gf_apply` | **BIT-EXACT** | `0 / 0.0` | — | *13.70** | *1.315* | **−0.043** (×96: **−0.174**) |
| `M2c`: `ldg128` pair (1×) | **REORDERED** | `1.91e-6` / `4.17e-7` | 44+28 / 0.5K / 0 | 13.65 | 1.311 | −0.045 |

*\*Note: `gf_apply_opt<1>` paid +1.34 µs solely from device-memory pointer-array indirection (`w_arr[b]`); keeping direct/by-value pointer args eliminates that overhead completely.*

**Suite 2B — Batched 3× attention controller triad (`q, k, v` at `runner.rs:1619–1622`, 24 visits/tok = 72 sites):**

| Variant | Exactness Status | Med (µs) | Min (µs) | ms/tok (×24) | Speedup | Save (ms/tok) |
|---|---|---:|---:|---:|---:|---:|
| `BASE`: 3× shipped (`rms2down` + `gf_apply`) | reference | 47.06 | 46.59 | 1.130 | 1.00× | +0.000 |
| `M2b-2`: `nosmem<2,3>` + `gf_opt<3>` | **BIT-EXACT** | 38.55 | 38.25 | 0.925 | 1.22× | −0.204 |
| **`M2b-4`: `nosmem<4,3>` + `gf_opt<3>`** | **BIT-EXACT** | **34.99** | **34.69** | **0.840** | **1.35×** | **−0.290** |
| `M2c-3`: `ldg128<3>` pair | **REORDERED** | 29.64 | 29.35 | 0.711 | 1.59× | −0.418 |

**Takeaway:** Dropping `sg[4608]` (`18,688 B` smem) in `rms2down` and batching the 24 attention triads (`M2b-4` + `M2a-4`) saves **−0.333 ms/tok strictly bit-exact** (and ~**−0.40 ms/tok** once pointer arrays are passed by value in a 3-element struct via constant memory instead of global pointer indirection), eliminating 96 launches/tok. Flagged reordered `M2c-3` saves an additional −0.13 ms/tok with `max_abs = 1.91e-6` (`rms2down` 32-lane warp tree) and `4.17e-7` (`gf_apply` 8-lane sub-warp tree).

### Suite 3 — `attn_sink_proj [24, 2304] BF16` + `rel_bias` SGEMM + `relbias_gather` (48×/tok)

| Variant | Exactness Status | Bit Diffs / Max Abs / Max Rel | Med (µs) | ms/tok | Speedup | Save (ms/tok) |
|---|---|---|---:|---:|---:|---:|
| `BASE` (`n_kv=64`): `cublasSgemm` + `mmvf` + `gather` | reference | — | 9.15 | 0.439 | 1.00× | +0.000 |
| **`M3a` (`n_kv=64`): `sink_relbias_fused` (1 launch)** | **REORDERED*** | `333/1536, 4.77e-7, 1.02e-7` | **3.10** | **0.149** | **2.95×** | **−0.290** |
| `BASE` (`n_kv=512`): `cublasSgemm` + `mmvf` + `gather` | reference | — | 9.50 | 0.456 | 1.00× | +0.000 |
| **`M3a` (`n_kv=512`): `sink_relbias_fused` (1 launch)** | **REORDERED*** | `2746/12288, 4.77e-7, 1.01e-7` | **3.87** | **0.186** | **2.46×** | **−0.270** |

*\*Flagged exactness detail: the `sinks[h]` dot uses the exact `mmvf-256` reduction order (`0` bit diffs), and `attn_rel_bias_pad` is natively BF16 in the model (exact conversion). The only difference is the 16-element FMA dot `rel_bias[pos, 0..16] · r[0..16, h]` vs opaque cuBLAS `sgemm_32x32x32_NT_vec` K=16 accumulation order, bounded by `≤ 16 · ε_f32 ≈ 1.9e-6` (`max_abs = 4.77e-7`, `max_rel = 1.02e-7` observed). In `nsys_mix_on.sqlite`, the 3 baseline kernels total `0.762 ms/tok`, so real savings are **~−0.55 ms/tok** plus **−512 KiB VRAM** (keep `attn_rel_bias_pad` in BF16 and drop the `[1025, 24]` scratch table).*

### Suite 4 — `k_sconv.weight_proj` + `v_sconv.weight_proj` `[32, 2304] BF16` (8 visits/tok = 16 launches)

| Variant | Exactness Status | Med (µs) | ms/tok | Speedup | Save (ms/tok) |
|---|---|---:|---:|---:|---:|
| `BASE`: 2× `mmvf_bf16_ref [32, 2304]` | reference | 5.50 | 0.044 | 1.00× | +0.000 |
| **`M4-1`: `kv_sconv_batched<1>` (`grid=(32,2)`)** | **BIT-EXACT** (`0` diffs) | **3.74** | **0.030** | **1.47×** | **−0.014** |
| `M4-2`: `kv_sconv_batched<2>` (`grid=(16,2)`) | **BIT-EXACT** (`0` diffs) | 3.76 | 0.030 | 1.46× | −0.014 |

### Suite 5 — MoE Hadamard + `gemv_moe_s` on Real EXL3 Weights (`/tmp/sgemv_gold`)

All seven end-to-end variants verified **`BIT-EXACT` (`bit_diffs=0`, `max_ulp=0`, `max_abs=0.000e+00`)** across all 2,304 output floats against the shipped 8-kernel chain.

**Suite 5A — Isolated kernel timings: PCIe PINNED descriptors vs VRAM descriptors vs vectorized/fused (modeled ×30 `on_dev` visits/tok):**

| Kernel | PINNED descs (µs) | VRAM descs (µs) | Optimized VRAM (µs) | Δ PINNED→VRAM | Δ PINNED→Opt | Save ×30 `on_dev` (ms/tok) |
|---|---:|---:|---:|---:|---:|---:|
| `had_in_moe` GU (`k=2304`) | 17.76 | 3.78 | **2.21** (`had_in_moe_vec`) | −13.98 µs | −15.55 µs | −0.419 → **−0.466** |
| `had_out_moe` GU (`n=1152`) | 11.76 | 2.72 | **1.85** (`had_out_moe_vec`) | −9.04 µs | −9.91 µs | −0.271 → **−0.297** |
| `had_in_moe` DN (`k=1152`) | 9.20 | 2.23 | **2.32*** (`situ_had_in_moe`) | −6.97 µs | −9.61 µs* | −0.209 → **−0.288*** |
| `had_out_moe` DN (`n=2304`) | 16.32 | 2.59 | **2.02*** (`had_out_wsum_moe`) | −13.73 µs | −16.58 µs* | −0.412 → **−0.497*** |
| **Subtotal Hadamard (4k)** | **55.04** | **11.32** | **8.40*** *(incl. situ+wsum)* | **−43.72 µs** | **−51.65 µs*** | **−1.312 → −1.550** |
| `gemv_moe_s` GU (144 CTAs) | 224.73 | 170.74 | *(keep shipped)* | **−53.99 µs** | −53.99 µs | **−1.620** |
| `gemv_moe_s` DN (144 CTAs) | 139.00 | 91.26 | *(keep shipped)* | **−47.74 µs** | −47.74 µs | **−1.432** |

*\*`situ_had_in_moe` (2.32 µs) replaces `situ_pair` (~2.73 µs) + `had_in_moe(DN)` (2.23 µs) = 4.96 µs; `had_out_wsum_moe` (2.02 µs) replaces `had_out_moe(DN)` (2.59 µs) + `expert_wsum` (~2.28 µs) = 4.87 µs.*

**Suite 5B — End-to-end 8-kernel MoE chain (real EXL3 weights, interleaved):**

| Variant | Launches | Exactness | Regs / Stack (`gemv`) | Med (µs) | Min (µs) | Save vs `H0` (×30 `on_dev`) | Save vs `H1` (×48 all) |
|---|---:|---|---|---:|---:|---:|---:|
| `H0`: shipped 8k **[PINNED descs]** (`on_dev` today) | 8 | ref | 40 / 0 B | 422.73 | 408.06 | +0.000 | — |
| **`H1`: shipped 8k [VRAM descs]** (`120 KiB` fix) | 8 | **BIT-EXACT** | 40 / 0 B | **282.61** | **279.81** | **−4.204 ms/tok** | +0.000 |
| `H2`: fuse `had_out` into `gemv_moe_s` (6k) | 6 | BIT-EXACT | 40 / **16 B spill** | 302.34 | 299.39 | −3.612 ms/tok | **+0.947 ms (WORSE)** |
| `H3a`: fuse `had_in+out` into `gemv_moe_s` (4k) | 4 | BIT-EXACT | 40 / **16 B spill** | 300.92 | 298.11 | −3.654 ms/tok | **+0.879 ms (WORSE)** |
| `H3d`: `GU(in+out)` + `situ_hi` + `DN` + `ho_wsum` | 4 | BIT-EXACT | 40 / 16 B spill | 337.66 | 335.05 | −2.552 ms/tok | +2.642 ms (WORSE) |
| **`H3e`: clean 6k (shipped `gemv` untouched!)** | **6** | **BIT-EXACT** | **40 / 0 B** | **277.71** | **274.43** | **−4.351 ms/tok** | **−0.235 ms/tok** |

*Note on `nsys` cross-check: on real Inkling-2-Mini (`/tmp/nsys_mix_on.sqlite`), `ON_DEV` `gemv_moe_s(DN)` is `131.03 µs` vs `CACHED` VRAM-desc `85.35 µs` (`−45.68 µs` vs bench `−47.74 µs`, **96% match**), and `ON_DEV` Hadamards total `42.31 µs` vs `CACHED` VRAM-desc `~11.5 µs` (`−30.8 µs`). Thus `H1` yields **~−3.5 to −4.2 ms/tok** end-to-end in the real runtime.*

---

## 4. Ranked Implementation Plan (Exact Call Sites)

| Rank | Fix | Modeled Save (ms/tok) | Exactness | VRAM Δ | Exact Call Sites |
|---:|---|---:|---|---:|---|
| **1** | **`H1` — VRAM `descs_g/u/d` for `on_dev` (or all) MoE layers** | **−3.5 to −4.20** | **BIT-EXACT** (`bit_diffs=0`) | **+120 KiB** (10 `on_dev`) / +288 KiB (all 24) | `crates/inkwell-run/src/runner.rs:1129–1139` (`mk` allocates `DevBuf` instead of `host_alloc_mapped`), `crates/inkwell-run/src/moe.rs:282–290` (`MoeArgsExl3` stores `DevBuf` or enum `{Dev, Pinned}`), `moe.rs:451,472,585` (pass `.dev_ptr()`), `runner.rs:2151` (prefetch uses same device ptr). Zero CUDA kernel changes. |
| **2** | **`M1b-4` — Fused `mlp.gate` + parallel 4-warp `router` (`gate_router_fused_exact<4>`)** | **−0.46** *(bench)* / **~−0.55** *(nsys)* | **BIT-EXACT** (`ids=8/8, w_diffs=0`) | +516 B scratch (`logits[128] + ticket`) | Replace `t::mmq(c, l.ffn_gate_inp, fcur)` + `router_cb` at `crates/inkwell-run/src/runner.rs:1787–1798` with one custom op calling `gate_router_fused_exact<4>` in `crates/inkwell-cuda/src/kernels.cu` (replacing serial `inkwell_router` at `kernels.cu:4478`). Also add `router_par_128` as fallback when `INKWELL_NO_ROUTERFUSE` is off or for non-BF16 gates. |
| **3** | **`M2b-4` + `M2a-4` — 0-smem `rms2down_nosmem_mr<4>` + 3× batched attention controller triads** | **−0.33** *(up to −0.40 w/ struct args)* | **BIT-EXACT** (`bit_diffs=0`) | 0 B | Replace `inkwell_rms2down` (`kernels.cu`, 18.7 KiB smem) with `rms2down_nosmem_mr<4>` (640 B smem); batch the `j in 0..3` controller loop at `crates/inkwell-run/src/runner.rs:1618–1623` into one custom op launching `rms2down_nosmem_mr<4,3>` + `gf_apply_opt<3>` (pass 3-element pointer struct by value to avoid pointer-chase). Single FFN controller (`runner.rs:1782`) uses `rms2down_nosmem_mr<4,1>` + shipped `inkwell_gf_apply`. |
| **4** | **`H3e` — Clean 6-kernel MoE Hadamard/SITU/wsum chain (shipped `gemv_moe_s` untouched)** | **−0.24** *(+ reduces `ccopy` overlap on cached `hi_gu`)* | **BIT-EXACT** (`bit_diffs=0`) | 0 B | In `crates/inkwell-cuda/src/kernels.cu:3540–3571` (`ink_launch_exl3_moe`) and `3655–3723` (`ink_launch_exl3_moe_cached`): swap `had_in_moe` (GU) → `had_in_moe_vec` (keeping hit/miss predicate for cached path), `had_out_moe` (GU) → `had_out_moe_vec`, `situ_pair` + `had_in_moe` (DN) → `situ_had_in_moe`, and `had_out_moe` (DN) + `expert_wsum` → `had_out_wsum_moe`. Do **not** modify `inkwell_exl3_gemv_moe_s`. |
| **5** | **`M3a` *(Optional, FLAGGED REORDERED)* — `sink_relbias_fused` with BF16 `attn_rel_bias_pad`** | **−0.29** *(bench)* / **~−0.55** *(nsys)* | **REORDERED** (`max_abs=4.77e-7`, `max_rel=1.02e-7` vs cuBLAS K=16) | **−512 KiB** (saves VRAM) | Keep `attn_rel_bias_pad` in BF16 at `crates/inkwell-run/src/runner.rs:963–975` (drop F32 widening + `[1025, 24]` scratch); replace `sinks` + `table` + `relbias_gather` at `runner.rs:1740–1752` with one custom op launching `sink_relbias_fused`. Gate behind standard parity check (16/16 greedy token match). |
| **6** | **`M4-1` — Batched `k/v_sconv.weight_proj`** | **−0.014** *(bench)* / **~−0.020** *(nsys)* | **BIT-EXACT** (`bit_diffs=0`) | 0 B | Batch `k_dyn` + `v_dyn` (`crates/inkwell-run/src/runner.rs:1693–1694`) into one `kv_sconv_batched<1>` launch (`grid=(32, 2)`). Low priority. |

### Total Projected Impact
- **Ranks 1–4 + 6 (100% BIT-EXACT, `bit_diffs=0`):** **−4.5 to −5.25 ms/tok**, **+120 KiB VRAM** → decode **~37.8–38.4 ms/tok**.
- **Ranks 1–6 (adding flagged `M3a`):** **−5.1 to −5.8 ms/tok**, **−392 KiB net VRAM** → decode **~37.2–37.8 ms/tok**.

---

## 5. Reproduction

```bash
opt/kbench/mmvf_had/build_and_run.sh /tmp/sgemv_gold
```

Files in `opt/kbench/mmvf_had/`:
- `mmvf_had_kernels.cuh` — exact `mmvf_bf16_ref` replica + all candidate kernels (Suites 1–5)
- `bench_mmvf_had.cu` — CUDA-Graph interleaved benchmark harness + bit-exactness & ULP verifier
- `build_and_run.sh` — `nvcc` sm_61 build + ptxas reg/smem/spill parser + `flock /tmp/inkwell-gpu.lock` runner
- `REPORT.md` — this report
