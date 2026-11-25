# Greedy-Exact Speculative Decoding Feasibility Study — Inkling-2-Mini on GTX 1060 6 GB (`sm_61`)

**Date:** 2026-11-25  
**Target:** NVIDIA GeForce GTX 1060 6 GB (`sm_61`, 10 SMs, 64K regs/SM, 96 KB smem/SM, ~192 GB/s DRAM, 1.5 MB L2, PCIe 3.0).  
**Baseline:** `9c735bd` (`master` batch-1 greedy decode: **28.3 ms/tok** validate / **28.85 ms/tok** median under `nsys`, peak VRAM **5,032 MiB** vs **5,080 MiB** cap, generated-text `md5 5cbe8e06` at `ctx512`).  
**Goal:** Assess whether greedy-exact speculative decoding (identical output text, `md5 5cbe8e06` at `ctx512`) can reach **≤ 20.0 ms per accepted token** on this GPU.

---

## Executive Verdict

1. **Can greedy-exact speculative decoding reach ≤ 20.0 ms/tok with a training-free drafter? NO.**
   - Even with a **0 ms, 0 MiB oracle drafter**, our silicon-anchored bit-exact ragged verify cost model gives:
     - `B=2` (`k=1` draft): `verify = 40.25 ms` (`1.42×`) → theoretical floor **`20.13 ms/tok`** at 100% acceptance (`emit = 2.00`).
     - `B=3` (`k=2` drafts): `verify = 50.44 ms` (`1.78×`) → requires **`emit ≥ 2.522`** (`76.1%` avg acceptance of both drafts).
     - `B=4` (`k=3` drafts): `verify = 60.05 ms` (`2.12×`) → requires **`emit ≥ 3.002`** (`66.7%` avg acceptance of all 3 drafts).
   - **Why MoE caps the gain:** Although bit-exact `M=8` weight-decode sharing works as hoped on dense EXL3 projections (`0.95–1.16×` at `B=4`, `1.45×` on unembed, **0 bit diffs** across `B=1..8`), **MoE routing overlap between consecutive tokens is low**. On 384 real layer visits (`opt/kbench/ids_block.bin`), distinct experts per visit grow from `8.00` (`B=1`) to `12.89` (`B=2`, `1.61×`), `16.59` (`B=3`, `2.07×`), `19.96` (`B=4`, `2.50×`), and `28.36` (`B=8`, `3.55×`), with **61.5–75.9%** of union experts at `B=2..4` used by only **1 token** (`m=1`). Ragged MoE GEMV therefore scales at **`1.665× / 2.194× / 2.677×`** and exposed PCIe `ccopy` scales at **`1.622× / 2.134× / 2.569×`**.
2. **Best training-free drafter:** Re-quantizing/pruning the checkpoint's shipped DSpark MTP (`~/models/inkling/mtp.safetensors`, `225.6 MiB` EXL3+BF16 → `~120 MiB` int4/dp4a, `~1.45 ms` draft) + bit-exact ragged `B=2` verify reaches **`25.48 ms/tok` (`1.11×`)** — a real ~10% win, but 5.5 ms short of 20 ms/tok. As-shipped DSpark (`~336 MiB`, `45 ms` draft) is **`50.5–63.6 ms/tok`**; even with a fast `4 ms` EXL3 draft it is **`28.85 ms/tok`** because stealing `260 MiB` from the `1,200 MiB` kslot expert cache adds `+6.24 ms/round` of PCIe misses.
3. **Self-speculation (`np=1, epi=24`, 0 MiB extra VRAM)** has surprisingly strong acceptance (`p1 = 0.85`, 5-prompt mean `emit(1..4) = 1.85 / 2.59 / 3.25 / 3.78`; Rust code `P3` hits `emit(4) = 4.875`), **but** 24 layer visits still cost **`15.21 ms/draft`** (53.8% of full decode), yielding **`29.98 ms/tok`** on 5-prompt mean (`26.42 ms/tok` on best-case code). Dropping the epilogue (`noepi`) collapses `p1` to `0.06–0.25`.
4. **Prompt-lookup n-gram (`min_n=2, k=1`, gated)** is **`27.97 ms/tok` (`1.01×`)** on 5-prompt mean (0 MiB VRAM, only beats serial by ~5–7% on repetitive prose/refactor).
5. **What CAN approach ≤ 20 ms/tok:** A **trained** ultra-light EAGLE-1 drafter (`~30 MiB`, `0.23 ms/step`, `opt/kbench/drafter/REPORT.md`) + bit-exact ragged `B=3` (`k=2`) verify + n-gram hybrid reaches **`21.1–22.0 ms/tok`**, and crossing **≤ 20.0 ms/tok** requires pairing that with `~3–5 ms` of verify-kernel cuts (folding the `LDG.128` 4-tile interleave + 2-word coverage wins from `5f5efbe` into the batched MoE kernel + overlapping PCIe staging).

---

## 1. Verify Cost Model (`B = 2, 3, 4, 6, 8`)

### 1.1 Exact Batch-1 Baseline Breakdown (`9c735bd`, `nsys` Step 59)

Captured on `9c735bd` (`md5 5cbe8e06`, `ch2nopless: enabled`, `INKWELL_MAX_CTX=512`, `kslot` cache `940/1200 MiB`). Step 59 (complete 48-visit bucket-boundary capture, 2,338 kernels) measures **28.131 ms** GPU union time across concurrent streams (`34.590 ms` raw kernel sum). Steady-state graph-replay decode over 56 tokens measures **28.85 ms** median (**28.30 ms** on master `validate.sh`, `7.0 cudaGraphLaunch/tok`), giving **~0.72 ms/tok** host + graph-launch overhead:

| Component | Step 59 Raw (ms) | Step 59 Wall (ms) | Normalized (`28.30 ms`) | Share | Primary Kernels |
|---|---:|---:|---:|---:|---|
| EXL3 MoE GEMV | 10.551 | 10.551 | **10.35** | 36.6% | `inkwell_exl3_gemv_moe_s` (114 launches) |
| MoE PCIe staging (`ccopy`) | 3.471 | 1.922 | **1.89** | 6.7% | `inkwell_exl3_moe_ccopy` (1.55 ms overlapped) |
| MoE non-GEMV | 1.281 | 1.281 | **1.26** | 4.4% | `had_in/out_moe`, `kslot_resolve_w`, `wsum`, `situ_pair` |
| Dense EXL3 (48 visits) | 10.881 | 5.972 | **5.86** | 20.7% | `inkwell_exl3_gemv_sf_ch2` (288) + `had_in` (concurrent) |
| Unembed EXL3 (`V=201,024`) | 1.444 | 1.444 | **1.42** | 5.0% | `inkwell_exl3_gemv_sf_ch2` (1× K4.0) |
| Attention (ctx512) | 1.924 | 1.924 | **1.89** | 6.7% | `attn_decode_gqa3_exact` (0.84), `sgemm_32x32` (0.45), `clip_l2`, `set_rows`, `relbias` |
| Controller + mix + conv + norm | 4.132 | 4.132 | **4.05** | 14.3% | `mix_fused` (1.11), `gf_apply` (0.64), `rms2down` (0.62), `rms_norm` (0.57), `concat_cont` (0.49), `conv_step` (0.40), `k_bin_bcast` (0.30) |
| Router + small GEMVs + misc | 0.906 | 0.906 | **0.89** | 3.1% | `gate_router_fused_exact`, `mul_mat_vec_f`, `argmax` |
| Host + graph launch | — | 0.720 | **0.71** | 2.5% | `7.0 cudaGraphLaunch/tok` + D2H argmax |
| **Total (`B=1` serial)** | **34.590** | **28.851** | **28.30** | **100%** | `md5 5cbe8e06` |

### 1.2 Measured Bit-Exact `M=8` Batched EXL3 Scaling (`0` Bit Diffs)

Standalone `sm_61` kbench (`/tmp/spec_gemv_b.cu`), matching the exact `fmaf` order of shipped `inkwell_exl3_gemv_moe_s` / `gemv_sf_ch2` with a rolling 16-tile `xs[B][256]` shared-memory slab (≤ 8 KB even at `B=8`), sharing trellis hash + `I2F` across all `B` rows:

| Shape | `B=1` (µs) | `B=2` (`×B1`, sp) | `B=3` (`×B1`, sp) | `B=4` (`×B1`, sp) | `B=6` (`×B1`, sp) | `B=8` (`×B1`, sp) | Bit diffs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MoE GU K2.0 (`tk=144, tn=72, sl=16`) | 176.7 | 207.3 (`1.17×`, 1.70×) | 237.6 (`1.34×`, 2.23×) | 242.6 (`1.37×`, 2.91×) | 316.4 (`1.79×`, 3.35×) | 390.1 (`2.21×`, 3.62×) | **0** |
| MoE GU K3.0 (`tk=144, tn=72, sl=16`) | 228.4 | 249.9 (`1.09×`, 1.83×) | 253.3 (`1.11×`, 2.70×) | 273.2 (`1.20×`, 3.34×) | 358.2 (`1.57×`, 3.82×) | 423.8 (`1.86×`, 4.31×) | **0** |
| MoE DN K2.0 (`tk=72, tn=144, sl=8`) | 93.0 | 106.5 (`1.15×`, 1.75×) | 122.3 (`1.32×`, 2.28×) | 126.0 (`1.35×`, 2.95×) | 165.6 (`1.78×`, 3.37×) | 204.8 (`2.20×`, 3.63×) | **0** |
| Dense `wq` K3.0 (`tk=144, tn=192`) | 66.6 | 71.5 (`1.07×`, 1.86×) | 61.3 (`0.92×`, 3.26×) | 64.5 (`0.97×`, 4.13×) | 80.9 (`1.22×`, 4.94×) | 104.3 (`1.57×`, 5.11×) | **0** |
| Dense `wo` K3.5 (`tk=192, tn=144`) | 79.5 | 84.0 (`1.06×`, 1.89×) | 65.5 (`0.82×`, 3.64×) | 75.8 (`0.95×`, 4.20×) | 77.8 (`0.98×`, 6.13×) | 116.7 (`1.47×`, 5.45×) | **0** |
| Unembed K4.0 (`tk=144, tn=12568`) | 2066.4 | 2304.0 (`1.11×`, 1.79×) | 2558.7 (`1.24×`, 2.42×) | 2991.1 (`1.45×`, 2.76×) | 3671.0 (`1.78×`, 3.38×) | 4531.2 (`2.19×`, 3.65×) | **0** |

Weighting MoE by the model's K-mix (`56.1%` K1.5/K2 + `43.9%` K2.5–K4, `5f5efbe`) yields the bit-exact per-expert multiplicity time curve:
$$t_{\text{EXL3}}(m) = \{1: 1.000,\; 2: 1.138,\; 3: 1.241,\; 4: 1.295,\; 5: 1.494,\; 6: 1.693,\; 7: 1.873,\; 8: 2.053\}$$

### 1.3 Measured Real-Decode MoE Routing Overlap (`opt/kbench/ids_block.bin`, 384 Layer Visits)

Each record in `opt/kbench/ids_block.bin` stores `[layer_id, 8 tokens × 8 top-k expert IDs]` from real decode across all 24 physical layers:

| `B` | Mean Union `U(B)` (All 48) | `U(B)/8` | Streamed `U_s(B)/8` (18 visits) | Streamed Loop `L16..17` `U/8` | Avg Mult `8B/U` | Multiplicity Dist `P(m\|B)` | **Ragged MoE Mult** | **Padded MoE Mult** |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| **1** | 8.00 | 1.000× | 1.000× | 1.000× | 1.00 | `1:1.000` | **1.000×** | 1.000× |
| **2** | 12.89 | 1.612× | 1.622× | 1.505× | 1.24 | `1:0.759, 2:0.241` | **1.665×** | 1.835× |
| **3** | 16.59 | 2.074× | 2.134× | 1.909× | 1.45 | `1:0.662, 2:0.229, 3:0.108` | **2.194×** | 2.575× |
| **4** | 19.96 | 2.495× | 2.569× | 2.203× | 1.60 | `1:0.615, 2:0.232, 3:0.090, 4:0.064` | **2.677×** | 3.232× |
| **6** | 24.97 | 3.121× | 3.185× | 2.703× | 1.92 | `1:0.537, 2:0.236, 3:0.100, 4+:0.127` | **3.480×** | 5.284× |
| **8** | 28.36 | 3.545× | 3.582× | 2.997× | 2.26 | `1:0.475, 2:0.238, 3:0.108, 4+:0.179` | **4.124×** | 7.279× |

Ragged MoE multiplier is $(U(B)/8) \sum_{m=1}^B P(m|B)\,t_{\text{EXL3}}(m)$: each union expert runs at its actual consumer count $m$. Even with ideal ragged dispatch, `61.5–75.9%` of experts at `B=2..4` have `m=1`, so weight-decode sharing cannot amortize the union growth.

### 1.4 Per-Component Verify Cost Table (Normalized to `28.30 ms/tok` Baseline)

Scaling assumptions for non-MoE components (assuming batched exact kernels):
- **Dense layer EXL3** (`5.86 ms`): conservative stream-concurrent scaling `1.07× / 1.12× / 1.16× / 1.32× / 1.55×` (single-launch kbench is `0.87–1.07×` at `B=2..4`, but `B=1` already fills SMs via concurrent Q/K/V/gate/rel streams).
- **Unembed EXL3** (`1.42 ms`): directly from kbench (`1.11× / 1.24× / 1.45× / 1.78× / 2.19×`).
- **Attention** (`1.89 ms`): batched causal kernel reads KV once per `B` queries (`1 + 0.30(B-1)` at ctx512; current `step_block` serial loop scales `~0.95B`).
- **Controller BF16 (`1.26 ms`) & Router (`0.89 ms`)**: shared BF16 weights across `B` rows (`1 + 0.15(B-1)`).
- **Mix/norm/concat/bcast (`2.47 ms`)**: memory-bound over `[2304, B]` (`1 + 0.50(B-1)`).
- **Depthwise conv (`0.40 ms`)**: sequential dependency across `B` positions (`0.35 + 0.65B` fused).
- **Host + graph launch (`0.71 ms`)**: fixed per verify round (`0.71 + 0.08(B-1)`).

| `B` | MoE GEMV | Exposed `ccopy` | MoE Other | Dense Lyr | Unembed | Attn | Ctrl/Mix/Conv | Router/Misc | Host | **Verify Total (Ragged)** | **`×B1`** | **ms / row** | Padded Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 10.35 | 1.89 | 1.26 | 5.86 | 1.42 | 1.89 | 4.05 | 0.89 | 0.71 | **28.30 ms** | 1.00× | 28.30 | 28.30 ms |
| **2** | 17.23 | 3.06 | 2.14 | 6.27 | 1.57 | 2.45 | 5.71 | 1.04 | 0.78 | **40.25 ms** | **1.42×** | 20.13 | 42.01 ms |
| **3** | 22.71 | 4.02 | 2.95 | 6.56 | 1.76 | 3.02 | 7.36 | 1.20 | 0.86 | **50.44 ms** | **1.78×** | 16.81 | 54.39 ms |
| **4** | 27.71 | 4.85 | 3.75 | 6.80 | 2.05 | 3.59 | 9.01 | 1.36 | 0.94 | **60.05 ms** | **2.12×** | 15.01 | 65.79 ms |
| **6** | 36.02 | 6.00 | 5.25 | 7.73 | 2.52 | 4.72 | 12.32 | 1.67 | 1.10 | **77.33 ms** | **2.73×** | 12.89 | 96.00 ms |
| **8** | 42.68 | 6.75 | 6.66 | 9.08 | 3.10 | 5.85 | 15.62 | 1.98 | 1.26 | **92.99 ms** | **3.29×** | 11.62 | 125.64 ms |

### 1.5 Audit of Existing Batched (`NT=8`) Path in `runner.rs` & `kernels.cu`

An `NT=8` verify path already exists (`StepBlk` / `Runner::step_block` in `crates/inkwell-run/src/runner.rs:3050–3569`, `ink_launch_exl3_gemv_ntok` at `kernels.cu:3093`, `ink_launch_exl3_moe_ntok` at `kernels.cu:5471`), with three critical findings:
1. **Currently broken on HEAD `9c735bd`:** Running `VTEST8`, `INKWELL_SPEC=7`, or `INKWELL_SPEC=mtp` aborts at `ggml.c:1868: GGML_ASSERT(view_size + view_offs <= ggml_nbytes(view_src))` because `Runner::control()` (`runner.rs:1956–1988`) hardcodes `[n, 1]` output shape in the newer `rms2down`/`gf_apply` fusions when called on `[n, NT]` tensors (`INKWELL_NO_CTRLFUSE=1` is required to bypass).
2. **Not bit-exact with batch-1 (FP order diverges):** Both `inkwell_exl3_gemv_dp4a_ntok` and `inkwell_exl3_gemv_moe_ntok` use the legacy 4-thread-per-column ring-in-smem kernel with `nky` split-K **`atomicAdd`** (`kernels.cu:5634`), unlike batch-1's `M=8` `SHF.L.W` warp-tiled `fmaf` order. In `VTEST8` (`opt/ledger.md:85–93`), this yields `max|d| ~ 1e-4..1e-3` with occasional **`1.2–1.4` fp-chaos spikes** when a near-tie MoE router flips, causing rare greedy argmax divergence (`" Canada" → " Mexico"` at margin 0.15). **Therefore, reusing the legacy `ntok` kernels violates strict greedy-exactness (`md5 5cbe8e06`) and would require a KLD gate; a new `M=8` bit-exact batched kernel (matching `/tmp/spec_gemv_b.cu`) is mandatory for bitwise-identical greedy output.**
3. **Slow (`~498–570 ms` per 8-token block, `opt/ledger.md:94–103`):** Dominated by legacy `gemv_moe_ntok` (`358 ms`, 73.5%) and `gemv_dp4a_ntok` (`47 ms`), plus unbatched per-token loops for attention/conv (`~54 ms`) and fixed `NT=8`.

---

## 2. Training-Free Drafters: Inspection & Measured Acceptance

Evaluated on 5 diverse prompts (64 generated tokens each, dumped to `/tmp/spec_gen_p1..5.txt`):
- **P1** (prose capitals): `"The capital of France is"`
- **P2** (Python): `def merge_sort(arr): ...`
- **P3** (Rust): `pub fn binary_search(slice: &[i32], target: i32) -> Option<usize> {`
- **P4** (numbered list): `"Write a numbered list of 5 steps to set up a local git repository..."`
- **P5** (Python refactor): `"Refactor the following Python function to use type hints..."`

### (a) Prompt-Lookup / N-Gram Drafting (`0 ms`, `0 MiB`)

Evaluated offline on the exact greedy token streams of P1–P5 (searching backwards in `prompt + generated` for the most recent suffix match of length `n ∈ [min_n..4]`):

| Setting | `k=1` (`B=2`) | `k=2` (`B=3`) | `k=3` (`B=4`) | `k=4` (`B=5`) | `k=7` (`B=8`) |
|---|---:|---:|---:|---:|---:|
| `min_n=1` mean `emit/rnd` (match rate) | 1.239 (61%) | 1.310 (59%) | 1.391 (56%) | 1.404 (56%) | 1.411 (56%) |
| `min_n=2` mean `emit/rnd` (match rate) | 1.130 (27%) | 1.193 (24%) | 1.207 (23%) | 1.212 (22%) | 1.218 (22%) |
| **Gated `min_n=2` eff. ms/tok** (runs `B=1` on miss) | **27.97 ms** (`1.01×`) | 28.19 ms (`1.00×`) | 29.48 ms (`0.96×`) | 30.86 ms (`0.92×`) | 34.98 ms (`0.81×`) |
| **Gated `min_n=1` eff. ms/tok** | 28.84 ms (`0.98×`) | 31.75 ms (`0.89×`) | 33.58 ms (`0.84×`) | 36.50 ms (`0.78×`) | 45.32 ms (`0.62×`) |

Per-prompt best (`min_n=2, k=1`): P5 refactor `27.38 ms`, P2 Python `27.44 ms`, P4 list `27.91 ms`, P1 prose `28.40 ms` (`26.37 ms` at `k=2`), P3 Rust `28.74 ms`. On general text/code without long verbatim copies, n-gram drafting is essentially break-even (`~1.01×`).

### (b) Tiny Draft Model for This Tokenizer

Checked `~/models/` and `~/models/inkling/`: **none exists**. Inkling-2-Mini uses a custom 200,058-token multilingual BPE vocabulary (padded to `201,024`, leading-space English words in `149,860..199,999`), incompatible with Llama/Qwen/GPT-2 tokenizers.

### (c) Self-Speculation by Fewer Loop Visits / Early Exit (`0 MiB` Extra VRAM)

Inkling-2-Mini runs 6 prelude layers (`0..5`), `np=3` passes over 12 loop layers (`6..17`), and 6 epilogue layers (`18..23`) = **48 visits/token**. We measured both **graphs-ON reduced-pass timing** (`INKWELL_GENS=32`) and **4-step autoregressive acceptance** against full 48-visit greedy decode (`16` windows × 5 prompts = `80` samples per config):

| Mode | Visits | Measured ms/tok (`32 tok`) | Norm (`28.30 ms`) | P1 `emit(1..4)` | P2 `emit(1..4)` | P3 `emit(1..4)` | P4 `emit(1..4)` | P5 `emit(1..4)` | **5-Prompt Mean `emit(k=1..4)`** |
|---|---:|---:|---:|---|---|---|---|---|---|
| `np3_epi48` (full) | 48 | 30.06 ms | 28.30 ms | — | — | — | — | — | baseline |
| `np2_epi36` | 36 | 21.87 ms | 20.59 ms | 1.88/2.63/3.13/3.44 | 1.94/2.88/3.69/4.50 | 2.00/3.00/4.00/4.88 | 1.81/2.50/3.00/3.50 | 1.88/2.69/3.38/4.00 | **1.900 / 2.738 / 3.438 / 4.063** |
| `np1_epi24` | 24 | 16.16 ms | 15.21 ms | 1.88/2.56/3.00/3.13 | 1.81/2.50/3.19/3.88 | 2.00/3.00/4.00/4.88 | 1.75/2.38/2.88/3.19 | 1.81/2.50/3.19/3.81 | **1.850 / 2.587 / 3.250 / 3.775** |
| `np0_epi12` | 12 | 9.19 ms | 8.65 ms | 1.44/1.75/1.88/1.88 | 1.56/1.81/2.00/2.06 | 1.69/2.19/2.50/2.69 | 1.56/1.94/2.25/2.31 | 1.63/2.13/2.56/2.88 | **1.575 / 1.963 / 2.237 / 2.362** |
| `np2_noepi30` | 30 | 18.60 ms | 17.51 ms | 1.19/1.19/1.19/1.19 | 1.19/1.19/1.19/1.19 | 1.06/1.06/1.06/1.06 | 1.25/1.25/1.25/1.25 | 1.13/1.13/1.13/1.13 | 1.163 / 1.163 / 1.163 / 1.163 |
| `np1_noepi18` | 18 | 13.09 ms | 12.32 ms | 1.13/1.13/1.13/1.13 | 1.19/1.25/1.25/1.25 | 1.19/1.25/1.31/1.31 | 1.06/1.06/1.06/1.06 | 1.13/1.13/1.13/1.13 | 1.138 / 1.163 / 1.175 / 1.175 |

**Key findings:**
- The 6 epilogue layers (`18..23`) are **non-negotiable**: skipping them drops `p1` from `0.85` to `0.14`.
- `np1_epi24` achieves remarkable `p1 = 0.85` (`100%` on Rust P3) with 0 MiB extra VRAM, **but** at `15.21 ms` per serial draft token, `k` draft steps + `verify(k+1)` still exceeds `28.30 ms/tok` on average (`29.98 ms/tok` at `k=1`). Even if `k=1` reused the anchor's prelude+pass-0 computation (~11 ms saved on the first draft only), it would land at `~24.5 ms/tok`, still far above 20 ms/tok.

### (d) Checkpoint MTP Heads (`~/models/inkling/mtp.safetensors`)

`model.safetensors.index.json` has no MTP tensors, **but `~/models/inkling/mtp.safetensors` ships a full DSpark MTP head (`225.60 MiB`)**, already wired in `crates/inkwell-run/src/draft.rs`:
- **3 dense EXL3 transformer layers** (`layers.0..2`: `H=2304, FF=9216`, `q/k/v/o` + `k_proj_ctx/v_proj_ctx`, `4×4 fuse_weight` over 4 target taps) = **`125.1 MiB`**.
- **Markov GRU head**: `rnn_proj` (`1.88 MiB`), `hidden_out` (`0.28 MiB`), and **`markov_w1` BF16 `[201024, 256]` = `98.16 MiB`**.
- **Runtime overhead in `draft.rs`:** `kc[3]/vc[3]` F32 KV cache (`24 MiB`) + per-weight EXL3 scratch (`~85 MiB`) → **`~336 MiB` total VRAM**.
- **Measured acceptance (`opt/ledger.md:81–139`):** `p1 ≈ 0.75`, continuation `q ≈ 0.46` → `emit(k=1/2/3/7) ≈ 1.75 / 2.10 / 2.25 / 2.45` (verified by numpy oracle; prose `1.86`, code `2.39`, tech `1.64` at `k=7`).
- **Critical VRAM penalty:** Every MiB stolen from the `1,200 MiB` kslot expert cache costs `+0.0148 ms/MiB × (U_stream(B)/8)` in extra PCIe misses (`opt/ledger.md:436–438`). Loading `336 MiB` of DSpark adds **`+8.1 ms` (`B=2`) to `+17.8 ms` (`B=8`)** per verify round!

---

## 3. Combined Estimate (`ms / accepted token`) & Recurrent State Rollback

### 3.1 Expected `ms / accepted token` Across Drafters × `B` (Baseline `28.30 ms/tok`)

$$\text{ms/tok} = \frac{T_{\text{draft}}(k) + T_{\text{verify}}^{\text{ragged}}(B) + 0.0148 \cdot \Delta\text{VRAM}_{\text{MiB}} \cdot (U_{\text{stream}}(B)/8)}{\text{emit}(k)}$$

| Drafter | Extra VRAM | `k` (`B`) | Draft (ms) | Verify + VRAM Pen (ms) | `emit(k)` | **Eff. ms/tok** | **Speedup** | ≤ 20 ms? |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Serial baseline (`9c735bd`)** | 0 MiB | 0 (1) | 0.00 | 28.30 | 1.000 | **28.30** | 1.00× | — |
| N-gram prompt-lookup (`min_n=2`, gated) | 0 MiB | 1 (2) | 0.02 | 28.30 / 40.25 | 1.130 | **27.97** | 1.01× | No |
| N-gram prompt-lookup (`min_n=2`, gated) | 0 MiB | 2 (3) | 0.02 | 28.30 / 50.44 | 1.193 | 28.19 | 1.00× | No |
| Self-spec `np0_epi12` (12 visits, mean) | 0 MiB | 1 (2) | 8.65 | 40.25 | 1.575 | 31.05 | 0.91× | No |
| Self-spec `np1_epi24` (24 visits, mean) | 0 MiB | 1 (2) | 15.21 | 40.25 | 1.850 | 29.98 | 0.94× | No |
| Self-spec `np1_epi24` (24 visits, mean) | 0 MiB | 2 (3) | 30.42 | 50.44 | 2.587 | 31.25 | 0.91× | No |
| Self-spec `np1_epi24` (best-case Rust P3) | 0 MiB | 3 (4) | 45.63 | 60.05 | 4.000 | 26.42 | 1.07× | No |
| DSpark MTP as-shipped (`draft.rs`) | +336 MiB | 2 (3) | 45.00 | 50.44 + 10.61 | 2.100 | 50.50 | 0.56× | No |
| DSpark MTP fast EXL3 (`k`-step) | +260 MiB | 1 (2) | 4.00 | 40.25 + 6.24 | 1.750 | 28.85 | 0.98× | No |
| **DSpark MTP pruned/int4 + `dp4a` head** | +120 MiB | **1 (2)** | **1.45** | **40.25 + 2.88** | **1.750** | **25.48** | **1.11×** | **No** |
| DSpark MTP pruned/int4 + `dp4a` head | +120 MiB | 2 (3) | 1.70 | 50.44 + 3.79 | 2.100 | 26.63 | 1.06× | No |
| *[Trained]* EAGLE-1-Med (`p1=0.82, q=0.65`) | +30 MiB | 1 (2) | 0.23 | 40.25 + 0.72 | 1.820 | 22.64 | 1.25× | No |
| *[Trained]* EAGLE-1-Med (`p1=0.82, q=0.65`) | +30 MiB | **2 (3)** | **0.46** | **50.44 + 0.95** | **2.353** | **22.04** | **1.28×** | No |
| *[Trained]* EAGLE-1-Med + N-gram (`0.85/0.72`) | +30 MiB | **2 (3)** | **0.46** | **50.44 + 0.95** | **2.462** | **21.06** | **1.34×** | Close |
| *0 ms / 0 MiB oracle floor (`B=3`)* | 0 MiB | 2 (3) | 0.00 | 50.44 | ≥ 2.522 | **≤ 20.00** | ≥ 1.42× | Floor |

### 3.2 Loop-MoE Recurrent State & Rollback Audit

Inspected every persistent buffer in `crates/inkwell-run/src/runner.rs` (confirmed by `Runner::save_state` at `runner.rs:3322`):

| Buffer | Shape / Count | Persistent Across Tokens? | Rollback Mechanism | Rollback Cost |
|---|---|---|---|---|
| **Loop-MoE residual `blocks` & `bank`/`msgs`** | Ephemeral graph tensors (`[2304, B]`) | **NO — purely intra-token!** Rebuilt from `norm(embd)` each step (`runner.rs:2455, 3495`). | None needed | **0 B, 0.00 ms** |
| **KV arenas (`k_arena`, `v_arena`)** | 48 slots × F16 `[128, cap, 8]` (8 full @ 512, 40 SWA @ 1024) | **YES** | Rejected slots `pos+emit..pos+B-1` are masked out by `n_kv_raw` and overwritten next round. | **0 B, 0.00 ms** |
| **Depthwise `conv_state[48]`** | 48 slots × F32 `[3, 6656]` = **`3.66 MiB`** total (`k/v/attn_post/ffn_post`) | **YES** | Already implemented (`runner.rs:1848–1868`): `conv_step` saves post-position state to `conv_snap[slot][j]`; on partial accept, `rollback_conv(emit)` runs 48 async D2D copies (`3.66 MiB`). | **~0.02 ms** (`3.66 MiB @ 192 GB/s`) |
| **MTP `kc[3]/vc[3]`** (only if MTP used) | 3 × 2 × F16 `[128, 1024, 8]` = `12 MiB` | **YES** (appended only for accepted tokens) | Only call `ctx_append` for the `emit` accepted tokens. | **0.00 ms** |

**Bottom line:** Loop-MoE controller banks and message states do **not** complicate speculative rollback at all. Only `conv_state` needs a `3.66 MiB` D2D copy (~20 µs).

---

## 4. VRAM Budget (`5,032 MiB` Current vs `5,080 MiB` Cap)

Only **`48 MiB`** of free headroom exists before any allocation steals from the `1,200 MiB` kslot expert cache (`used_pre_cache = 3,774 MiB`, `kslot = 1,200 MiB`, `used_post_cache = 4,974 MiB`, runtime peak `5,032 MiB`).

With **shared dense EXL3 scratch** (one reusable buffer sized for unembed instead of today's per-weight `Exl3Blk::scratch`) and **`B-1` `conv_snap` slots** (position `B-1` is already in `conv_state`):

| Component | `B=2` | `B=3` | `B=4` | `B=6` | `B=8` | Notes |
|---|---:|---:|---:|---:|---:|---|
| `conv_snap[48]` (`B-1` × 3.66 MiB) | 3.7 MiB | 7.3 MiB | 11.0 MiB | 18.3 MiB | 25.6 MiB | Rollback snapshots |
| `VScratch` MoE (`8B` × 32 KB) | 0.5 MiB | 0.7 MiB | 1.0 MiB | 1.5 MiB | 2.0 MiB | `xh_e/gu_e/hid_e/dn_e` + index arrays |
| Shared dense EXL3 scratch (`B*(k+n)` max) | 1.6 MiB | 2.4 MiB | 3.2 MiB | 4.8 MiB | 6.4 MiB | Must replace per-weight `Exl3Blk::scratch` |
| ggml `sched` graph activations + `logits[V,B]` | ~6.0 MiB | ~12.0 MiB | ~18.0 MiB | ~30.0 MiB | ~42.0 MiB | `[2304, B]` intermediates + `0.77B` logits |
| KV staging | 0.0 MiB | 0.0 MiB | 0.0 MiB | 0.0 MiB | 0.0 MiB | Writes directly into `k_arena`/`v_arena` |
| **Total Verify Delta** | **~11.8 MiB** | **~22.4 MiB** | **~33.2 MiB** | **~54.6 MiB** | **~76.0 MiB** | **`B=2..4` fit inside 48 MiB headroom!** |
| **Projected Peak (0 MiB drafter)** | **5,044 MiB** | **5,054 MiB** | **5,065 MiB** | 5,087 MiB* | 5,108 MiB* | *`B=6,8` trim `16–32 MiB` from kslot |

Adding a `30 MiB` EAGLE-1 drafter at `B=3` requires `~52 MiB` total → trims only `~8–12 MiB` from kslot (`+0.3 ms/round`). In contrast, `mtp.safetensors` (`225.6 MiB` weights + scratch/KV) forces `~260–336 MiB` out of kslot (`+6–11 ms/round`).

---

## 5. Ranked Implementation Plan & Risks

### Ranked Roadmap (Ordered by ROI & Risk)

1. **P0 — Do NOT build `NT=8` or self-speculation for throughput; first land remaining batch-1 exact wins (`5f5efbe`, ~1.0–1.4 ms/tok)** if not yet merged (`LDG.128` 4-tile interleave + 2-word coverage on K2.5–K4), because those exact kernel improvements also directly lower `verify(B)`'s MoE GEMV by `~10–14%` (`-2.5 to -3.5 ms` at `B=3`).
2. **P1 — Bit-exact ragged `B=2..4` verify engine (prerequisite for ALL drafters):**
   - **`crates/inkwell-cuda/src/kernels.cu`:**
     - Add `inkwell_exl3_gemv_sf_ch2_b<B>` (`B ∈ {2,3,4}`): exact `M=8` `fmaf` order with rolling 16-tile `xs[B][256]` smem slab (proven 0 bit diffs in `/tmp/spec_gemv_b.cu`).
     - Add `inkwell_exl3_gemv_moe_s_ragged<B>`: dispatches each union expert `u` with its exact consumer count `nc = estart[u+1] - estart[u]` (`1..B`), running the exact `M=8` loop (NO `atomicAdd`, NO legacy 4-thread reduction).
     - Add batched exact variants for `rms2down` + `gf_apply` (fixes the `GGML_ASSERT` crash in `Runner::control()`), `mix_fused`, `gate_router_fused_exact`, `attn_decode_gqa3_exact` (causal `B`-query kernel), and fused `B`-step `conv_step` with `conv_snap` write.
   - **`crates/inkwell-run/src/runner.rs` & `crates/inkwell-run/src/moe.rs`:**
     - Parameterize `StepBlk` / `step_block` over runtime/specialized `B ∈ {2, 3, 4}` (replacing hardcoded `const NT: usize = 8`).
     - Replace per-weight `Exl3Blk::scratch` with one shared scratch buffer (saves >300 MiB).
     - Add CUDA Graph caching for `step_block(B)` (mirroring `step_gcache`) and verify **0 bit diffs** against serial `step()` via `VTEST` across `ctx512` (`md5 5cbe8e06`).
3. **P2 — If strictly training-free is required:** Prune/re-quantize `mtp.safetensors` offline (1–2 layers + int4-g128 `V'=32K` frequency-ranked head, `≤ 80–120 MiB`, `dp4a` CUDA-graph draft in `~1.4 ms`) + `B=2` verify → **`~25.5 ms/tok` (`1.11×`)**. (Optionally add gated `min_n=2` n-gram fallback for copy-heavy prompts.)
4. **P3 — To approach/cross ≤ 20.0 ms/tok (requires training):** Train the `30 MiB` `EAGLE-1-Med` drafter (`opt/kbench/drafter/REPORT.md`: `1L D=1024 FF=2816` int8 `dp4a` + tied `V'=32,768` int4-g128 head, `0.46 ms` at `k=2`), run on the P1 bit-exact ragged `B=3` verifier + P0 `LDG.128` MoE kernel + PCIe staging overlap → **`~19.5–21.5 ms/tok` (`1.32–1.45×`)**, preserving exact `md5 5cbe8e06` greedy output.

### Key Risks & Mitigations
- **FP-order exactness:** Any `atomicAdd` split-K (`nky > 1`) or changed FMA tree breaks bitwise equality with batch-1 and can flip near-tie MoE routers (`max|d| ~ 1.4` seen in `VTEST8`). Keep `nky = 1` or deterministic reduction matching batch-1's exact `M=8` `fmaf` order, and gate every batched kernel with a 0-bit-diff `VTEST` harness.
- **SWA ring wrap in `step_block`:** `runner.rs:3421` currently clamps `kv_idx_swa` at `self.swa - 1` instead of modular ring indexing for `pos >= 1024`; fix before testing `ctx > 1024`.
- **VRAM cliff at `5,080 MiB`:** Never let drafter + verify scratch silently evict >50 MiB of kslot cache without accounting for the `+0.0148 ms/MiB × U_stream(B)/8` PCIe miss tax.
