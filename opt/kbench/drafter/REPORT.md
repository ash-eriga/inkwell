# Speculative Decoding Drafter Architecture, Cost Model & Pascal `sm_61` Microbenchmarks

**Target Hardware:** NVIDIA GeForce GTX 1060 6 GB (`sm_61`, 10 SMs, 64K regs/SM, 96 KB smem/SM, ~204 GB/s DRAM, PCIe 3.0 ~8.6 GB/s staged ccopy, no tensor cores).  
**Target Model:** `thinkingmachines/Inkling-2-Mini` (SAGE REXL3 2.26 bpw, 24 layers, 128 experts, top-8 routing, 48 visits/token, `H = 2304`, `V = 200,058`).  
**Serial Baseline (measured, master `952e1ff`):** **46.8 ms/token (21.37 tok/s)** (re-confirmed on this worktree: 46.5–48.7 ms/tok across 5 × 128-token prompts).  
**Quality Rule:** Greedy argmax verification against the target model — lossy drafter kernels (int8/int4 weights, int8/int16 activations, reduced vocabulary) **never change output text**; their only cost is lower acceptance.

---

## Executive Verdict & Plain Honesty Check

1. **The drafter is NOT the bottleneck once moved to `sm_61` `dp4a`/`dp2a` inside a single CUDA Graph.** Hand-tuned kernels (`drafter_kernels.cuh`, zero host sync, device-side argmax) run a full 1–3 token draft round in **0.15–0.99 ms** (a **50–180× speedup** over today's 45–50 ms DSpark path), achieving **160–176 GB/s (78–86% of 204 GB/s DRAM peak)** at 38–40 regs, 0 spills, 6 blocks/SM.
2. **The binding constraint is MoE expert-union growth + PCIe staging inside `verify(k)`.** On real routing traces (`opt/kbench/ids_block.bin`, 384 layer visits), the mean union of distinct experts per visit grows from `8.00` (`B=1`) to `12.89` (`B=2`, `1.61×`), `16.59` (`B=3`, `2.07×`), `19.96` (`B=4`, `2.50×`), and `22.91` (`B=5`, `2.86×`), with **61–76%** of union experts hit by only 1 token. Even with an **ideal ragged (per-multiplicity) batched MoE verify kernel** (`M=16-Batch C0c`):
   - **Can speculation beat serial on this GPU?** **YES, by +12–17% (24.0–25.1 tok/s), PROVIDED the verifier implements compact ragged-batched `B = 2..3` (`k = 1..2`) verification** (today's verifier is fixed `NT = 8` and padded):
     - **`k = 1` (`B = 2` verify rows, modeled `verify = 69.92 ms = 1.49×` serial)** beats serial right now even under the prompt's strict acceptance curve (`P(d1)=0.75 -> emit(1)=1.750`): break-even drafter budget is **`+11.98 ms`**, +10% budget is **`+4.53 ms`**, yielding **`24.14 tok/s` (+13.0%)** with our recommended `0.24 ms` / `29.9 MiB` drafter.
     - **`k = 2` (`B = 3` verify rows, modeled `verify = 90.39 ms = 1.93×` serial)** **CANNOT beat serial** under the strict prompt continuation rate (`p1=0.75, q=0.18 -> emit=1.885`; break-even budget **`−2.17 ms`** — negative even for a 0 ms drafter!). Under the higher ledger empirical curve (`emit=2.10`), `k=2` barely beats serial (`+2.2%`, `21.84 tok/s`, budget `+7.66 ms` for 0%, `−1.26 ms` for +10%). Under a stronger trained EAGLE drafter (`p1=0.82, q=0.65 -> emit=2.35`), `k=2` reaches **`24.3–24.4 tok/s` (+14%)**, still slightly behind `k=1` (`25.1 tok/s`, **+17.4%**) unless PCIe staging overlaps (`+35%` overlap puts `k=2` at `25.1 tok/s`, **+17.5%**).
     - **`k >= 3` (`B >= 4`, modeled `verify >= 108.9 ms = 2.33×` serial)** loses to `k = 1..2` across every realistic acceptance curve (needs `emit(3) >= 2.83` for +20%).
     - **+20% speedup (`25.6 tok/s`, `39.0 ms/tok`) is NOT reachable with today's `p1=0.75` acceptance** (needs `p1 >= 0.83` at `k=1` or `emit(2) >= 2.36` at `k=2` + ragged verify + staging overlap).
3. **Every MiB of drafter VRAM steals expert cache slots** (`+0.0148 ms/MiB` at `B=1` from `opt/ledger.md:436–438`, scaling with `U(B)/8` to `+0.031 ms/MiB` at `B=3`). Re-quantizing DSpark (`182–338 MiB`) loses **4–10 ms/round** in extra PCIe misses alone, making `DSpark-Int8` (`337.5 MiB`) **slower than serial** (`−0.2%`) despite running in `2.28 ms`!
4. **Recommended Design:** **`EAGLE-1-Med` (`1L, D=1024, FF=2816, 16q/4kv hd=64, SWA-128`, 4-tap fuse, symmetric per-row `int8` `dp4a` body) + tied `V'=32,768` `int4-g128` nibble-unpack `dp4a` head** (**`29.9 MiB` total VRAM**, **`0.24 ms` (`k=1`) / `0.46 ms` (`k=2`) / `0.68 ms` (`k=3`)** in one CUDA Graph, **94.84%** real-token coverage, **100% d1 / 94.4% d1..3** Q5 proxy match). Run at **`k = 1` (`B = 2`)** first (immediate +13% win), then scale to **`k = 2` (`B = 3`)** if training achieves `q >= 0.60`.

---

## 1. Cost Model (`opt/kbench/drafter/cost_model.py`)

### 1.1 Exact Serial Baseline Breakdown (46.8 ms/tok, master `952e1ff`)

Reconciled from the graphs-ON `nsys` profile (`opt/ledger.md:541–552`) minus landed optimizations (`gemv_sf` −1.7 ms, `step_gcache` −6.7 ms, `ctrl/relbias` fusion −3.5 ms, `kslot` −1.1 ms):

| Component | Serial `B=1` (ms) | Share | Verify `B = k + 1` Scaling Law (Modeled) |
|---|---:|---:|---|
| EXL3 MoE GEMV (`gemv_moe_s`) | `19.8` | 42.3% | Ragged: `19.8 · (U(B)/8) · Σ_m P(m\|B)·t_EXL3(m)` <br> Padded: `19.8 · (U(B)/8) · t_EXL3(B)` |
| MoE PCIe staging (exposed `ccopy`) | `7.7` | 16.5% | `7.7 · [1 + (U(B)/8 - 1)·(1 - ov)]` + VRAM theft `0.0148·M_MiB·(U(B)/8)` |
| MoE non-GEMV (`had_in/out_moe`, `resolve`, `wsum`) | `2.5` | 5.3% | `2.5 · (0.5·U(B)/8 + 0.5·0.65·B)` |
| EXL3 Dense GEMV (48 visits attn `9.2` + unembed `1.9`) | `11.1` | 23.7% | `11.1 · t_EXL3(B)` (all `B` rows share exact same dense weights) |
| Non-EXL3 compute (`ctrl`, `router`, `conv`, `attn`, glue) | `4.9` | 10.5% | `4.9 · [0.60·(1 + 0.12(B-1)) + 0.40·(1 + 0.55(B-1))]` |
| Host turnaround & graph launch (`gcache`) | `0.8` | 1.7% | `0.8 + 0.15·(B - 1)` |
| **Total (`B = 1` serial, measured)** | **`46.8`** | **100%** | **`21.37 tok/s`** |

### 1.2 Inputs from Real Routing Traces (`ids_block.bin`, 384 Visits) & Batched EXL3 (`exl3_math2/REPORT.md`)

Measured directly from `opt/kbench/ids_block.bin` (384 layer visits) and `opt/kbench/exl3_math2/REPORT.md` (`M=16-Batch C0c` exact FP32: `t_EXL3(B) = B / speedup(B)`):

| `B` (`k` drafts) | Mean Union `U(B)` | `U(B)/8` | Avg Mult `8B/U` | Multiplicity Dist `P(m\|B)` | `t_EXL3(B)` (`speedup`) | Ragged MoE Mult | Padded MoE Mult |
|---:|---:|---:|---:|---|---:|---:|---:|
| `1` (`k=0`) | `8.00` | `1.000×` | `1.00` | `1:1.000` | `1.000×` (`1.00×`) | `1.000×` | `1.000×` |
| `2` (`k=1`) | `12.89` | `1.612×` | `1.24` | `1:0.759, 2:0.241` | `1.190×` (`1.68×`) | `1.686×` | `1.919×` |
| `3` (`k=2`) | `16.59` | `2.074×` | `1.45` | `1:0.662, 2:0.229, 3:0.108` | `1.422×` (`2.11×`) | `2.260×` | `2.949×` |
| `4` (`k=3`) | `19.96` | `2.495×` | `1.60` | `1:0.615, 2:0.232, 3:0.090, 4:0.064` | `1.581×` (`2.53×`) | `2.793×` | `3.945×` |
| `5` (`k=4`) | `22.91` | `2.864×` | `1.75` | `1:0.587, 2:0.221, 3:0.098, 4:0.050, 5:0.045` | `1.748×` (`2.86×`*) | `3.282×` | `5.007×` |
| `8` (`k=7`) | `28.36` | `3.545×` | `2.26` | `1:0.475, 2:0.238, 3+:0.287` | `2.247×` (`3.56×`*) | `4.393×` | `7.966×` |

*\*`B=5,8` linearly extrapolated from measured `B=1..4` slope (`+0.19/row`); `B=2..4` directly measured in `exl3_math2/REPORT.md` (and re-confirmed on `K=4.0` unembed below: `1.69×, 2.09×, 2.44×` at peak boost).*

### 1.3 Modeled `verify(k)` and Drafter Time Budgets (0%, +10%, +20% vs Serial)

Expected emitted tokens formula:
$$\text{emit}(k) = 1 + \sum_{j=1}^{k} p_1 q^{j-1} = 1 + p_1 \frac{1 - q^k}{1 - q}$$
Break-even drafter budget for speedup $S \in \{1.00, 1.10, 1.20\}$:
$$\text{draft\_budget}(k, S) = \text{emit}(k) \cdot \frac{46.8\text{ ms}}{S} - \text{verify}(k)$$

| Acceptance Regime (Modeled) | `k` (`B`) | `emit(k)` | `verify` Ragged (`×Ser`) | Budget **0%** (`46.8 ms`) | Budget **+10%** (`42.5 ms`) | Budget **+20%** (`39.0 ms`) | `verify` Ragged +35% Ov | Budget **0% / +10%** (ov35) | `verify` Padded (0% budget) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. Prompt baseline** (`p1=0.75, q=0.18`) | **1 (2)** | `1.750` | **`69.92 ms`** (`1.49×`) | **`+11.98 ms`** | **`+4.53 ms`** | `−1.67 ms` | `68.27 ms` | `+13.63 / +6.18` | `74.53 ms` (`+7.37`) |
| | **2 (3)** | `1.885` | `90.39 ms` (`1.93×`) | `−2.17 ms` | `−10.19 ms` | `−16.87 ms` | `87.49 ms` | `+0.72 / −7.30` | `104.04 ms` (`−15.82`) |
| | **3 (4)** | `1.909` | `108.87 ms` (`2.33×`) | `−19.52 ms` | `−27.64 ms` | `−34.41 ms` | `104.84 ms` | `−15.49 / −23.61` | `131.69 ms` (`−42.34`) |
| | **4 (5)** | `1.914` | `126.11 ms` (`2.69×`) | `−36.55 ms` | `−44.69 ms` | `−51.47 ms` | `121.08 ms` | `−31.52 / −39.66` | `160.26 ms` (`−70.70`) |
| **B. Ledger empirical** (`p1=0.75, q=0.46`) | **1 (2)** | `1.750` | **`69.92 ms`** (`1.49×`) | **`+11.98 ms`** | **`+4.53 ms`** | `−1.67 ms` | `68.27 ms` | `+13.63 / +6.18` | `74.53 ms` (`+7.37`) |
| | **2 (3)** | `2.095` | `90.39 ms` (`1.93×`) | **`+7.66 ms`** | `−1.26 ms` | `−8.68 ms` | `87.49 ms` | `+10.55 / +1.64` | `104.04 ms` (`−5.99`) |
| | **3 (4)** | `2.254` | `108.87 ms` (`2.33×`) | `−3.40 ms` | `−12.99 ms` | `−20.98 ms` | `104.84 ms` | `+0.63 / −8.96` | `131.69 ms` (`−26.22`) |
| | **4 (5)** | `2.327` | `126.11 ms` (`2.69×`) | `−17.22 ms` | `−27.11 ms` | `−35.36 ms` | `121.08 ms` | `−12.19 / −22.09` | `160.26 ms` (`−51.37`) |
| **C. Trained EAGLE** (`p1=0.82, q=0.65`) | **1 (2)** | `1.820` | `69.92 ms` (`1.49×`) | **`+15.26 ms`** | **`+7.51 ms`** | **`+1.06 ms`** | `68.27 ms` | `+16.90 / +9.16` | `74.53 ms` (`+10.64`) |
| | **2 (3)** | `2.353` | `90.39 ms` (`1.93×`) | **`+19.73 ms`** | **`+9.72 ms`** | **`+1.38 ms`** | `87.49 ms` | `+22.63 / +12.62` | `104.04 ms` (`+6.08`) |
| | **3 (4)** | `2.699` | `108.87 ms` (`2.33×`) | **`+17.46 ms`** | **`+5.98 ms`** | `−3.59 ms` | `104.84 ms` | `+21.49 / +10.01` | `131.69 ms` (`−5.36`) |
| | **4 (5)** | `2.925` | `126.11 ms` (`2.69×`) | `+10.77 ms` | `−1.67 ms` | `−12.04 ms` | `121.08 ms` | `+15.79 / +3.35` | `160.26 ms` (`−23.39`) |

With a 1.5 ms drafter (ragged verify), required `emit(k)` thresholds are:
- **`k=1`:** `>= 1.53` (0%), `>= 1.68` (+10%), `>= 1.83` (+20%)
- **`k=2`:** `>= 1.96` (0%), `>= 2.16` (+10%), `>= 2.36` (+20%)
- **`k=3`:** `>= 2.36` (0%), `>= 2.59` (+10%), `>= 2.83` (+20%)
- **`k=4`:** `>= 2.73` (0%), `>= 3.00` (+10%), `>= 3.27` (+20%)

---

## 2. Head Comparison & Reduced-Vocabulary Coverage

### 2.1 Measured Silicon Head Latencies (`opt/kbench/drafter/bench_drafter.cu`, GTX 1060 6GB)

All reduced heads execute **GEMV + Argmax + `vocab_ids[V']` remap in ONE kernel launch** (atomic-ticket last-block finalize, zero DRAM logit buffer, zero host sync, CUDA-graph-capturable):

| Head Option (All Measured on Silicon) | Hidden `H` | Extra VRAM | `B=1` (us / GB/s) | `B=2` (us / GB/s) | `B=3` (us / GB/s) | `B=3` `dp2a` (int16 X) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| **(2a) Resident EXL3 K=4.0 (`200,058`)** | 2304 | **0.0 MiB** | `1758 us` (`136 GB/s`) | `2112 us` (`115 GB/s`, `1.69×`) | `2609 us` (`95 GB/s`, `2.09×`) | exact FP32 | `B=4`: `2998 us` (`2.44×`). Warm boost: `2.01 / 2.37 / 2.78 ms`. Autoreg `k=2/3` pays `3.5–4.0 / 5.3–6.0 ms`! |
| **(2b) Int8 `dp4a` `V'=8K`** | 2304 | 18.1 MiB | `117.8 us` (`160.8`) | `128.5 us` (`147.3`) | `117.6 us` (`161.1`) | `157.7 us` (`120.1`) | 77.8% coverage (too low) |
| **(2b) Int8 `dp4a` `V'=16K`** | 2304 | 36.1 MiB | `223.7 us` (`169.3`) | `239.5 us` (`158.2`) | `223.1 us` (`169.8`) | `300.0 us` (`126.3`) | 86.7% coverage |
| **(2b) Int8 `dp4a` `V'=32K`** | 2304 | 72.2 MiB | `434.1 us` (`174.5`) | `459.8 us` (`164.8`) | **`436.2 us` (`173.7`)** | `593.4 us` (`127.7`) | **85.2% of DRAM peak!** 94.8% cov |
| **(2b) Int8 `dp4a` `V'=64K`** | 2304 | 144.5 MiB | `862.8 us` (`175.6`) | `896.0 us` (`169.1`) | **`860.2 us` (`176.2`)** | `1126.4 us` (`134.5`) | **86.4% of DRAM peak!** Too much VRAM |
| **(2c) Int4-g128 nibble-`dp4a` `V'=32K`** | 2304 | **37.2 MiB** | **`238.6 us` (`163.7`)** | `244.3 us` (`159.9`) | `284.6 us` (`137.2`) | n/a | 100% d1 / 94.4% d1..3 (Q5), 0 spills! |
| **(2c) Int4-row nibble-`dp4a` `V'=32K`** | 2304 | 36.2 MiB | `228.4 us` (`166.5`) | `245.3 us` (`154.9`) | `327.7 us` (`116.0`) | n/a | **Disqualified by Q5** (67% d1 match) |
| **(2c) Narrow Int8 `V'=32K` (`H=1024`)** | 1024 | 32.2 MiB | `201.7 us` (`167.7`) | `208.7 us` (`162.1`) | `204.7 us` (`165.2`) | `281.2 us` (`120.3`) | 100% d1..3 quant lossless |
| **(2c) Narrow Int4-g128 `V'=32K` (`H=1024`)** | 1024 | **16.6 MiB** | **`108.5 us` (`160.6`)** | **`115.5 us` (`151.0`)** | **`146.4 us` (`119.0`)** | n/a | **WINNER: 16.6 MiB, 0.11 ms/step, 0 spills!** |
| **(2c) Narrow Int4-g128 `V'=16K` (`H=1024`)** | 1024 | 8.3 MiB | `58.4 us` (`149.3`) | `63.1 us` (`138.1`) | `78.8 us` (`110.5`) | n/a | Fast, but loses −8.1 pp coverage |
| **(2c) Narrow Int4-g128 `V'=32K` (`H=512`)** | 512 | **8.4 MiB** | **`81.9 us` (`107.2`)** | `88.6 us` (`99.1`) | `98.3 us` (`89.3`) | n/a | Ultra-light slim option |
| **(2c) 2-Stage Shortlist `r=256->M=64` (`V'=32K`)** | 2304 | `8.8 MiB` + rescore | `114.4 us` | `103.0 us` | `133.1 us` | n/a | Needs rescore rows (dominated by narrow `H=1024` i4-g128) |

### 2.2 Reduced-Vocabulary Coverage on Real Target Greedy Tokens (`opt/kbench/drafter/vocab_coverage.py`)

**Scored set:** **640 real serial greedy tokens** collected from Inkling-2-Mini (`./target/release/inkwell ~/models/inkling --run`, `INKWELL_GENS=128` × 5 prompts: `prose`, `python`, `rust`, `tech`, `factual`).  
**Frequency ranking source (`opt/kbench/drafter/vocab_rank_corpus.bin`):** Strictly independent **4.12M-token** local multi-domain corpus (`1.35M` `/usr/share/man` [40%], `1.26M` `/usr/share/doc` [30%], `1.03M` `/usr/lib/python3.14` [20%], `0.49M` `/usr/include` [10%], 25,625 unique observed tokens), tokenized with Inkling's exact GGUF BPE vocabulary and merges, with GGUF BPE merge-rank order on ASCII tokens for corpus-unseen tokens (`0..255` bytes and `EOS=200006` pinned). **Zero overlap with evaluation prompts.**

| `V'` | Independent Corpus Ranking (Overall) | `prose` | `python` | `rust` | `tech` | `factual` | Naive `id < V'` (Overall) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **4,096** | `71.41%` | 60.9% | 89.1% | 78.1% | 68.0% | 60.9% | `31.56%` |
| **8,192** | `77.81%` | 70.3% | 89.8% | 81.3% | 86.7% | 60.9% | `32.19%` |
| **16,384** | `86.72%` | 83.6% | 89.8% | 89.8% | 96.9% | 73.4% | `32.19%` |
| **32,768** | **`94.84%`** | **93.8%** | **100.0%** | **93.8%** | **100.0%** | **86.7%** | `33.12%` |
| **65,536** | `95.94%` | 96.9% | 100.0% | 94.5% | 100.0% | 88.3% | `35.62%` |

**Why naive `id < V'` collapses to 33%:** In Inkling's 200,058-token multilingual vocabulary, CJK tokens sit in low IDs (`500..16000`) while common leading-space English words live in `149,860..199,999` (`' the' = 149876`, `' to' = 149882`, `' is' = 149884`). A frequency-ranked `uint32_t vocab_ids[V']` table is mandatory, and **`V' = 32,768`** is the clear knee (+8.1 pp over 16K, only +1.1 pp left to 64K for 2× VRAM).

---

## 3. Drafter Body Design & Full CUDA-Graph Round Measurements

- **Target hidden-state taps (EAGLE/DSpark-style) vs token-only:** Fusing the target's 4 mixed hidden taps (`[4, 2304] -> [D]`) costs only **~8–15 us once per round** and is essential — a 1–2 layer token-only LM has ~25–35% acceptance against a 48-visit looped MoE, whereas EAGLE-style tap-conditioned drafters hit 75–85% `p1` in a single `D=1024` layer.
- **Short-window attention (SWA-128) vs none (Medusa MLP):** SWA-128 (`nkv=4, hd=64`) costs only **~12 us/step** and **< 0.13 MiB KV/layer**, preserving autoregressive conditioning across drafted tokens (`d1 -> d2 -> d3`), which is critical for continuation probability `q`.
- **Single-graph capture:** `drafter_embed_add` reads `out_tok[0]` on device from the prior step's fused `head_argmax_*`, so `k = 1..3` autoregressive steps run inside **one `cudaGraphLaunch`** with zero host sync.

### Measured Full-Round Times (Captured CUDA Graph, `bench_drafter.cu`, Median-of-31)

| Drafter Architecture | Body Shape | Head | Total VRAM (tied / untied) | `k=1` (ms) | `k=2` (ms) | `k=3` (ms) |
|---|---|---|---:|---:|---:|---:|
| **`EAGLE-1-Slim`** | 1L `D=512` `FF=1536` `8q/2kv` SWA-128 + 4-tap (i8) | `i4-g128 V'=32K` (`H=512`) | **12.6** / 20.6 MiB | **0.15** | **0.28** | **0.42** |
| **`EAGLE-1-Med` (REC)** | 1L `D=1024` `FF=2816` `16q/4kv` SWA-128 + 4-tap (i8) | `i4-g128 V'=32K` (`H=1024`) | **29.9** / 45.9 MiB | **0.24** | **0.46** | **0.68** |
| **`EAGLE-1-Med (i8 32K)`** | 1L `D=1024` `FF=2816` `16q/4kv` SWA-128 + 4-tap (i8) | `i8 V'=32K` (`H=1024`) | 45.4 / 61.4 MiB | 0.33 | 0.63 | 0.95 |
| **`EAGLE-1-Med (i8 16K)`** | 1L `D=1024` `FF=2816` `16q/4kv` SWA-128 + 4-tap (i8) | `i8 V'=16K` (`H=1024`) | 29.2 / 37.2 MiB | 0.23 | 0.45 | 0.66 |
| **`EAGLE-2L-Med`** | 2L `D=1024` `FF=2816` `16q/4kv` SWA-128 + 4-tap (i8) | `i4-g128 V'=32K` (`H=1024`) | **40.7** / 56.7 MiB | **0.34** | **0.66** | **0.99** |
| **`EAGLE-1-Full (i8+i4g)`** | 1L `D=2304` `FF=4608` `18q/6kv` SWA-128 + 4-tap (i8) | `i4-g128 V'=32K` (`H=2304`) | 81.6 / 117.6 MiB | 0.55 | 1.10 | 1.65 |
| **`EAGLE-1-Full + ResEXL3`** | 1L `D=2304` `FF=4608` `18q/6kv` SWA-128 + 4-tap (i8) | Resident EXL3 (`200K`) | **44.3 MiB** | 2.08–2.33 | 4.16–4.66 | 6.24–6.99 |
| **`Medusa-MLP (i8+i4g)`** | 0-attn 2L ResMLP `D=1024` `FF=2048` (batched `B=k`) | `i4-g128 V'=32K` (`H=1024`) | 31.0 MiB | 0.23 | 0.25 | 0.30 |
| **`Medusa-MLP + ResEXL3`** | 0-attn 2L ResMLP `D=1024` + `up2304` (batched `B=k`) | Resident EXL3 (`200K`) | **16.5 MiB** | 1.88–2.13 | 2.24–2.49 | 2.74–2.91 |
| **`DSpark-Int4-g128` (0-retrain)** | Existing 3L `D=2304` `FF=9216` + Markov GRU (i4-g128) | `i4-g128 V'=32K` (`H=2304`) | 182.5 MiB | 1.46 | 1.90 | 2.53 |
| **`DSpark-Int8` (0-retrain)** | Existing 3L `D=2304` `FF=9216` + Markov GRU (i8) | `i8 V'=32K` (`H=2304`) | 337.5 MiB | 2.28 | 2.93 | 3.50 |

---

## 4. Hand-Tuned `sm_61` Kernel Metrics (`opt/kbench/drafter/drafter_kernels.cuh`)

Compiled with `nvcc 12.6 -O3 --use_fast_math -gencode arch=compute_61,code=sm_61 -Xptxas -v`. All kernels verified bit-exact against CPU `int64`/`fp64` reference (`relerr = 5.8e-8–6.1e-8`, argmax 100% PASS):

| Kernel (`sm_61`) | Threads/Blk | Regs/Thr | Spills | Static Smem | Blks/SM | Measured Speed & BW (vs 204 GB/s peak) |
|---|---:|---:|---:|---:|---:|---|
| `gemv_i8_dp4a<1..3>` | 256 | `40` | `0 B` | `0` (+dyn `B·K`) | **6** | `Head_32K B=2`: `436.2 us` = **173.4 GB/s (85.0%)**; `FFN_2304 B=3`: **162.3 GB/s** |
| `gemv_i8_dp2a<3>` (int16 X) | 256 | `40` | `0 B` | `0` (+dyn `2B·K`) | **6** | `Head_32K B=3`: `501.8 us` = **150.7 GB/s (73.9%)** (+13% time vs `dp4a`, 256× act precision) |
| `gemv_i4_dp4a<1..4>` (nibble `+8`) | 256 | `39–40` | `0 B` | `0` (+dyn `B·K`) | **6** | `Head_32K B=3`: `304.1 us` = **124.6 GB/s** (`1.46×` faster than `i8` at `B=3`, `1.9×` at `B=1`) |
| `head_argmax_i8_dp4a<3>` | 256 | `40` | `0 B` | `192 B` (+dyn `3K`) | **6** | `V'=64K`: **176.2 GB/s (86.4%)**; `V'=32K`: `436.2 us` = **173.7 GB/s (85.2%)** |
| **`head_argmax_i4g128_dp4a<1..3>`** | **256** | **`38–40`** | **`0 B`** | **`192–576 B`** | **6** | **`V'=32K H=1024`: `108.5 / 115.5 / 146.4 us` (`160.6 GB/s` = 78.7%)**; **`H=2304`: `238.6 us` (`163.7 GB/s`)** |
| `head_argmax_i8_dp2a<3>` | 256 | `48` | `12 B` | `192 B` (+dyn `6K`) | **5** | `V'=32K H=2304`: `593.4 us` (`127.7 GB/s`); `H=1024`: `281.2 us` (`120.3 GB/s`) |
| `fused_rmsnorm_quant<i8/i16>` | 256 | `64` | `0 B` | `32–64 B` (+dyn `4K`) | **4** | `K=512..2304, B=1..3`: **`3.1–4.1 us`** (single DRAM pass + `sum_x`!) |
| `fused_swiglu_quant_i8` | 256 | `40` | `0 B` | `64 B` (+dyn `4FF`) | **6** | Single DRAM pass over `gate, up` -> quantized `int8` + `scale_x` + `sum_x` |
| `head_exl3_k4_batch_c0c<1..4>` | 256 | `40` | `0 B` | `2.1–8.2 KB` | **6** | Real `unembed.bin`: `B=1: 1.70 ms` (`273 GMAC/s`) -> `B=4: 2.78 ms` (`666 GMAC/s`, `2.44×`) |
| `exl3_had_out_argmax<1..4>` | 256 | `20–36` | `0 B` | `576–768 B` | **6–8** | `B=1: 59 us`, `B=3: 171 us` (fused H128 + `svh` + device argmax) |
| `drafter_swa_attn` | 128 | `53` | `0 B` | `1040 B` | **9** | Short-window (`win<=128`) GQA attention, 1 block per `(row, head)` |

---

## 5. Acceptance Proxy on Real DSpark Feature Taps (`opt/kbench/drafter/acceptance_proxy.cu`)

**Setup (all measured on GPU with real weights & real target feature taps):**
- Collected **12 diverse prompts** (factual, prose, Python, Rust, tech, math, dialogue, JSON, SQL, CUDA, history, science) × 7 DSpark draft steps = **84 real draft steps** (`feats.bin`, `kc/vc`, `x.bin`, `hid.bin`, `f2.bin`, logged `vtok`, and target greedy `veri`).
- Dequantized real `model.llm.unembed` (EXL3 K=4.0, `200,058 × 2304`) + real `mtp.safetensors` (all 3 layers + Markov GRU).
- **Oracle verification:** FP32 replay matched Inkwell's exact logged `vtok` on **84 / 84 (100.00%)** draft steps (`12/12` on d1). Baseline real target acceptance on these 12 diverse single-round prompts: `P(d1) = 6/12 (50.0%)`, `emit(k=1/2/3) = 1.50 / 1.75 / 2.00`.

### 5A. Head-Only Quantization + Top-`V'` (Exact FP32 `u2_i` -> Quantized Unembed)

| Head Quant Mode | `V'` | Match vs FP32 `d1` | Match vs FP32 `d1..3` | Match vs FP32 `d1..7` | Target `P(d1)` | Real `emit(1/2/3)` |
|---|---:|---:|---:|---:|---:|---:|
| **FP32 Oracle** | `200K / 64K / 32K` | `12/12 (100%)` | `36/36 (100.0%)` | `84/84 (100.0%)` | `6/12 (50%)` | `1.50 / 1.75 / 2.00` |
| FP32 Oracle | `16K` / `8K` | `92%` / `83%` | `94.4%` / `91.7%` | `97.6%` / `95.2%` | `50%` / `50%` | `1.50 / 1.75 / 2.00` |
| **Int8 W + Int16 X (`dp2a`)** | **`32K` (`200K`)** | **`12/12 (100%)`** | **`36/36 (100.0%)`** | **`84/84 (100.0%)`** | `6/12 (50%)` | **`1.50 / 1.75 / 2.00`** |
| **Int8 W + Int8 X (`dp4a`)** | **`32K` (`200K`)** | **`12/12 (100%)`** | **`36/36 (100.0%)`** | **`83/84 (98.8%)`** | `6/12 (50%)` | **`1.50 / 1.75 / 2.00`** |
| Int8 W + Int8 X (`dp4a`) | `16K` / `8K` | `92%` / `83%` | `94.4%` / `91.7%` | `96.4%` / `94.0%` | `50%` / `50%` | `1.50 / 1.75 / 2.00` |
| **Int4-g128 W + Int8 X (`dp4a`)** | **`32K` (`200K`)** | **`12/12 (100%)`** | **`34/36 (94.4%)`** | **`79/84 (94.0%)`** | `6/12 (50%)` | **`1.50 / 1.75 / 2.00`** |
| Int4-g128 W + Int8 X (`dp4a`) | `16K` / `8K` | `92%` / `83%` | `88.9%` / `86.1%` | `91.7%` / `89.3%` | `50%` / `50%` | `1.50 / 1.75 / 2.00` |
| **Int4-row W + Int8 X (`dp4a`)** | `32K` (`200K`) | **`8/12 (67%)`** | **`30/36 (83.3%)`** | **`74/84 (88.1%)`** | `5/12 (42%)` | **`1.42 / 1.58 / 1.75`** |

### 5B. Full End-to-End Quantized DSpark (ALL 3 Layers + Markov GRU + Unembed, Autoregressive)

| Full DSpark Quant Mode | `V'` | Match vs FP32 `d1` | Match vs FP32 `d1..3` | Match vs FP32 `d1..7` | Target `P(d1)` | Real `emit(1/2/3)` |
|---|---:|---:|---:|---:|---:|---:|
| **Full Int8 + Int16 X (`dp2a`)** | **`200K` / `32K`** | **`12/12 (100%)`** | **`36/36 (100.0%)`** | **`84/84 (100.0%)`** | `6/12 (50%)` | **`1.50 / 1.75 / 2.00`** |
| Full Int8 + Int16 X (`dp2a`) | `16K` / `8K` | `92%` / `83%` | `91.7%` / `77.8%` | `90.5%` / `78.6%` | `50%` / `50%` | `1.50 / 1.75 / 2.00` |
| **Full Int8 + Int8 X (`dp4a`)** | **`200K` / `32K`** | **`12/12 (100%)`** | **`36/36 (100.0%)`** | **`80/84 (95.2%)`** | `6/12 (50%)` | **`1.50 / 1.75 / 2.00`** |
| Full Int8 + Int8 X (`dp4a`) | `16K` / `8K` | `92%` / `83%` | `91.7%` / `77.8%` | `85.7%` / `73.8%` | `50%` / `50%` | `1.50 / 1.75 / 2.00` |
| **Full Int4-g128 + Int8 X (`dp4a`)** | `200K` / `32K` | `11/12 (92%)` | `29/36 (80.6%)` | `51/84 (60.7%)` | `6/12 (50%)` | `1.50 / 1.75 / 2.00` |
| **Full Int4-row + Int8 X (`dp4a`)** | `200K` / `32K` | **`7/12 (58%)`** | **`23/36 (63.9%)`** | **`36/84 (42.9%)`** | **`4/12 (33%)`** | **`1.33 / 1.50 / 1.67`** |

**Three decisive findings from Q5:**
1. **Per-row `int8` weights + `int8` activations (`dp4a`) are completely lossless on drafts 1..3** (`36/36 = 100.0%` end-to-end across all 3 layers + GRU + 32K head; `dp2a` int16 activations recover even deep drafts 4..7 to `84/84 = 100.0%`).
2. **Per-row `int4` fails badly** (`67%` head-only / `58%` full on d1), whereas **group-128 `int4` (`int4-g128`)** in the head recovers **100% d1 (`12/12`) and 94.4% d1..3 (`34/36`)** while halving head VRAM and running in `108.5 us` (`160.6 GB/s`, 0 spills)!
3. **Hybrid recipe proven:** keep the small body in **per-row `int8` (`dp4a`)** (zero compounding body error!) and put the large `V'=32K` head in **`int4-g128` (`dp4a`)** (saves 16–35 MiB VRAM at zero d1 loss!).

---

## 6. Master Ranked Table of Drafter Designs (`opt/kbench/drafter/rank_designs.py`)

Serial baseline: **`46.8 ms/tok` = `21.37 tok/s`** (measured).  
All round times, VRAM, coverage, and Q5 proxy retention are **MEASURED**; `verify(k)` and end-to-end tok/s are **MODELED** (ragged batched MoE verify + VRAM-theft cache penalty `0.0148·M_MiB·U(B)/8`).

| Rank | Design (Head Option) | Body Shape | VRAM (MiB) | Measured Round `k=1` / `k=2` / `k=3` | Coverage / Q5 Proxy (Measured) | Modeled E2E tok/s — Regime A (`p1=.75, q=.18`) `k=1 / k=2` | Modeled E2E tok/s — Regime B (`p1=.75, q=.46`) `k=1 / k=2` | Modeled E2E tok/s — Regime C (`p1=.82, q=.65`) `k=1 / k=2` | Best vs Serial |
|---:|---|---|---:|---|---|---:|---:|---:|---:|
| **1 (REC)** | **`EAGLE-1-Med`** (`i4-g128 V'=32K` tied) | 1L `D=1024 FF=2816` SWA-128 + 4-tap (`i8`) | **29.9** | **`0.24 / 0.46 / 0.68 ms`** | 94.8% cov; Q5: 100% d1 / 94.4% d1..3 | **`24.14` / `19.89`** | **`24.14` / `21.84`** | **`25.08` / `24.30`** | **+13.0% (A/B) / +17.4% (C)** |
| **2** | **`EAGLE-2L-Med`** (`i4-g128 V'=32K` tied) | 2L `D=1024 FF=2816` SWA-128 + 4-tap (`i8`) | 40.7 | `0.34 / 0.66 / 0.99 ms` | 94.8% cov; Q5: 100% d1 / 94.4% d1..3 | `24.02` / `19.78` | `24.02` / `21.71` | `24.95` / `24.16` | +12.4% / +16.8% |
| **3** | **`EAGLE-1-Med`** (`i8 V'=32K` tied) | 1L `D=1024 FF=2816` SWA-128 + 4-tap (`i8`) | 45.4 | `0.33 / 0.63 / 0.95 ms` | 94.8% cov; Q5: 100% d1..3 / 95.2% d1..7 | `23.99` / `19.83` | `23.99` / `21.87` | `24.92` / **`24.42`** | +12.3% / +16.6% |
| **4** | **`EAGLE-1-Slim`** (`i4-g128 V'=32K` tied) | 1L `D=512 FF=1536` SWA-128 + 4-tap (`i8`) | **12.6** | **`0.15 / 0.28 / 0.42 ms`** | 94.8% cov; Q5: 100% d1 (est 0.96 cap) | `23.91` / `19.61` | `23.91` / `21.38` | `24.82` / `23.66` | +11.9% / +16.1% |
| **5** | **`EAGLE-1-Full`** (Resident EXL3 `200K`) | 1L `D=2304 FF=4608` SWA-128 + 4-tap (`i8`) | 44.3 | `2.08–2.33 / 4.16–4.66 / 6.24–6.99 ms` | **100% cov**; Q5: 100% d1..3 | `23.87` / `19.55` | `23.87` / `21.73` | `24.83` / `24.41` | +11.7% / +16.2% |
| **6** | **`EAGLE-1-Full`** (`i4-g128 V'=32K` tied) | 1L `D=2304 FF=4608` SWA-128 + 4-tap (`i8`) | 81.6 | `0.55 / 1.10 / 1.65 ms` | 94.8% cov; Q5: 100% d1 / 94.4% d1..3 | `23.62` / `19.42` | `23.62` / `21.31` | `24.54` / `23.72` | +10.6% / +14.9% |
| **7** | **`Medusa-MLP`** (`i4-g128 V'=32K`) | 0-attn 2L ResMLP `D=1024 FF=2048` | 31.0 | `0.23 / 0.25 / 0.30 ms` | 94.8% cov; 0-attn lower `q` (modeled) | `23.33` / `18.91` | `23.33` / `20.24` | `24.20` / `22.10` | +9.2% / +13.2% |
| **8** | **`Medusa-MLP`** (Resident EXL3 `200K`) | 0-attn 2L ResMLP `D=1024 + up2304` | **16.5** | `1.88–2.13 / 2.24–2.49 / 2.74–2.91 ms` | **100% cov**; 0-attn lower `q` (modeled) | `23.33` / `19.03` | `23.33` / `20.48` | `24.22` / `22.46` | +9.2% / +13.3% |
| **9** | **`EAGLE-1-Med`** (`i8 V'=16K` tied) | 1L `D=1024 FF=2816` SWA-128 + 4-tap (`i8`) | 29.2 | `0.23 / 0.45 / 0.66 ms` | 86.7% cov; Q5: 92% d1 / 91.7% d1..3 | `23.30` / `19.10` | `23.30` / `20.82` | `24.15` / `23.02` | +9.0% / +13.0% |
| **10** | **`DSpark-Int4-g128`** (0-retrain) | Existing 3L `D=2304 FF=9216` + GRU (`i4-g128`) | 182.5 | `1.46 / 1.90 / 2.53 ms` | 94.8% cov; Q5 full: 92% d1 / 80.6% d1..3 | `21.81` / `17.79` | `21.81` / `19.22` | `22.61` / `21.11` | +2.1% / +5.8% |
| **11** | **`DSpark-Int8`** (0-retrain) | Existing 3L `D=2304 FF=9216` + GRU (`i8`) | 337.5 | `2.28 / 2.93 / 3.50 ms` | 94.8% cov; Q5 full: 100% d1..3 / 95.2% d1..7 | `21.32` / `17.68` | `21.32` / `19.50` | `22.15` / `21.77` | **−0.2%** / +3.7% |
| **12** | **Today DSpark EXL3** (no graph) | Existing 3L EXL3 + Markov (`f32`) | 236.0 | `~22 / 32 / 42 ms` | 100% cov (baseline) | `17.94` / `14.54` | `17.94` / `16.16` | `18.66` / `18.15` | **−16.0%** |

**Why Rank #1 (`EAGLE-1-Med`, `29.9 MiB`) wins:**
- At **29.9 MiB**, it steals less than 1 cache slot across the 14 cached layers (`~0.7 ms` penalty at `B=2`, `~0.9 ms` at `B=3`) — leaving ample headroom for compact `B=2..3` verify scratch without OOM.
- If **zero retraining** is desired as a stepping stone to test compact `B=2` verify first, **`DSpark-Int4-g128` (`182.5 MiB`, `1.46 ms` at `k=1`)** is the only 0-retrain option that slightly beats serial (+2.1%), whereas training **`EAGLE-1-Med`** unlocks **+13–17%**.
- Note also **Rank #3 (`EAGLE-1-Med` with `i8 V'=32K` head, `45.4 MiB`)**: virtually ties Rank #1 (+12.3% vs +13.0%) and wins slightly at `k=2` in Regime C (`24.42` vs `24.30 tok/s`) with **zero int4 head quantization error** — an easy fallback if `int4-g128` training/PTQ has any friction.

---

## 7. Format the Kernel Expects (Specification for Quantization / Training Agent)

Ship `drafter.safetensors` with the following exact contract so Inkwell can load and execute directly inside one CUDA Graph with zero runtime re-packing:

1. **Dimension Multiples (Strict):**
   - Every linear matrix `W` is row-major `[N_out, K_in]` (`K_in` contiguous).
   - **`K_in % 128 == 0`** and **`N_out % 128 == 0`** (required for coalesced 16-byte `int4`/`uint4` loads, 128-element group scales, and 8-warp blocks with zero tail branches).
   - Recommended `EAGLE-1-Med` dims: `D = 1024`, `FF = 2816` (`22 × 128`), `nh = 16`, `nkv = 4`, `hd = 64`, `V' = 32768` (`256 × 128`), `tap_dim = 2304` (`18 × 128`).
2. **Weight Dtypes & Scale Layouts:**
   - **Body linear weights (symmetric per-output-row `int8` for `dp4a` / `dp2a`):**
     - `*.weight`: dtype `I8`, shape `[N_out, K_in]`, values in `[-127, +127]`.
     - `*.scale`: dtype `F32`, shape `[N_out]`, `scale[r] = max_c |W[r, c]| / 127.0f`, `W_fp32[r,c] ≈ W[r,c] * scale[r]`.
     - *(Works identically with both `dp4a` int8 activations and `dp2a` int16 activations — zero weight format change needed if a layer uses `dp2a`!)*
   - **Head linear weights (`int4-g128` nibble-packed, symmetric per-128-group; OR per-row `int8` above):**
     - `lm_head.weight_i4`: dtype `U8`, shape `[V', D / 2]` (equivalently `U32 [V', D / 8]`).
     - **Exact bit packing (zero-shuffle `dp4a.s32.u32` layout):** Each little-endian `uint32_t` word `p` stores **8 consecutive weights `q[0..7] ∈ [-7, +7]`** biased by `+8` into unsigned nibbles `u[i] = q[i] + 8 ∈ [1, 15]`:
       - Low nibbles (`p & 0x0F0F0F0F`) hold `[u0, u1, u2, u3]` (columns `k..k+3`).
       - High nibbles (`(p >> 4) & 0x0F0F0F0F`) hold `[u4, u5, u6, u7]` (columns `k+4..k+7`).
     - `lm_head.scale`: dtype `F16`, shape `[V', D / 128]`, `scale[r, g] = max_{c in group g} |W[r, c]| / 7.0`.
     - **Do NOT use per-row int4** (single scale over `D`) — Q5 measured a catastrophic drop (`67%` head-only / `58%` full d1 match vs `100%` for `int4-g128`).
   - **Norm weights & biases:** `F32` (or `BF16` widened at load, `< 0.05 MiB`).
   - **Target tap weights:** `tap.fuse_weight`, `F32 [4]` (softmax weights over target layer taps `[5, 23, 41, 48]`).
3. **Reduced Vocabulary Table:**
   - `vocab_ids`: dtype `U32`, shape `[V']` (`V' = 32768`, multiple of 128). Maps reduced head row `r ∈ [0, V')` to target token ID in `[0, 200058)`.
   - `inv_vocab_ids`: dtype `I32`, shape `[200058]` (`0.76 MiB`). Maps full target token ID -> reduced row `r ∈ [0, V')`, or `-1` (fallback to row `0`) for OOV tokens during anchor embedding lookup on device.
   - Must pin all 256 byte tokens (`0..255`) and `EOS = 200006`, then top corpus-frequency tokens (never naive `0..V'-1`). Reference table: `opt/kbench/drafter/vocab_rank_corpus.bin`.
4. **Proposed Tensor Names (`drafter.safetensors`):**
   - `vocab_ids` (`U32 [32768]`), `inv_vocab_ids` (`I32 [200058]`)
   - `tap.fuse_weight` (`F32 [4]`), `tap.norm.weight` (`F32 [2304]`), `tap.proj.{weight,scale}` (`I8 [D, 2304]`, `F32 [D]`)
   - `layers.{l}.in_norm.weight` (`F32 [D]`)
   - `layers.{l}.attn.qkv_proj.{weight,scale}` (`I8 [qdim+2*kvdim, D]`, fused Q/K/V)
   - `layers.{l}.attn.q_norm.weight`, `layers.{l}.attn.k_norm.weight` (`F32 [hd]`)
   - `layers.{l}.attn.o_proj.{weight,scale}` (`I8 [D, qdim]`)
   - `layers.{l}.post_norm.weight` (`F32 [D]`)
   - `layers.{l}.mlp.gate_up_proj.{weight,scale}` (`I8 [2*FF, D]`, fused Gate/Up)
   - `layers.{l}.mlp.down_proj.{weight,scale}` (`I8 [D, FF]`)
   - `final_norm.weight` (`F32 [D]`)
   - `lm_head.{weight_i4,scale}` (`U8 [32768, D/2]`, `F16 [32768, D/128]`) — or `lm_head.{weight,scale}` (`I8 [32768, D]`, `F32 [32768]`)
   - Optional untied `embed.{weight,scale}` (`I8 [32768, D]`, `F32 [32768]`) — omit when tied with `lm_head`.

---

## 8. Reproduction Commands

```bash
# 1. Cost model (CPU)
python3 -I opt/kbench/drafter/cost_model.py

# 2. Collect real serial tokens (640 tok) & real DSpark MTP feature taps (12 prompts, 84 steps) under flock
flock /tmp/inkwell-gpu.lock bash opt/kbench/drafter/collect_data.sh
python3 -I opt/kbench/drafter/build_vocab_rank.py
python3 -I opt/kbench/drafter/vocab_coverage.py
python3 -I opt/kbench/drafter/prep_proxy_data.py

# 3. Build & run sm_61 drafter microbenchmarks + real acceptance proxy (under flock)
bash opt/kbench/drafter/build_and_run.sh

# 4. Generate ranked design synthesis
python3 -I opt/kbench/drafter/rank_designs.py
```
