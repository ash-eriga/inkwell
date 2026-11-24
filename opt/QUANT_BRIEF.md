# Inkling-2-Mini on Inkwell (GTX 1060 6 GB): quantization and drafter brief

This document is self-contained. It tells the quantization agent what the Inkwell runtime needs from a new quant and from a speculative drafter. Every number comes from a measurement in the Inkwell repo unless it is marked as an estimate or modeled.

## 1. The bottom line

1. **Keep the EXL3 mul1 trellis for the experts.** Do not move them to an int8-constrained codebook (TQIR-INT8, CPTC-INT8), int4, int8 or fp16. Section 4 gives the reasons. The dense weights are the one exception worth testing (item 8).
2. **On the GTX 1060, a lower bpw does not make the GEMV kernels faster per weight.** They are limited by instruction issue, not memory bandwidth. Bits per weight change speed only through which experts fit in VRAM and how many bytes cross PCIe per token.
3. **The master spec's 128K plan fits on paper, with 5–25 MiB to spare.** Its expert layout is "loop layers resident, other layers streamed every token" (§6). Under that plan:
   - the loop-layer bpw decides whether the plan fits at all;
   - the non-loop expert bpw sets the PCIe bytes per token (~1.1 ms per 0.1 bpw).
4. **Bytes matter more than the choice of K.** Where quality allows a choice, K 2.0, 3.0 and 4.0 get Inkwell's faster decode core; 1.5, 2.5, 3.5 and 5.0 do not (§5.3). The gain is small.
5. **Report quality the same way as the current quant:** mean KLD and top-1 agreement against the source, using float64 logits over the full vocabulary.
6. **For the drafter, acceptance decides the gain, not drafter speed.** Inkwell's drafter kernels already run a round in 0.24–0.68 ms. The first-draft acceptance p1 must reach ~0.75 for a +17% gain; at 0.5 speculation only breaks even (§7).
7. **Build the drafter by surgery on today's DSpark head plus a distillation finetune, with no pretraining** (§7). Use the file format in §9.
8. **Optional, measured: dense weights as group-wise INT4.** Attention and the unembed at INT4 g128 (4.125 bpw) cut the dense chain from ~10.24 to 6.98 ms/token (−3.26 ms) for +43 MiB. Inkwell adopts it only if KLD and top-1 are equal or better than all-EXL3 (§5.4).

## 2. What Inkwell is

- Inkwell is a custom inference runtime. A Rust host drives ggml/CUDA over FFI, and hand-written CUDA kernels decode EXL3 trellis weights directly inside the matrix-vector products. It uses CUDA graphs and decodes one token at a time.
- **Target GPU:** NVIDIA GTX 1060 6 GB. Pascal sm_61, 10 SMs, ~204 GB/s DRAM, PCIe 3.0 (~8.6 GB/s measured for staging), no tensor cores. The runtime should also run well on RTX 30/40/50 cards.
- **Model:** `thinkingmachines/Inkling-2-Mini`, `Inkling2LoopForConditionalGeneration`.
  - 24 layers. Each has 128 experts, top-8 routing.
  - Layers 0–5 run once, layers 6–17 run 3 times (the loop), and layers 18–23 run once. That gives 48 layer visits per token.
- **Current quant:** "SAGE REXL3 2.26 bpw", a mixed-precision EXL3 quant.
  - Size: 8.00 GiB. Experts take 6.130 GiB (2.153 bpw on average, including suh/svh). Loop-layer experts take 3.441 GiB (2.42 bpw including suh/svh). The dense trellis takes 0.488 GiB and the unembed 221.3 MiB.
  - Quality: mean KLD 0.0475, top-1 agreement 94.66%.
- **Current speed (2026-11-24):** ~41 ms/token (~24 tok/s) at short context, on a desktop with apps running. Master measured 42.98 ms after per-layer cache sizing; a host-latency pass then measured −2.06 ms in its own A/B on master. Earlier sections of this brief that quote 46.8 ms predate both.

## 3. How the runtime uses the quant

### 3.1 Format requirements

- **Codebook.** Every expert and dense linear layer must be EXL3 with the mul1 codebook. The kernels hardcode the multiplier `0x83DCD12D` and the fp16 constants KINV = `0x1eee` and KOFF = 1024·KINV + `0xc931`. A different codebook needs new kernels.
- **Supported K.** The fast stream-decode kernels support exactly K ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5}. Half-K tiles count, since that is how 1.5, 2.5 and 3.5 are encoded. Any other K would fall off the fast path, so check with the Inkwell owner before using one.
- **K is read per tensor.** It comes from each trellis tensor's shape at load. Mixed K across tensors is supported.
- **Tensor names.** Experts are `...mlp.experts.{e}.{gate_proj,up_proj,down_proj}.{trellis,suh,svh}`. Keep the current naming and safetensors layout.
- **Other tensors.**
  - Embeddings are dequantized on the host one row at a time, so their size does not matter for VRAM.
  - The ctrl and router weights are BF16 in the file. The runtime now keeps them as BF16 on device (done; it saved ~76 MiB against widening them to F32). Keep them BF16.

### 3.2 Where each expert lives

The runtime splits a fixed VRAM budget for experts as follows:

1. **Resident layers.** All 128 experts of the layer stay in VRAM. Layers are chosen by visits per byte, so loop layers come first. Today layers 6–15 are resident, within a budget of 3,121,348,608 B (2.907 GiB). That budget is tuned to the current quant.
2. **Cached layers.** Every other layer gets an LFU expert cache in VRAM. Each layer's cache size comes from a measured table: replayed expert-ID traces give a miss-bytes curve per layer, and the budget goes where it saves the most bytes. That covers 14 layers today, 940 MiB in total; the loop layers L16 and L17 get 156 MiB each, the others 44–64 MiB. Slots are packed by each expert's actual size, so smaller experts fit more of them.
3. **Misses.** An expert that is neither resident nor cached is copied from pinned host memory over PCIe during the token. With the per-layer table the caches hit ~81% of selections, which leaves ~28 misses and ~53 MiB copied per token (the uniform sizing it replaced: 73.9%, ~37.6 misses, ~64 MiB).
4. **The table is tied to this quant.** A new quant that makes L16/L17 resident drops their entries by itself; Inkwell re-runs the trace replay to re-size the rest.

### 3.3 Measured VRAM limits

- **The process cap is 5,120 MiB.** The owner keeps the display reserve as a hard safety margin. The best config peaks at 5,052 MiB.
- **Free VRAM after load:** 150–435 MiB.
- **OOM thresholds observed** (re-checked against the logs):
  - a crash in `cudaGraphInstantiate` with ~96 MiB free;
  - a failed 16 MiB allocation with ~44 MiB free.
  Runs passed with ~209 MiB free. Today's runtime needs that free space for compute buffers and graph instantiation.
- **k = 28 caches OOM, and k = 24 sits at the edge.**
- **Pinning all 12 loop layers at today's quant was slower.** It left room for only 6 caches and ran at 62.64 ms/token, against 57.85 ms with the default placement (+8%). The loop layers have to get smaller before pinning them pays off.

### 3.4 Cache slots follow expert size

- Since 2026-11-24, cache slots are allocated by each expert's actual size; before, every slot was the size of the layer's largest expert.
  - The hit rate rose from 66% to 74% in the same VRAM.
  - Decode got 2.3% faster.
- Mixing K inside a layer no longer wastes cache VRAM.
- Staging misses still cost ~8.8 ms per token on the critical path (measured by skipping the copies). Every byte cut from non-resident experts helps.

## 4. Why not an int8-optimized codebook (TQIR-INT8 / CPTC-INT8)

1. **It does not touch the expensive part.**
   - The GEMV issues 9.47 instructions per weight today. The multiply is only ~2 of them (I2F + FFMA). The rest is window extraction and the mul1 hash.
   - An int8 codebook still extracts and hashes, so it saves at most the multiply part.
2. **mul1 weights do not fit in int8.** The decoded values span ±487 to ±510. TQIR would be a new quant constrained to int8, with unknown quality.
3. **It needs int8 activations.** `dp4a` needs both operands in int8. In Inkwell, int8 activations with EXL3 broke the gate/up output, and the change was reverted.
4. **CPTC reads a lookup table per weight.** That costs ≥7.5 instructions per weight on Pascal, which is worse than the exact mul1 rework (6.75).
5. **Measured:** another agent benchmarked the int8-optimized format, and it came out slightly slower than mul1.
6. **On RTX 30–50, int4, int8 and fp16 also lose.** They carry 2–7× more bytes per weight, and decode there is bandwidth-bound.

## 5. What a new quant should aim for

There are two expert layouts, and they reward different bpw choices.

### 5.1 The master spec's 128K plan (Recipe A-Strict)

The 12 loop layers stay resident in VRAM. The experts of the 12 non-loop layers (L0–5, L18–23) are copied over PCIe on every token. There are no expert caches. Its quant targets, with sizes re-derived from the safetensors headers:

| Component | Spec target | Spec MiB | Today (SAGE 2.26) | Change needed |
|---|---|---|---|---|
| Loop experts L6–17 (12,230.59M params) | 2.25 bpw | 3,280.5 | 3,523.6 at 2.42 bpw | −243 MiB |
| Attention, 24 layers (645.81M params) | 4–5 bpw | 362.0 | 278.7 at 3.63 bpw | +83 MiB |
| lm_head (460.93M params) | NariQ 3.5 bpw | 192.4 | 221.3, EXL3 trellis | −29 MiB; see note below |
| Message gates, routers, convs/norms | BF16 | 73.6 | 148 as F32 on device | runtime change, not quant |

- **Every MiB in the loop matters.** The plan's total is 5,095–5,115 MiB against a 5,120 MiB cap, a margin of 5–25 MiB. It holds only if the runtime overhead stays at 200 MiB or less. Today's runtime takes 387–518 MiB, so Inkwell has runtime work to do too.
- **Non-loop expert bpw sets the PCIe cost.** These layers stream 764.4M params per token, 207.0 MB at today's bpw. At ccopy's 8.6 GB/s that takes 24.1 ms per token; at copy-engine DMA's 12.4–12.6 GB/s (measured) it takes 16.6 ms.
  - Each 0.1 bpw on those experts is ~9.6 MB per token, ~1.1 ms per token with ccopy (estimate).
  - Quality there trades directly against speed. Measure KLD per step of bpw.
- **lm_head format.** Inkwell has no NariQ kernel; it decodes only EXL3 mul1. Either ship the head as EXL3 mul1 at about the same bpw, or tell the Inkwell owner so a kernel can be planned. NariQ saves 28.9 MiB over today's head.
- **KV.** The plan uses K4V2 KV (864 MiB at 128K). Inkwell has no K4V2 kernel yet, and its quality is not measured.
- **Expected decode speed at 128K under this plan** (reviewer estimate): 12–13 tok/s with today's kernels. After the planned kernel work it reaches 15–20 tok/s, and 23–31 tok/s only if all PCIe traffic is hidden.

### 5.2 Today's Inkwell layout (short context, expert caches)

Today Inkwell keeps 2.907 GiB of experts resident: loop layers first, layers 6–15. Every other layer gets an LFU cache of 20 slots. For a quant aimed at this layout:

- To fit all 12 loop layers within 2.907 GiB and keep the 14 caches, the loop needs ≈2.0 bpw including suh/svh (≈1.9 bpw trellis). This is an estimate, scaled from 3.441 GiB at 2.42 bpw.
- At 2.2–2.25 bpw the loop fits, but the extra ~0.2–0.3 GiB comes out of the caches, about 3–5 of them.

### 5.3 For both layouts

- **K choice and kernel speed (measured).** Inkwell's newer exact decode core, measured inside the real MoE kernel against the old core:

  | K | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 5.0 |
  |---|---|---|---|---|---|---|---|
  | kernel time change | slower | −6 to −8% | 0 to −5% | −4 to −8% | slower | −7 to −9% | slower |
  | uses the new core | no | yes | yes | yes | no | yes | no |

  - Over a whole token at today's K mix, the new core saves ~1.07 ms of kernel time (nsys). It is in Inkwell but still off by default, pending a cleaner A/B.
  - Today the loop experts spread over K1.5 21%, K2 22%, K2.5 27%, K3 19%, K3.5 8%, K4 3%. The non-loop experts are mostly K2 (61%) and K1.5 (33%).
  - VRAM and PCIe bytes outweigh this. Choose K for quality and size first; when two choices tie, take K 2.0, 3.0 or 4.0.

- **Quality cost (estimate).** Lowering the loop from 2.42 to ~2.2 bpw is expected to cost KLD +12–25% (0.0475 → ~0.053–0.060). This is extrapolated from exllamav3's KLD-vs-bpw slope. The loop layers run 3 times, so their error may weigh more. Measure it.
- **Keep the 2.26 bpw SAGE quant for cards with 8–12 GB or more.** The low-bpw variant is for 6 GB cards.

### 5.4 Optional: dense weights as group-wise INT4 (needs your quality check)

This applies only to the dense, non-expert weights: attention (645.81M params, 278.7 MiB today at ~3.6 bpw) and the unembed (461M params, 221 MiB at K=4). The experts stay EXL3.
- **Measured kernel speed (2026-11-24, `opt/kbench/w4dense/REPORT.md` in the Inkwell repo):**

  | Format | bpw | ms/token, dense chain | vs EXL3 today | VRAM Δ |
  |---|---|---|---|---|
  | EXL3 today | ~3.79 | ~10.24 | — | 0 |
  | INT4 g128 symmetric | 4.125 | 6.98 | −3.26 ms | +43 MiB |
  | INT4 g128 asymmetric | 4.25 | 7.09 | −3.18 ms | +60 MiB |
  | INT4 g64 symmetric | 4.25 | 7.24 | −3.19 ms | +60 MiB |
  | INT5 g128 | 5.125–5.25 | 8.12 | −2.20 ms | +175 MiB |
  | INT6 g128 | 6.125–6.25 | 9.28–9.52 | −1.0 ms | +307 MiB |

  Activations stay F32. INT4 runs within 5% of the minimum instruction count per weight on Pascal, and the whole step is bandwidth-bound. INT5/INT6 cost too much VRAM to use everywhere; use them only on single tensors that need them.
- **The open question is quality.** Group-wise INT4 at ~4.1–4.25 bpw is not obviously as good as an EXL3 trellis at ~3.6 bpw.
- **Ask:** quantize the dense weights with GPTQ (or AWQ), with the experts unchanged. Try in order: INT4 g128 symmetric, then g128 asymmetric and g64 symmetric, then INT5/INT6 on the most sensitive tensors only. Report mean KLD and top-1 against the source, next to the all-EXL3 quant measured with the same harness (today 0.0475 and 94.66%). Inkwell adopts it only if KLD and top-1 are equal or better.
- **File format:**
  - Store each matrix row-major `[N_out, K_in]`, with no Hadamard tensors. Inkwell re-tiles it at load.
  - Tensors per projection (the 6 attention projections per layer and `lm_head`):
    - `{prefix}.qweight`: U8 `[N_out, K_in/2]`.
    - `{prefix}.qweight_hi1`: U8 `[N_out, K_in/8]`, W5 only.
    - `{prefix}.qweight_hi2`: U8 `[N_out, K_in/4]`, W6 only.
    - `{prefix}.scales`: F16 `[N_out, K_in/G]` for symmetric, `[N_out, K_in/G, 2]` = {scale, min} for asymmetric.
  - Header metadata `quant_type`, one of `w4_g128_sym`, `w4_g128_asym`, `w4_g64_sym`, `w4_g64_asym`, `w5_g128_sym`, `w5_g128_asym`, `w6_g128_sym`.
  - Groups run along K_in. With `u` unsigned in [0, 2^b − 1]:
    - symmetric: `w = scale · (u − 2^(b−1))`;
    - asymmetric: `w = scale · u + min`.
  - Nibble packing, for every 8 weights `u[0..7]` of a row: byte `b` (0–3) holds `u[b]` in its low nibble and `u[4+b]` in its high nibble. In NumPy, for `u` shaped `[N, K/8, 8]`:
    ```python
    lo4 = u & 0x0F
    qweight = (lo4[..., 0:4] | (lo4[..., 4:8] << 4)).astype(np.uint8).reshape(N, K // 2)
    ```
  - The W5/W6 upper-bit planes are specified in §6.3 of the report.

## 6. Context length: 128K re-check (done)

A reviewer re-derived the master spec's full 128K VRAM budget from the safetensors headers, the model config and measured probes.

- **Verdict:** 128K fits under the spec's 5,120 MiB cap with 5–25 MiB to spare. Against the card's physical limit (6,057 MiB visible to CUDA, minus the desktop's 494–512 MiB) the margin is 430–468 MiB.
- **Conditions:**
  1. The runtime fits in 200 MiB, allocated from one pool at load. The bare CUDA context is 65 MiB.
  2. The three requants in §5.1 ship.
  3. K4V2 KV kernels exist and pass a KLD check.
  4. The 123.6 MiB batch-verify scratch is removed.
  5. MTP is off, or the spec's reduced MTP recipe (−60.75 MiB) applies.
  6. The desktop stays under 937 MiB.
- **Corrections to the spec:**
  - The display reserve is 937 MiB, not 1,024: the driver reserves 87 MiB.
  - The conv ring takes 3.66 MiB, not 3.10.
  - Two lines are missing: the single-row GEMV scratch (15.7 MiB) and per-tensor allocation overhead (~100 MiB unless pooled). In total the missing lines add a net +15.3 MiB.
- **KV reference:** Q4 KV costs 8,704 B/token over the 8 full-attention passes. At 128K, reading the KV costs an estimated 5–10 ms per token on the 1060.
- **Full derivation:** §2.5 of `opt/REVIEW_ARCH.md` in the Inkwell repo.

## 7. Drafter: acceptance and plan

Inkwell plans speculative decoding with a small drafter. The drafter's precision never reaches the output, because the verifier checks every drafted token against the full model's argmax. A lossy drafter only lowers the acceptance rate, so int8 and int4 weights are allowed in it.

### 7.1 Acceptance decides the gain

Modeled speed against the 46.8 ms/token serial speed of the time (the runtime is now ~41 ms; the ratios still apply), with p1 = P(first draft accepted) and q = P(each later draft accepted). Draft rounds take 0.24 ms (1 draft) to 0.68 ms (3 drafts), measured.

| p1 | q | best k (drafts per round) | ms/token | vs serial |
|---|---|---|---|---|
| 0.50 | 0.18 | 1 | 46.8 | +0% |
| 0.75 | 0.18 | 1 | 40.1 | +17% |
| 0.75 | 0.46 | 1 | 40.1 | +17% |
| 0.82 | 0.65 | 1–2 | 38.5 | +21% |
| 0.90 | 0.80 | 2–3 | 34.3–34.7 | +35–37% |

- The model assumes a batched verifier that Inkwell has not built yet. Verifying 2 rows costs ~1.5× a serial step, because the rows route to different experts: the mean union is 12.9 experts per layer instead of 8 (measured from real routing). With 3 rows it is 16.6, so a third row rarely pays.
- Today's checkpoint drafter (DSpark) measured p1 ≈ 0.50–0.75 and q ≈ 0.18–0.46, depending on the prompt and the sample.
- **Train for p1 first.** With one draft per round, every +0.1 of p1 is worth ~+6–7% speed. A higher q pays only once p1 is above ~0.8.

### 7.2 Plan: surgery on DSpark, then a distillation finetune

No pretraining.

1. **Keep the width at D = 2304**, so DSpark's weights still fit. Cut 3 layers to 1 and prune the FF from 9216 to 4608.
2. **Draft 1 token per round (k = 1).** Keep 1 mask row instead of 7; the GRU recurrence then matters little.
3. **Head:** 32K rows taken from the target's own unembed, int4 with group-128 scales (§8.3).
4. **Body:** int8 with one scale per output row (§8.3).
5. **Finetune data:** run the target in prefill mode (teacher forcing) over text. Record the 4 taps and the target's argmax at every position, and train the drafter to predict that argmax.
6. **Metric:** p1 on held-out text, before and after the finetune. Below ~0.6 the drafter does not pay.

Measured kernel cost of this shape: 0.55 ms per round and 81.6 MiB; modeled gain at p1 = 0.75 is ~+11%. A drafter trained from scratch at D = 1024 would cost 0.24 ms and 29.9 MiB (~+13%); it is the fallback if surgery cannot reach the p1 target.

**VRAM.** In today's layout, every drafter MiB comes out of the expert caches; the model above charges for it. The 128K plan has no room for a drafter (§6, condition 5).

## 8. Drafter: measured results

All kernel times are measured on the GTX 1060, inside one CUDA graph with no host sync. End-to-end speeds are modeled.

### 8.1 Head options

| Head | Extra VRAM | Time, 1 row |
|---|---|---|
| Target's resident EXL3 unembed, full 200,058 vocab | 0 MiB | 1.76 ms |
| int8, V′ = 32K, D = 2304 | 72.2 MiB | 0.43 ms |
| **int4 group-128, V′ = 32K, D = 2304** | **37.2 MiB** | **0.24 ms** |
| int8, V′ = 16K, D = 2304 | 36.1 MiB | 0.22 ms |
| int4 group-128, V′ = 32K, D = 1024 | 16.6 MiB | 0.11 ms |

The heads fuse GEMV, argmax and the id remap in one launch and reach 160–176 GB/s, 78–86% of DRAM peak.

### 8.2 Vocabulary coverage

Coverage = the share of the target's real greedy tokens (640 tokens over 5 prompts: prose, Python, Rust, technical, factual) inside the top-V′ set. The ranking came from a separate 4.12M-token corpus with no overlap with the prompts.

| V′ | Frequency-ranked | Naive ids 0..V′−1 |
|---|---|---|
| 8K | 77.8% | 32.2% |
| 16K | 86.7% | 32.2% |
| **32K** | **94.8%** | 33.1% |
| 64K | 95.9% | 35.6% |

- **A frequency-ranked id table is required.** Low ids hold CJK tokens, while common English words with a leading space sit at ids 149,860–199,999 (" the" = 149876).
- **V′ = 32K is the knee:** +8.1 points over 16K, and only +1.1 more at 64K for twice the VRAM.
- Rank by frequency in your distillation corpus. Pin the 256 byte tokens and EOS (200006) first.

### 8.3 Quantization cost on today's DSpark (acceptance proxy)

Agreement = how often the quantized DSpark picks the same token as the fp32 DSpark, on real taps. **It is not acceptance by the target.** In the same test the target accepted 6 of 12 first drafts (50%). The sample is small: 12 rounds.

| Quantization (whole drafter, V′ = 32K) | 1st draft | drafts 1–3 | drafts 1–7 |
|---|---|---|---|
| int8 weights + int16 activations (dp2a) | 12/12 | 36/36 | 84/84 |
| int8 weights + int8 activations (dp4a) | 12/12 | 36/36 | 80/84 |
| int4 group-128 weights everywhere | 11/12 | 29/36 | 51/84 |
| int4 one scale per row | 7/12 | 23/36 | 36/84 |

- Keep the body int8 per row: it matched the fp32 drafter on every draft 1–3.
- int4 group-128 is fine in the head alone (12/12 and 34/36 when only the head is int4) but costs agreement when applied to the whole body.
- **Do not use int4 with one scale per row.**

## 9. Drafter: file format the Inkwell kernels expect

File `drafter.safetensors`:

- **Matrices:** row-major [N_out, K_in] (K_in contiguous). K_in and N_out are multiples of 128.
- **Body linears:** `*.weight` I8 [N_out, K_in], values in [−127, 127], plus `*.scale` F32 [N_out] = max |W_row| / 127, so W ≈ weight × scale[row]. The same format serves int8 and int16 activations.
- **Head, int4 group-128:** `lm_head.weight_i4` U8 [V′, D/2] and `lm_head.scale` F16 [V′, D/128] = max |W_group| / 7.
  - Each little-endian uint32 holds 8 consecutive weights q[0..7] in [−7, 7], stored as nibbles u = q + 8.
  - The low nibbles (`p & 0x0F0F0F0F`) hold u0–u3 (columns k..k+3). The high nibbles (`(p >> 4) & 0x0F0F0F0F`) hold u4–u7 (columns k+4..k+7).
  - Alternative: `lm_head.weight` I8 [V′, D] with `lm_head.scale` F32 [V′], like the body.
- **Vocabulary:** `vocab_ids` U32 [V′] maps head row → target token id; `inv_vocab_ids` I32 [200058] maps target id → head row, −1 when out of set.
- **Taps:** `tap.fuse_weight` F32 [4] (softmax weights over the 4 tapped visits), `tap.norm.weight` F32 [2304], `tap.proj.{weight,scale}` I8 [D, 2304] + F32 [D] (omit the projection when D = 2304 and it is the identity).
- **Per layer l:** `layers.{l}.in_norm.weight` F32 [D]; `layers.{l}.attn.qkv_proj.{weight,scale}` (fused Q/K/V); `layers.{l}.attn.q_norm.weight` and `k_norm.weight` F32 [head_dim]; `layers.{l}.attn.o_proj.{weight,scale}`; `layers.{l}.post_norm.weight` F32 [D]; `layers.{l}.mlp.gate_up_proj.{weight,scale}` (fused gate/up); `layers.{l}.mlp.down_proj.{weight,scale}`.
- **Final:** `final_norm.weight` F32 [D], then the head. An untied input embedding, if used, is `embed.{weight,scale}` I8 [V′, D] + F32 [V′].
- **Norms and biases:** F32 (BF16 is widened at load).

## 10. What to send back

1. **A per-tensor table:** name, K and bytes. Include the per-layer totals for experts, split into loop and non-loop layers.
2. **Quality:** mean KLD and top-1 agreement against the source checkpoint (float64 logits, full vocabulary), plus the same figures for the current 2.26 bpw quant measured with your harness.
3. **The total file size** and the bytes of the 12 loop layers' experts, including suh/svh.
4. **If you build a drafter:** a `drafter.safetensors` in the §9 format, its dims (D, layers, FF, heads, V′), and p1 and q on held-out text against the target, before and after the finetune.
5. **If you test §5.4:** KLD and top-1 for each dense format, and the per-tensor bits you chose.

Inkwell will measure speed with its own A/B protocol: ≥8 interleaved pairs, generated-text checks, and the resident budget set through `INKWELL_DEV_EXPERT_MB`.
