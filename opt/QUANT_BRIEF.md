# Quantization brief for Inkling-2-Mini on Inkwell (GTX 1060 6 GB)

This document is self-contained. It tells the quantization agent what the Inkwell runtime needs from a new quant and what measurements say about bits per weight. Every number comes from a measurement in the Inkwell repo (`opt/ledger.md`, `opt/THEORY.md`, `opt/REVIEW_ARCH.md`) unless it is marked as an estimate.

## 1. The bottom line

1. **Keep the EXL3 mul1 trellis.** Do not move to an int8-constrained codebook (TQIR-INT8, CPTC-INT8), int4, int8 or fp16. Section 4 gives the reasons.
2. **On the GTX 1060, a lower bpw does not make the GEMV kernels faster per weight.** They are limited by instruction issue, not memory bandwidth. Bits per weight change speed only through which experts fit in VRAM and how many bytes cross PCIe per token.
3. **The master spec's 128K plan fits on paper, with 5–25 MiB to spare.** Its expert layout is "loop layers resident, other layers streamed every token" (§6). Under that plan:
   - the loop-layer bpw decides whether the plan fits at all;
   - the non-loop expert bpw sets the PCIe bytes per token (~1.1 ms per 0.1 bpw).
4. **Prefer K = 2.0 for the bulk of the experts, and use as few distinct K values per layer as possible.** The decode kernels are fastest on K=2.0 and slowed by mixed-K launches (§5.3).
5. **Report quality the same way as the current quant:** mean KLD and top-1 agreement against the source, using float64 logits over the full vocabulary.

## 2. What Inkwell is

- Inkwell is a custom inference runtime. A Rust host drives ggml/CUDA over FFI, and hand-written CUDA kernels decode EXL3 trellis weights directly inside the matrix-vector products. It uses CUDA graphs and decodes one token at a time.
- **Target GPU:** NVIDIA GTX 1060 6 GB. Pascal sm_61, 10 SMs, ~204 GB/s DRAM, PCIe 3.0 (~8.6 GB/s measured for staging), no tensor cores. The runtime should also run well on RTX 30/40/50 cards.
- **Model:** `thinkingmachines/Inkling-2-Mini`, `Inkling2LoopForConditionalGeneration`.
  - 24 layers. Each has 128 experts, top-8 routing.
  - Layers 0–5 run once, layers 6–17 run 3 times (the loop), and layers 18–23 run once. That gives 48 layer visits per token.
- **Current quant:** "SAGE REXL3 2.26 bpw", a mixed-precision EXL3 quant.
  - Size: 8.00 GiB. Experts take 6.130 GiB (2.153 bpw on average, including suh/svh). Loop-layer experts take 3.441 GiB (2.42 bpw including suh/svh). The dense trellis takes 0.488 GiB and the unembed 221.3 MiB.
  - Quality: mean KLD 0.0475, top-1 agreement 94.66%.
- **Current speed:** 47.7 ms/token (~21 tok/s) at short context, on a desktop with apps running.

## 3. How the runtime uses the quant

### 3.1 Format requirements

- **Codebook.** Every expert and dense linear layer must be EXL3 with the mul1 codebook. The kernels hardcode the multiplier `0x83DCD12D` and the fp16 constants KINV = `0x1eee` and KOFF = 1024·KINV + `0xc931`. A different codebook needs new kernels.
- **Supported K.** The fast stream-decode kernels support exactly K ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5}. Half-K tiles count, since that is how 1.5, 2.5 and 3.5 are encoded. Any other K would fall off the fast path, so check with the Inkwell owner before using one.
- **K is read per tensor.** It comes from each trellis tensor's shape at load. Mixed K across tensors is supported.
- **Tensor names.** Experts are `...mlp.experts.{e}.{gate_proj,up_proj,down_proj}.{trellis,suh,svh}`. Keep the current naming and safetensors layout.
- **Other tensors.**
  - Embeddings are dequantized on the host one row at a time, so their size does not matter for VRAM.
  - The ctrl and router weights are BF16 in the file. The runtime widens them to F32 today, which costs 148 MiB on device. Moving them to BF16 on device is a runtime change, not a quant change.

### 3.2 Where each expert lives

The runtime splits a fixed VRAM budget for experts as follows:

1. **Resident layers.** All 128 experts of the layer stay in VRAM. Layers are chosen by visits per byte, so loop layers come first. Today layers 6–15 are resident, within a budget of 3,121,348,608 B (2.907 GiB). That budget is tuned to the current quant.
2. **Cached layers.** Every other layer gets an LFU expert cache in VRAM, sized at 20 of the layer's largest expert. That covers 14 layers today, ~811 MiB in total. Slots are packed by each expert's actual size, so smaller experts fit more of them.
3. **Misses.** An expert that is neither resident nor cached is copied from pinned host memory over PCIe during the token.
   - Today's miss rate is 18.6%, which is ~83 MiB copied per token.
   - The copies take 9.84 ms of GPU time per token. How much of that delays the token is being measured now.

### 3.3 Measured VRAM limits

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
- Mixing K inside a layer no longer wastes cache VRAM. It still slows the decode kernels (§5.3).
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

- **Prefer K = 2.0 for the bulk of the experts, and keep the number of distinct K values per layer small.** This finding is new; the kernel port is in progress.
  - Inkwell's newer exact decode kernel is measured +14–16% faster on pure K=2.0 tiles. K=4.0 also gets the cheaper aligned-window path.
  - On a mix of all 7 K values the gain falls to +3%. One MoE launch covers experts of several K, so it runs at the pace of the slowest K. K ∈ {2.5, 3.0, 3.5, 5.0} have no cheap paired-window path.
  - Today the loop experts are spread over six K values: K1.5 21%, K2 22%, K2.5 27%, K3 19%, K3.5 8%, K4 3%. The non-loop experts are mostly K2 (61%) and K1.5 (33%).
  - A 2.25 bpw loop built mostly from K=2.0 tiles, plus a few higher-K tiles where quality needs them, would suit the kernels best. Quality decides the final split.

- **Quality cost (estimate).** Lowering the loop from 2.42 to ~2.2 bpw is expected to cost KLD +12–25% (0.0475 → ~0.053–0.060). This is extrapolated from exllamav3's KLD-vs-bpw slope. The loop layers run 3 times, so their error may weigh more. Measure it.
- **Keep the 2.26 bpw SAGE quant for cards with 8–12 GB or more.** The low-bpw variant is for 6 GB cards.

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

## 7. What to send back

1. **A per-tensor table:** name, K and bytes. Include the per-layer totals for experts, split into loop and non-loop layers.
2. **Quality:** mean KLD and top-1 agreement against the source checkpoint (float64 logits, full vocabulary), plus the same figures for the current 2.26 bpw quant measured with your harness.
3. **The total file size** and the bytes of the 12 loop layers' experts, including suh/svh.

Inkwell will measure speed with its own A/B protocol: ≥8 interleaved pairs, generated-text checks, and the resident budget set through `INKWELL_DEV_EXPERT_MB`.
