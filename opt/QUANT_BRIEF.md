# Quantization brief for Inkling-2-Mini on Inkwell (GTX 1060 6 GB)

This document is self-contained. It tells the quantization agent what the Inkwell runtime needs from a new quant and what measurements say about bits per weight. Every number comes from a measurement in the Inkwell repo (`opt/ledger.md`, `opt/THEORY.md`, `opt/REVIEW_ARCH.md`) unless it is marked as an estimate.

## 1. The bottom line

1. **Keep the EXL3 mul1 trellis.** Do not move to an int8-constrained codebook (TQIR-INT8, CPTC-INT8), int4, int8 or fp16. Section 4 gives the reasons.
2. **On the GTX 1060, a lower bpw does not make the GEMV kernels faster per weight.** They are limited by instruction issue, not memory bandwidth. Bits per weight change speed only through which experts fit in VRAM and how many bytes cross PCIe per token.
3. **The loop layers (6–17) are where bytes matter most.** They run 3 times per token. Today their experts take 3.441 GiB, and that does not fit in VRAM alongside the caches the runtime needs.
4. **Keep the experts within one layer at one K, or at as few K values as possible.** Section 3.4 explains why.
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
2. **Cached layers.** Every other layer gets an LFU cache of k = 20 expert slots in VRAM. That covers 14 layers today, ~811 MiB in total.
3. **Misses.** An expert that is neither resident nor cached is copied from pinned host memory over PCIe during the token.
   - Today's miss rate is 18.6%, which is ~83 MiB copied per token.
   - The copies take 9.84 ms of GPU time per token. How much of that delays the token is being measured now.

### 3.3 Measured VRAM limits

- **Free VRAM after load:** 150–435 MiB.
- **OOM thresholds observed:** 126 MiB free (graph instantiate), 153 MiB free, and a failed 16 MiB graph allocation. Plan for at least ~200 MiB of headroom.
- **k = 28 caches OOM, and k = 24 sits at the edge.**
- **Pinning all 12 loop layers at today's quant was slower.** It left room for only 6 caches and ran at 62.64 ms/token, against 57.85 ms with the default placement (+8%). The loop layers have to get smaller before pinning them pays off.

### 3.4 Why one K per layer matters (today)

- Each cached layer's slots are all the size of the largest expert in that layer.
- Over the 14 cached layers, the sum of the per-layer maximum K is 42.5, while the sum of the mean K is 28.3. Mixed K inside a layer therefore wastes ~1/3 of cache VRAM.
- An Inkwell change that sizes slots per K class is being built now. Until it lands, mixing K within a layer's experts costs speed.

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

Ranked by expected speed impact on the 6 GB card:

1. **Shrink the loop-layer experts (layers 6–17).** Each byte there is read 3 times per token, so this matters most.
   - To fit within today's resident budget (2.907 GiB) with all 14 caches intact, the 12 loop layers need ≈2.0 bpw including suh/svh. That is ≈1.9 bpw of trellis, an estimate scaled from 3.441 GiB at 2.42 bpw.
   - At 2.2 bpw the loop takes ~3.13 GiB (estimate). All 12 loop layers then fit, but the extra ~0.22 GiB comes out of the caches, about 3–4 of them.
   - Expected gain for a 2.2 bpw loop fit: −3 to −5 ms/token (estimate).
   - Expected quality cost: KLD +12–25%, from 0.0475 to ~0.053–0.060. This is extrapolated from exllamav3's KLD-vs-bpw slope, and the loop layers may weigh more because they run 3 times. It must be measured.
2. **Use a uniform K per layer for the non-loop experts**, so cache slots are not wasted (§3.4).
3. **Leave the dense trellis, attention and unembed alone.** Moving attention to int4/int5 g32 would save an estimated 1–2.5 ms on the 1060, but it costs +70–150 MiB of VRAM and loses on RTX.
4. **Keep the 2.26 bpw SAGE quant for cards with 8–12 GB or more.** The low-bpw variant is for 6 GB cards.

## 6. Context length (pending verification)

- The master spec plans 128K context on 6 GB with Q4/K4V2 KV and states 932.85 MiB for the KV and attention state at 128K.
- A reviewer is now re-checking the spec's full VRAM budget at 128K against measured sizes. Treat the 128K fit as unverified until that check is back. This section will be updated with the result.
- Reference figures:
  - Q4 KV costs 8,704 B/token for the 8 full-attention passes, or 9,792 B/token with the draft layer.
  - Fixed sliding-window and convolution state is ~0.17 GiB.
  - Reading the KV at 128K adds an estimated ≥4 ms/token on the 1060.

## 7. What to send back

1. **A per-tensor table:** name, K and bytes. Include the per-layer totals for experts, split into loop and non-loop layers.
2. **Quality:** mean KLD and top-1 agreement against the source checkpoint (float64 logits, full vocabulary), plus the same figures for the current 2.26 bpw quant measured with your harness.
3. **The total file size** and the bytes of the 12 loop layers' experts, including suh/svh.

Inkwell will measure speed with its own A/B protocol: ≥8 interleaved pairs, generated-text checks, and the resident budget set through `INKWELL_DEV_EXPERT_MB`.
