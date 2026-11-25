# Challenge: pick the exact next token without reading the whole output head, proved in Lean 4

This is a math and algorithms problem that can be worked on a CPU alone. The deliverable is a certified scheme with a Lean 4 soundness proof, a CPU implementation, and measurements on the real data we supply. The Inkwell team ports and measures the result on the GPU.

**Context.** Inkwell is an inference engine for a language model on a 6 GB GPU. This problem is about the last step of each decoding step: choosing the next token from 200,058 candidates. There is no security aspect.

## 1. Why it matters
- The output head is a 200,058 × 2304 matrix stored as EXL3 at 4 bits per weight: 221 MiB of VRAM.
- It costs 1.76 ms per token at batch 1, and 2.11 / 2.61 / 3.00 ms when it scores 2 / 3 / 4 positions at once for speculative decoding.
- Decoding is greedy. **Only the index of the largest logit is used**, never the full logit vector.
- The process has a hard 5,120 MiB VRAM cap. Every MiB freed goes to expert caches. Expert misses cost ~8.8 ms per token at 512 tokens of context. At 32K context the KV cache takes 1.19 GiB and the expert caches drop to zero, so freed VRAM matters even more.

**Goal.** Return exactly the same token as today on every step, while reading far fewer head bytes. Then the full head can either stay in VRAM and run faster, or move to host RAM and free ~180 MiB of VRAM.

## 2. The exact computation
Input: x ∈ ℝ^2304 in fp32, the normalised final hidden state.

1. **Input rotation:** x̃ = (1/√128) · H₁₂₈(suh ⊙ x), applied blockwise on 18 blocks of 128. H₁₂₈ is the ±1 Walsh–Hadamard matrix, and suh ∈ {fp16}^2304 is a stored sign/scale vector.
2. **Trellis product:** t = Wᵀ x̃, with t ∈ ℝ^201088 (201,088 is the padded vocabulary). W is 2304 × 201088. Every entry is an fp16 value decoded from the EXL3 trellis (see `opt/MATH_CHALLENGE.md` §2 for the decode).
3. **Output rotation:** for each of the 1,571 output blocks b of 128 columns, z_b = (1/√128) · H₁₂₈ t_b ⊙ svh_b, where svh ∈ {fp16}^201088.
4. **Softcap:** ℓ_i = 30 · tanh(z_i / 30), for i < 200,058.
5. **Token:** A(ℓ), the shipped argmax. Its tie rule is fixed and depends only on the set of indices that attain the maximum.

Consequences of this structure:
- Each logit depends on exactly one output block of 128 columns. Computing one block exactly reads 128 × 2304 weights, ~147 KB.
- Blocks are independent. Running the shipped kernels on a subset of blocks gives bit-identical values for those blocks.

**Rounding.** The GPU computes steps 1–4 in fp32 (FMA available), so the computed value ℓ̂_i differs from the real-number ℓ_i. The certificate must bound ℓ̂, not ℓ.
- Use the standard model: fl(a ∘ b) = (a ∘ b)(1 + δ) with |δ| ≤ 2⁻²⁴.
- Use a summation bound valid for **any** order, such as |fl(Σ) − Σ| ≤ γ_{n−1} Σ|terms| with γ_n = nu / (1 − nu). Then the exact reduction tree of the kernel does not matter.
- CUDA `tanhf` has a maximum error of 2 ulp, and its monotonicity is not guaranteed. Treat both as stated hypotheses.

## 3. The ask
Design a scheme with three parts.

1. **Sketch.** An offline, precomputed summary S of the head, at most 40 MiB. It may take any amount of offline compute.
2. **Per-token pass.**
   - Compute an upper bound U_b(x) ≥ max_{i∈b} ℓ̂_i for every block b, using x and S.
   - Pick a candidate set C of blocks, and compute ℓ̂ exactly on C with the shipped kernels.
   - Let L = max over the candidates of ℓ̂.
   - **Certify** U_b(x) < L for every block b ∉ C.
   - If the certificate fails, enlarge C deterministically. In the worst case C contains all blocks.
3. **Result.** A applied to the candidates equals A applied to everything, exactly, because every maximum lies inside C.

**Success:**
- S ≤ 40 MiB.
- The bound pass costs ≤ 0.3 ms on the GPU, estimated as bytes / 170 GB/s plus instruction count.
- On the supplied data, the 99th percentile of |C| is ≤ 32 blocks, with a reported maximum.
- If the full head moves to host RAM, every candidate block crosses PCIe at ~11.5 GB/s (~13 µs per block). A full fallback costs ~20 ms, so it must occur on < 0.1% of tokens.

**A negative answer is also useful:** a proof, or a measurement on the supplied data, that no sketch of a stated family under 40 MiB can certify with a small C.

Directions, none of them known to work:
- low rank plus residual: t ≈ Uᵀ(Vx̃), with the residual bounded per block (Cauchy–Schwarz or something tighter);
- a coarse 1–2 bit copy of W with rigorous per-block error bounds of the form Σ_g ‖E_{b,g}‖ · ‖x̃_g‖;
- tree or cone bounds from exact maximum inner product search;
- the output Hadamard: bounds on t_b map to bounds on z_b through the rows of H₁₂₈;
- getting a high L early, from the drafter's guess or the previous token;
- integer sketch arithmetic, exact under dp4a, with rigorously bounded quantisation of x.

## 4. Deliverables
1. **Lean 4 with Mathlib** (pin the toolchain and Mathlib version). No `sorry` and no new axioms.
   - **Selection theorem.** For any argmax rule that depends only on the argmax set: if every value outside C is strictly below max over C, then the rule applied to the whole vector equals the rule applied to C.
   - **Soundness theorem.** For every x, U_b(x) ≥ max_{i∈b} ℓ̂_i under the rounding model of §2, stated as explicit hypotheses. Mathlib has no IEEE float semantics, so the model goes in as hypotheses over ℝ.
   - **Parametric in the data.** The 232 MB head cannot enter Lean. State the theorem for any head W and any sketch S that satisfy a property P(W, S), such as "these stored constants bound these norms". Supply a separate checker that verifies P on the real head in exact integer or rational arithmetic, or with directed rounding.
2. **CPU code in C or Rust:**
   - a head dequantiser, validated against the GPU logits we supply;
   - the sketch builder, the bound pass, the certifier, and the checker for P.
3. **Measurements on the supplied data:**
   - the distribution of |C| (p50, p99, p99.9, max);
   - sketch bytes, bound-pass bytes, and instructions per block;
   - certified token = supplied GPU token on 100% of steps;
   - the same at batch 2–4, which is speculative verification.

## Workflow and gates
1. **Cheap falsification first.** Test each idea on CPU against the reference before you formalise it. Most wrong ideas die in seconds on random or exhaustive inputs.
2. **Lean proof.** Formalise only the ideas that survive step 1. The proof is the gate: `lake build` passes, there is no `sorry`, and `#print axioms` on the main theorem shows only the standard axioms (propext, Classical.choice, Quot.sound), plus `Lean.ofReduceBool` if `native_decide` is used and declared.
3. **Final CPU implementation.** It must implement exactly the algorithm that the Lean file defines, and pass the tests above against the reference.
4. **Port (done by the Inkwell team).** The CUDA/Rust port starts only after step 2 is checked on our machine. Then it goes through the usual interleaved A/B on the GTX 1060.

## 5. Data supplied
- `unembed.trellis`: int16 [144, 12568, 64]. There are 144 × 12,568 tiles of 16 × 16 weights at K = 4 (1,024 bits per tile).
- `unembed.suh`: fp16 [2304]. `unembed.svh`: fp16 [201088]. The constant mul1 = 0x83DCD12D.
- **Tile layout.**
  - The window extraction and decode are given in `opt/MATH_CHALLENGE.md` §2.
  - Ring index i maps to tile position (r, c): with tt = i >> 3 and j = i & 7, r = (tt & 3)·2 + (j & 1) + ((j & 2) ? 8 : 0), and c = (tt >> 2) + ((j & 4) ? 8 : 0). Here r indexes k (input) and c indexes n (output). Source: `opt/kbench/exl3_math/test_cpu.cpp`.
- **Real hidden states.** Several thousand final x vectors (fp32) from greedy decoding of varied prompts: prose, code, technical, factual and Portuguese. Each comes with the GPU's chosen token and its top-16 logits, so the dequantiser can be checked.

### Head-dump binary format
`INKWELL_DUMP_HEAD=<path>` creates or truncates `<path>` and writes one fixed-size record for the last prompt step and for every serial decode step whose logits select a token. When the variable is unset the runner does not perform the dump readbacks. All integer and floating-point fields are little-endian; fp32 fields are their IEEE-754 bit patterns.

Header (24 bytes):

| offset | type | value |
|---:|---|---|
| 0 | byte[8] | ASCII `INKWHEAD` |
| 8 | u32 | format version = 1 |
| 12 | u32 | x dimension = 2304 |
| 16 | u32 | unpadded vocabulary = 200058 |
| 20 | f32 | softcap = 30.0 |

Each record is 9,352 bytes and starts at offset 24:

| offset | type | meaning |
|---:|---|---|
| 0 | u32 | absolute position `pos` of the input token processed by this step |
| 4 | u32 | token id chosen greedily from this step's logits |
| 8 | f32[2304] | x, the normalised final hidden state entering the output head, before `suh` and the input Hadamard |
| 9224 | 16 × (u32, f32) | top-16 post-softcap logits as `(token index, logit)`, sorted by descending logit |

`opt/tools/head_dump_check.py` validates the header, record size, finite x values, top-1 agreement and descending top-16 order.
