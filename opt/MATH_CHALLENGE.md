# Challenge: an exact, cheaper way to evaluate the EXL3 trellis dot product, proved in Lean 4

This is a math problem that can be worked on a CPU alone. The deliverable is a theorem with a Lean 4 proof, a CPU reference implementation with an exhaustive test, and an operation count. The Inkwell team ports and measures the result on the GPU.

## 1. Why it matters
Decoding one token of Inkling-2-Mini on a GTX 1060 takes ~38–41 ms. The largest single kernel, the MoE expert GEMV (~16 ms per token in nsys), is dot products whose weights are stored compressed as an EXL3 trellis.
- ~3.06 billion weights are decoded per token: 48 layer visits × 8 experts × 3 matrices × 2304 × 1152, plus the dense layers.
- The cost is instruction count, not memory traffic. Today it takes ~7.2 GPU instructions per weight.

## 2. The object
For one 16×16 tile and an input vector x, the GPU computes

  y[n] = Σ_k v(w_{n,k}) · x[k]

- **Bitstream.** A tile stores a cyclic bitstream ("ring") of R = 256·K bits, with K ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5} bits per weight. For half-integer K, weights alternate between ⌊K⌋ and ⌈K⌉ bits.
- **Index.** Weight i (i = 0..255) has the 16-bit index w_i: the 16 bits of the ring ending at the inclusive prefix position s_i of the bit counts, taken cyclically. Consecutive indices overlap in 16 − K bits.
- **Exact reference** for the window extraction (`crates/inkwell-cuda/src/kernels.cu`):
  ```c
  r_bits = halfk ? 256*ka + 128 : 256*ka;
  s_i    = halfk ? (i+1)*ka + ((i+1)>>1) : (i+1)*ka;      // inclusive prefix
  b0     = s_i - 16 + (s_i < 16 ? r_bits : 0);
  nw = r_bits>>5; i0 = b0>>5; s0 = 48 - (b0 & 31); i1 = (i0+1 == nw) ? 0 : i0+1;
  w_i    = (uint32)((((uint64)ring[i0] << 32) | ring[i1]) >> s0) & 0xffff;
  ```
  Here `ring` is an array of 32-bit words.
- **Code value.** With c = 0x83DCD12D and bytesum(a) the sum of the 4 bytes of a 32-bit a:

  s(w) = bytesum((c·w) mod 2^32) ∈ [0, 1005]

- **Weight value.** v(w) = fp16_round(fp16(1024 + s) · KINV + KOFF), with KINV = 0x1eee and KOFF = 0xc931 as fp16 (≈ 0.006767·s − 3.45). **The weight depends only on s(w).** Over all 65,536 indices, s takes only 913 distinct values. The runtime already maps s → v exactly with an int-to-float conversion and one FMA. So any exact integer reformulation of s(w), or of the sum, keeps the output bit-identical.

## 3. What today's GPU code spends per weight (Pascal sm_61)
| Step | Instructions |
|---|---|
| Window extract | SHF + LOP3 |
| (c·w) mod 2^32 | 3 XMAD (Pascal has no single 32-bit multiply) |
| bytesum + bias | 1 IDP (dp4a against 0x01010101) |
| s → float, multiply-accumulate with x | I2F + FFMA |

That is ~8 useful instructions, ~7.2 achieved per weight on average. The best exact variant found reaches ~5.6.

Other cost facts on sm_61:
- `dp4a`, `prmt`, `lop3` (any 3-input bitwise function), and shifts each cost 1 instruction.
- A shared-memory table read is 1 instruction, but random indices cause bank conflicts (32 banks of 4 bytes).
- Tables up to a few KiB fit in shared memory; 64–128 KiB tables do not.

## 4. Facts already checked on CPU (true; use them or prove them)
- bytesum(a) ≡ a (mod 255), because 256 ≡ 1 (mod 255).
- **Warning:** s(w) is *not* ≡ c·w (mod 255). The wrap mod 2^32 subtracts ⌊c·w / 2^32⌋.
- **Sliding recurrence.** If w' = ((w << K) mod 2^16) | b, with hi = w >> (16 − K) the K bits shifted out and b the K new bits, then

  c·w' ≡ (c·w) << K + c·b − c·2^16·hi (mod 2^32)

  So the product for the next weight comes from the previous one with a shift plus a term that depends only on (hi, b), which is 2K ≤ 10 bits. It follows from expanding w' = 2^K·w − 2^16·hi + b. It was also checked on 100,000 random cases for each K ∈ {1, …, 5}.

## 5. What has failed (do not repeat without a new idea)
- **A 65,536-entry table of v(w) in global memory:** 15× slower.
- **Approximations:** int8 rounding of v, byte-sum shortcuts, 2-byte codebooks. Their error is 0.6×–23× of a quantization step, and **any loss of output quality is rejected.**
- **Batching several tokens per weight decode:** it works (2.5× at 4 tokens) but does not help batch-1 decode.

## 6. The ask
Find an algorithm that computes, for one tile, either
(a) the sequence s(w_0), …, s(w_255), or
(b) directly the integer-weighted sum Σ_k s(w_{n,k}) · x[k] up to the exact s → v map,
with **provably the same result** as the reference and **fewer GPU instructions per weight** than ~5.6 under the cost model of §3.

Directions, none of them known to work:
- exploit the recurrence of §4 to replace the 3-XMAD multiply;
- compute bytesum of the product without forming the full product (carry structure);
- split w into parts with small tables (≤ 4 KiB);
- bucket by s, so that Σ_k v(s_k) x_k = Σ_s v(s) · (Σ_{k : s_k = s} x_k), when that is cheaper;
- use structure of c = 0x83DCD12D itself.

A proof that ~5.6 instructions per weight is a floor under this cost model is also a valid answer.

## 7. Deliverables
1. **Lean 4 (with Mathlib; pin the toolchain and Mathlib version):**
   - faithful definitions of the reference: the ring, the window extraction including the half-K schedule and wraparound, and s(w);
   - the definition of your algorithm;
   - the theorem that they agree for every ring and every valid K.
   - No `sorry` and no new axioms. If you use `decide` or `native_decide` for a finite lemma (for example over all 2^16 indices), say so explicitly, because `native_decide` trusts the compiler. A structural proof is preferred.
2. **A CPU reference in C or Rust:** the reference, your algorithm, and an exhaustive test over all 2^16 indices and every K, plus ≥ 10^6 random tiles.
3. **An operation count** per weight under §3's cost model, with a sketch of the GPU inner loop.
