# Challenge: decode EXL3 mul1 trellis weights faster on a GTX 1060 (batch-1 MoE GEMV)

## The problem in one line
Most decode time goes to one operation: turning compressed EXL3 trellis weights into floats inside a matrix-vector product. Find a mathematically exact way to do it in fewer GPU instructions per weight.

## Where the time goes (measured, Inkwell runtime, 2026-11-24)
- One token takes ~38 ms. The MoE expert GEMV (`inkwell_exl3_gemv_moe_s`) takes ~15.9 ms of it (~42%).
- The dense EXL3 GEMVs (attention + unembed, same weight format) take ~10 ms more.
- The kernel is **instruction-issue bound, not memory-bandwidth bound**. It runs at ~7.2 SASS instructions per weight. The best exact variant found so far is ~5.6–6.75 per weight.

## Hardware
- NVIDIA GTX 1060 6 GB, Pascal sm_61: 10 SMs × 128 FP32 lanes, ~1.7–1.9 GHz (~4.4 TFLOP/s FP32).
- `dp4a`/`dp2a` integer dot products are available.
- FP16 arithmetic runs at 1/64 rate, so it is useless. There are no tensor cores.
- DRAM ~180–192 GB/s; shared memory 48 KiB per block, 96 KiB per SM.

## Model and work per token
- Inkling-2-Mini: 24 layers, hidden size 2304, 128 experts with top-8 routing, expert intermediate size 1152, and gate/up/down projections. Layers 6–17 run 3 times (a loop), giving 48 layer visits per token.
- Decode is batch 1 (one token at a time).
- Expert weights decoded per token: 48 visits × 8 experts × 3 matrices × 2304 × 1152 ≈ **3.06 billion weights**. At ~15.9 ms, that is ~190 G weights/s.

## The weight format (EXL3, "mul1" codebook)
- Each linear layer is W (N_out × K_in). The input is rotated: `x' = H128(suh ⊙ x)`. Here H128 is a 128-point Walsh-Hadamard transform and suh holds ±scale signs.
- The output is rotated back the same way: `y = svh ⊙ H128(W'·x')`. The rotations are cheap separate kernels, so they are not the problem.
- W' is stored in 16×16 tiles. Each tile is a bitstream ("ring") with K bits per weight, where K ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5}. Half-K values alternate bit counts.
- The index of weight i is the **16-bit window** of the ring ending at bit position `s_i` (an inclusive prefix sum of bit counts), and it wraps around the ring. Consecutive weights' windows overlap in 16 − K bits.
- Decode of a 16-bit index `w` into a weight value (exact reference, from `crates/inkwell-cuda/src/kernels.cu` `exl3_mul1`):
  ```c
  w *= 0x83DCD12Du;                                  // mod 2^32
  s  = bytesum(w) + 0x6400;                           // sum of the 4 bytes, as fp16 bits
  h  = (half)s;                                       // reinterpret the low 16 bits as fp16
  v  = fp16_round( fp32(h) * fp32(0x1eee as half) + fp32(0xc931 as half) );
  ```
- The value depends only on the 16-bit index, so it is a fixed 65,536-entry codebook.
- Product: `y'[n] += v(w_{n,k}) · x'[k]`, accumulated in FP32.

## What has been tried (do not repeat without a new idea)
- **Lookup tables:** a 65,536-entry table (128 KiB as fp16) does not fit in shared memory. Reading it from L2/DRAM ran at ~17 G weights/s, 15× slower.
- **Approximations:** dp4a on int8-rounded weight values, byte-sum shortcuts, 2-byte codebooks. They cost 0.6× to 23× of one quantization step of error. **The owner rejects any quality loss.**
- **Conversions:** I2F conversion beats the "magic number" FADD trick by +12% (it offloads to the SFU pipe). Folding the scale per tile loses.
- **Multi-token batching:** decoding each weight once for B tokens gives 1.68× at B = 2 and 2.53× at B = 4. That only helps speculative decoding or prefill, not batch-1 decode.

## The ask
1. Find an **exact** reformulation, bit-identical in the final FP32 output after the same FP16 rounding of each weight. Floating-point summation order may change only if you bound the error. Target: fewer instructions per weight than ~5.6 on sm_61. Directions worth checking:
   - the algebra of `bytesum(w · c mod 2^32)` for a 16-bit w;
   - sharing work between overlapping windows of consecutive weights;
   - splitting the 16-bit index into parts with small tables that fit in shared memory or registers;
   - restructuring the dot product so that the codebook is applied to a sum, not per weight (the value depends only on w, so group equal or related indices);
   - the integer pipes (`dp4a`, `prmt`, `lop3`, `vabsdiff4`) or the SFU.
2. Prove exactness: an argument, plus a test against the reference decode over all 65,536 indices and real tiles.
3. Give an instruction count per weight (from SASS on sm_61, or a careful estimate) and the kernel sketch.

If the answer is that ~5.6 instructions per weight is a hard floor for an exact decode, a proof of that floor is also useful.
