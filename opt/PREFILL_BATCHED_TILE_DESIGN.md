# Batched prefill kernel design, sm_61

Date: 2026-11-25. Scope: `crates/inkwell-cuda/src/prefill.cu` on GTX 1060 6 GB.

## Hardware rooflines used

The GTX 1060 here has 10 SMs, 128 FP32 lanes/SM, and a boost clock near 1.7 GHz, so the FP32 FMA ceiling is about 4.3-4.4 TFLOP/s. There are no tensor cores and FP16 math is not useful. Shared memory can supply at most 128 B/cycle/SM. The practical issue ceiling is lower than the FMA ceiling because every shared-memory load, integer decode instruction, branch, and barrier consumes issue slots.

The prefill GEMMs are compute kernels, not DRAM kernels. At the real shapes the compressed EXL3 payload is tiny compared with the FMA work:

| shape | K | N | EXL3 setting | compressed bytes per 16x16 tile |
|---|---:|---:|---|---:|
| MoE gate/up | 2304 | 1152 | ka=2, halfk=1 | 80 |
| MoE down | 1152 | 2304 | ka=2, halfk=1 | 80 |
| dense wq | 2304 | 3072 | ka=3, halfk=0 | 96 |
| dense wo | 3072 | 2304 | ka=3, halfk=1 | 112 |

A 64x64x16 tile performs 65,536 FMAs (131,072 FLOPs). Even at ka=3.5 it reads only 448 compressed bytes for four 16-column weight tiles. Decode costs roughly five integer instructions per weight, but a decoded weight is reused across the rows of the tile: 64 rows amortize it to about 0.08 integer instructions per FMA lane-use at BM=64, and 0.16 at BM=32. Weight decode is therefore not the roofline. Shared-memory operand traffic is.

For a thread tile of RMxRN, each inner-loop step loads RM activation values and RN weight values and performs RM*RN FMAs. Ignoring broadcasts, the shared bytes per lane-FMA are approximately `4*(RM+RN)/(RM*RN)`:

| thread tile | shared B/FMA | shared-memory roof at 128 B/cy |
|---|---:|---:|
| 2x4 | 3.00 | 42.7 FMA/cy/SM, about 1.45 TFLOP/s device-wide |
| 4x4 | 2.00 | 64 FMA/cy/SM, about 2.2 TFLOP/s |
| 4x8 or 8x4 | 1.50 | 85 FMA/cy/SM, about 2.9 TFLOP/s |
| 8x8 | 1.00 | 128 FMA/cy/SM, the FP32 pipe limit |

These are upper bounds before issue overhead, decode, barriers, edge predicates, and occupancy losses. The 1.5 TFLOP/s target requires at least a conflict-free 4x4 schedule, and preferably 4x8 on dense BM=64 tiles. Bank-conflict freedom matters as much as the nominal byte count: a stride-four 32-bit shared load uses only eight banks per quarter-warp group and can cost four times the minimum. Column ownership therefore has to be strided so adjacent lanes touch adjacent banks; making each lane own a contiguous four- or eight-column strip is the wrong layout on Pascal.

## EXL3 GEMM plan

1. Keep the public launcher `ink_launch_exl3_gemm_smem` unchanged.
2. Keep the per-output reduction order exactly as the reference for the first implementation: k-tile ascending, then r=0..15 ascending, one FP32 accumulator per output. This preserves bit equality even though M/N tiling changes.
3. Decode every 16x16 compressed weight tile once per CTA into shared memory. Do not decode trellis bits in the FMA loop.
4. Use strided column ownership and padded shared rows. The dense-tile design is BM=64, BN=64, 128 threads, RM=4, RN=8. Each thread owns four contiguous rows and eight strided columns (`tx + 8*j`). Narrow-N shapes use BM=32, BN=64, 128 threads, RM=4, RN=4 to restore output parallelism. This gives 32 FMAs per k-step for 12 shared scalar loads, avoids stride-four bank conflicts, and needs no requantization or numeric reordering.
5. Load the compressed ring words and the activation tile before the barrier, decode them into the weight tile, then run the unrolled outer product. Where register pressure allows, prefetch the next compressed/X tile during the current FMA loop; this is double buffering of inputs, not a megakernel or a change in reduction order.
6. For smaller dense M, keep the existing BM specializations as fallbacks. Correctness matters at M=1/16/64; performance matters at M=64.

## Grouped MoE GEMM plan

The realistic grouped shape is C=512, top-8 of 128 experts, approximately 32 rows per expert. A BM=64 tile therefore does half of its FMAs on zero padding. That waste, not weight decode, is the main grouped-GEMM roofline loss.

The grouped kernel will branch on `rows`:

* `rows <= 32`: use a BM=32, BN=128 tile with 256 threads and strided 4x4 register blocking (`tx + 32*j`). It performs no padded-row FMAs. Because the existing grid supplies one CTA per 64 columns, even `blockIdx.x` values cover 128 columns and odd values exit. This preserves the kernel entry point and launch contract.
* `rows > 32`: use the dense-style BM=64, BN=128 tile with strided 4x8 register blocking.

The reduction order remains k-tile then r ascending. Split-K is deliberately not part of the first implementation because it would change FP32 summation order and require a tolerance rather than bit equality.

## Flash attention plan

The flash kernel's global-memory intensity is high. For BQ=32, BK=32, three query heads per KV head, one K/V tile is 16 KiB and the corresponding QK+PV work is about 1.57 MFLOP, roughly 98 FLOP/B. DRAM is not the limit. The limit is shared/register traffic, conversion overhead, and occupancy: the current CTA uses 168 registers and about 32 KiB shared memory, so only three CTAs/SM are resident and there are already small spills.

The first attention changes are therefore structural but numerically conservative:

1. Keep online softmax and the same FP32 accumulation type.
2. Reuse each K tile for all three GQA query heads in the QK phase instead of rereading it once per head.
3. Keep V resident in shared memory while applying the three head-specific probability tiles, instead of reloading it from global memory per head.
4. Reduce register pressure before considering wider tiles. In particular, avoid keeping the 16 relative-position coefficients live across the QK inner loop and avoid deep unroll temporaries that do not pay for themselves on Pascal.
5. Preserve the deterministic row-wise softmax order. The existing harness already allows a small FP32 tolerance for flash because exp/order differences are inherent in the oracle comparison; report max error explicitly.

## Gates and iteration order

For each experiment: build cleanly, run the smallest relevant kbench suite through `gpu-run -p normal`, record bit differences or FP32-reorder error, and only then run the full harness. A change is kept only if correctness passes and the realistic-shape TFLOP/s improves beyond run-to-run noise. The decode path is not part of the optimization surface; final validation must leave canonical md5 `5cbe8e06` unchanged.

## Measured design corrections

The first vectorized-float4 weight-load variant was slower on sm_61 than conflict-free scalar strided loads, so it was rejected. A narrow-everything dense schedule also lost to the adaptive layout; narrow tiling is used only where N is small. Splitting the three flash heads across CTAs, converting K/V tiles to FP32 in shared memory, and vectorizing probability loads were all measured and rejected as neutral or worse. Those negative results are why the final flash kernel remains the baseline schedule.
