# Why EXL3 decode on GTX 1060 (sm_61) is instruction-bound — claims to refute

Target: Inkling-2-Mini (MoE, 128 experts, 8/token) in REXL3 2.26bpw on a
GTX 1060 6GB. Measured floor today: ~127 ms/tok serial (~7.9 TPS), ~330-430ms
per 8-token verify round (spec decode, emit ~2.5/round, ~5.2 tok/s).

## Claim 1 — The gemv inner loop is instruction-bound, not bandwidth-bound
- Evidence: full expert residency in VRAM (LFU pool, zero PCIe misses) changed
  verify time by <2%. PCIe measured 11-12GB/s headroom; union ~12-28 experts
  (~1.3GB/round). Kernels measured at ~5GB/s effective on ~150GB/s VRAM.
- Refute with: a kernel variant that exceeds ~15GB/s effective on the SAME
  sm_61 hardware + real weights, or a profile showing byte-bound stalls.

## Claim 2 — ~5 decode instructions per weight is structural at ~3bpw
Per weight in tile_row: funnelshift-isolate a 16-bit window (shift+mask),
multiply by 0x83DCD12D (spreads bit-planes into 4 bytes), __dp4a sums the 4
plane-bytes → integer weight (range measured: ±487 on rexl3-00001), cvt to
float. 4 weights share one 32-bit X but each needs its own window+mul+bytesum.
- Refute with: a scheme decoding an EXL3-trellis weight in <3 instructions on
  average WITHOUT pre-decoded storage (a LUT fails: needs ≥128KB smem for the
  16-bit input space; multi-weight muls fail: window positions overlap and
  carries cross byte lanes).

## Claim 3 — No legal integer dot product exists for the MAC
- Weights are exact integers ±487 → need int16 lanes. sm_61 has IDP4A (int8x4)
  and IDP2A — but measured on-device: __dp2a_lo is int8-pair × int16, NOT
  int16x2 dot (0x00020003·0x00040005 = 15, not 23). No int16 dot exists.
- Weights can't be int8 without re-quantization (EAR loss — forbidden).
- dp4a already IS maxed where legal: inside bs_f32 (the bytesum IS dp4a).
- fp16x2 hfma2: +2 cvt-pack −2 fma, decode unchanged → measured neutral/worse.
- Refute with: an instruction sequence on sm_61 computing int16·int16 or
  int8·int8 on REAL trellis-decoded weights at ≥2x the fp32 FMA rate.

## Claim 4 — 30 TPS serial is unreachable on this HW+format
- Needs ~4x total. MoE gemv ~45ms + dense ~34ms + PCIe ~15ms + had/swarm ~20ms
  + host ~13ms ≈ 127ms. Even zeroing all non-gemv time and getting 2x on every
  gemv gives ~65ms — half the needed speedup and no measured variant got close.
- Refute with: ANY combination of kernels/schedule achieving sustained
  <70 ms/tok with the identical REXL3 checkpoint on a GTX 1060.

## Claim 5 — A "DP4A-optimized EXL3" can't fix it at this size
- dp4a needs int8 both sides. A trellis whose weights decode to int8-codable
  values keeps ~4-4.5bpw density (int4-pair + group scale) → ~9-10GB → doesn't
  fit 6GB → PCIe-bound (~10GB/s) → slower. Any ≤~3.5bpw format forces per-
  weight bit-window extraction on a non-tensor-core GPU.
- Refute with: a concrete ≤3.5bpw weight layout whose decode is ≤2 instructions
  per weight on sm_61, or a ≤6GB repack of Inkling with dp4a-native weights.

## The only wins I believe remain (open)
- ~15% of the MoE inner loop: dp4a([x,x,x,x], plane-bytes) with int8-x per
  16-group (in progress, INKWELL_MOE_D4A — correctness bug under debug).
- Launch/gap removal on the ntok verify path (~85ms/round of ~500 launches).
- CUDA-graphable step_block would cut host overhead ~10-15%.

All measurements in ~/inkwell/opt/ledger.md + HANDOFF.md, kernel at
crates/inkwell-cuda/src/exl3_gemv_v2.cuh (tile_row, bs_f32).
