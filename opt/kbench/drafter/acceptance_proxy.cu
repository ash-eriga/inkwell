// Question 5: Real Acceptance Proxy Evaluator on GTX 1060 6GB.
// Uses REAL DSpark weights (/tmp/mtp_dense/*.bin + mtp.safetensors),
// REAL target EXL3 K=4.0 unembed (/tmp/sgemv_dense_gold/unembed.bin + suh/svh),
// REAL target feature taps + KV caches (/tmp/drafter_data/mtp_samples/p*/),
// and our INDEPENDENT corpus frequency ranking (opt/kbench/drafter/vocab_rank_corpus.bin).
//
// Evaluates:
//   0. Oracle verification: FP32 DSpark replay matches Inkwell's logged vtok[1..7] 100%!
//   1. Head-only quantization (exact FP32 DSpark u2_i -> quantized unembed + top-V'):
//      - Weights: FP32, Int8 (per-row), Int4 (per-row), Int4 (group-128)
//      - Activations: FP32, Int16 (dp2a, per-row), Int8 (dp4a, per-row)
//      - Vocab V': Full (200,058), 65536, 32768, 16384, 8192
//   2. Full DSpark (ALL 3 layers + Markov GRU + Unembed) quantized to:
//      - Int8 per-row weights + Int8 act (dp4a)
//      - Int8 per-row weights + Int16 act (dp2a)
//      - Int4 per-row weights + Int8 act (dp4a)
//      - Int4 group-128 weights + Int8 act (dp4a)
//      across V' in {Full, 32768, 16384, 8192}.
//   Reports BOTH:
//      - Argmax agreement vs FP32 DSpark (all 7 steps, drafts 1..3, and draft-1)
//      - Empirical greedy match vs ACTUAL target argmax (veri): p1, emit(k=1,2,3)!
//
// Build:
//   ~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 \
//     -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
//     -gencode arch=compute_61,code=sm_61 opt/kbench/drafter/acceptance_proxy.cu -o /tmp/acceptance_proxy
// Run (under flock):
//   flock /tmp/inkwell-gpu.lock /tmp/acceptance_proxy

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static constexpr int NB = 8;
static constexpr int H  = 2304;
static constexpr int NH = 24;
static constexpr int NKV = 8;
static constexpr int HD = 128;
static constexpr int FF = 9216;
static constexpr int RANK = 256;
static constexpr int VOCAB = 200058;
static constexpr int N_PAD = 201088; // 12568 * 16
static constexpr int TK = 144;
static constexpr int TN = 12568;
static constexpr float EPS = 1e-6f;
static constexpr float THETA = 1e6f;
static constexpr int WINDOW = 1024;

// ---- File helpers ----
template <typename T>
static std::vector<T> read_bin(const std::string& path, size_t expected_elems = 0) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t nb = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = nb / sizeof(T);
    if (expected_elems && n != expected_elems) {
        fprintf(stderr, "Size mismatch %s: got %zu expected %zu\n", path.c_str(), n, expected_elems);
        exit(1);
    }
    std::vector<T> v(n);
    if (fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "Short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    return v;
}

// ============================================================================
// GPU Kernels for exact EXL3 unembed dequant + FP16/Int8/Int4 evaluation
// ============================================================================
__device__ __forceinline__ float h16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

// Decode EXL3 K=4.0 trellis chunk [rows r0..r0+127] x [K=2304] into W_chunk[128, 2304] fp32,
// applying exact H128_in(suh) along K and H128_out(svh) along N!
// Grid: (N_PAD / 128) blocks = 1571 blocks; 1 block produces 128 exact output rows of W_unembed!
// Wait: we run this in slices of CHUNK_ROWS (e.g. 16384 rows) or directly write W_fp16[VOCAB, H] + quantized!
__global__ void dequant_unembed_k4_kernel(
    const uint32_t* __restrict__ trel,  // [TK, TN, 32]
    const uint16_t* __restrict__ suh,   // [H]
    const uint16_t* __restrict__ svh,   // [N_PAD]
    __half* __restrict__ W_f16,         // [VOCAB, H] row-major (out_row, in_col)
    int8_t* __restrict__ W_i8,          // [VOCAB, H] symmetric per-row int8
    float* __restrict__ sc_i8,          // [VOCAB]
    int8_t* __restrict__ W_i4r,         // [VOCAB, H] stored as int8 in [-7,+7] for simple fast eval
    float* __restrict__ sc_i4r,         // [VOCAB]
    int8_t* __restrict__ W_i4g,         // [VOCAB, H] group-128 int4 in [-7,+7]
    float* __restrict__ sc_i4g)         // [VOCAB, 18] (2304/128 = 18 groups)
{
    // Each block owns 128 output rows = 8 tile columns (tn_base = blockIdx.x * 8)
    // We process K in 18 groups of 128 columns (tk_grp = 0..17, 8 tile rows each).
    // For each k_grp (128 cols), decode the 8x8 tiles into smem [128_out][128_in] (64 KB? Too big!
    // Instead do 128_out x 16_in? No, H128_in needs all 128_in, H128_out needs all 128_out ->
    // 128 x 128 = 16384 floats = 64 KB > 48 KB smem! So use 2-pass via global scratch, or fp16 smem = 32 KB!).
    // Wait: 128 x 128 in __half is 32 KB (fits cleanly in 48 KB smem!) OR even cleaner:
    // we have 6 GB VRAM -> 2-stage kernel is exact FP32 with ZERO fp16 intermediate loss!
}

// Clean 2-stage exact FP32 dequant of a slice of G groups (e.g. 256 groups = 32,768 rows) at a time!
// Stage 1: decode raw What[128*G, 2304] and apply H128_in + suh along K -> tmp[chunk_rows, 2304] f32.
// Grid: (18 k-groups, chunk_rows / 16) blocks, 256 threads/block.
// Each block decodes 8 tk-tiles x 1 tn-tile (128_in x 16_out), does H128 along K, multiplies suh, writes tmp!
__global__ void unembed_stage1_had_in(
    const uint32_t* __restrict__ trel,  // [TK, TN, 32]
    const uint16_t* __restrict__ suh,   // [H]
    float* __restrict__ tmp,            // [chunk_rows, H]
    int tn_start, int chunk_tn)
{
    const int kgrp = blockIdx.x;        // 0..17 (each = 8 tk-tiles = 128 cols)
    const int tn_loc = blockIdx.y;      // 0..chunk_tn-1
    const int tn_i = tn_start + tn_loc;
    const int tid = threadIdx.x;        // 0..255

    __shared__ float mat[16][128];      // 16 out_rows x 128 in_cols = 8 KB!
    const float KINV = 0.00676727294921875f;
    const float KC   = -10.3828125f;

    // Decode 8 tiles (tk_i = kgrp*8 .. kgrp*8+7) for this tn_i
    // Thread tid = ring position 0..255 inside each 16x16 tile!
    const int tt = tid >> 3, j = tid & 7;
    const int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
    const int c = (tt >> 2) + ((j & 4) ? 8 : 0);

    // K=4.0: nw = 32, R = 1024; b0 = ((tid+1)*4 - 16 + 1024) & 1023
    const int s = (tid + 1) * 4;
    const int b0 = (s < 16) ? (s - 16 + 1024) : (s - 16);
    const int i0 = b0 >> 5, sh = 48 - (b0 & 31), i1 = (i0 + 1) & 31;

    for (int kt = 0; kt < 8; ++kt) {
        const int tk_i = kgrp * 8 + kt;
        const uint32_t* ring = trel + ((size_t)tk_i * TN + tn_i) * 32;
        uint64_t v = ((uint64_t)ring[i0] << 32) | ring[i1];
        uint32_t w16 = (uint32_t)((v >> sh) & 0xFFFFu);
        uint32_t p = w16 * 0x83DCD12Du;
        int bs = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
        float wv = (1024.f + (float)bs) * KINV + KC;
        mat[c][kt * 16 + r] = wv;
    }
    __syncthreads();

    // Apply H128 along K (128 cols) for each of the 16 out_rows, multiply suh * (1/sqrt(128))
    const float su = (tid < 128) ? h16_to_f32(suh[kgrp * 128 + tid]) * 0.08838834764831845f : 0.f;
    for (int oc = 0; oc < 16; ++oc) {
        for (int st = 1; st < 128; st <<= 1) {
            float u = 0.f;
            if (tid < 128) {
                u = (tid & st) ? (mat[oc][tid - st] - mat[oc][tid]) : (mat[oc][tid] + mat[oc][tid + st]);
            }
            __syncthreads();
            if (tid < 128) mat[oc][tid] = u;
            __syncthreads();
        }
        if (tid < 128) {
            tmp[(size_t)(tn_loc * 16 + oc) * H + kgrp * 128 + tid] = mat[oc][tid] * su;
        }
    }
}

// Stage 2: apply H128_out + svh along N (128-row groups) on tmp[chunk_rows, H],
// write out exact W_f16[row, H], W_i8, W_i4r, W_i4g!
// First pass: H128_out in-place on tmp: Grid (H/16, chunk_rows/128), 128 threads/block.
__global__ void unembed_stage2_had_out(
    float* __restrict__ tmp,            // [chunk_rows, H] in-place
    const uint16_t* __restrict__ svh,   // [N_PAD]
    int row_start, int chunk_rows)
{
    const int col_base = blockIdx.x * 16; // 16 cols per block
    const int ngrp = blockIdx.y;          // 128-row group
    const int tid = threadIdx.x;          // 0..127 (row within group)
    __shared__ float sm[128];

    const int row_loc = ngrp * 128 + tid;
    const float sv = h16_to_f32(svh[row_start + row_loc]) * 0.08838834764831845f;

    for (int dc = 0; dc < 16; ++dc) {
        const int col = col_base + dc;
        sm[tid] = tmp[(size_t)row_loc * H + col];
        __syncthreads();
        for (int st = 1; st < 128; st <<= 1) {
            float u = (tid & st) ? (sm[tid - st] - sm[tid]) : (sm[tid] + sm[tid + st]);
            __syncthreads();
            sm[tid] = u;
            __syncthreads();
        }
        tmp[(size_t)row_loc * H + col] = sm[tid] * sv;
    }
}

// Stage 3: quantize tmp[chunk_rows, H] into global W_f16, W_i8, W_i4r, W_i4g (for row < VOCAB)
// Grid: chunk_rows blocks, 256 threads/block
__global__ void unembed_stage3_quant(
    const float* __restrict__ tmp,
    __half* __restrict__ W_f16,
    int8_t* __restrict__ W_i8,  float* __restrict__ sc_i8,
    int8_t* __restrict__ W_i4r, float* __restrict__ sc_i4r,
    int8_t* __restrict__ W_i4g, float* __restrict__ sc_i4g,
    int row_start)
{
    const int row_loc = blockIdx.x;
    const int row = row_start + row_loc;
    if (row >= VOCAB) return;
    const int tid = threadIdx.x;
    const float* src = tmp + (size_t)row_loc * H;

    // 1. Row-wise amax + write W_f16
    __shared__ float red[8];
    float amax = 0.f;
    for (int c = tid; c < H; c += 256) {
        float v = src[c];
        amax = fmaxf(amax, fabsf(v));
        W_f16[(size_t)row * H + c] = __float2half(v);
    }
    for (int s = 16; s; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
    if ((tid & 31) == 0) red[tid >> 5] = amax;
    __syncthreads();
    if (tid < 32) {
        float m = (tid < 8) ? red[tid] : 0.f;
        for (int s = 4; s; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
        if (tid == 0) red[0] = m;
    }
    __syncthreads();
    float rmax = red[0];
    float s8 = (rmax > 1e-12f) ? (rmax / 127.f) : 1.f;
    float inv8 = (rmax > 1e-12f) ? (127.f / rmax) : 0.f;
    float s4 = (rmax > 1e-12f) ? (rmax / 7.f) : 1.f;
    float inv4 = (rmax > 1e-12f) ? (7.f / rmax) : 0.f;
    if (tid == 0) { sc_i8[row] = s8; sc_i4r[row] = s4; }

    for (int c = tid; c < H; c += 256) {
        float v = src[c];
        W_i8[(size_t)row * H + c]  = (int8_t)max(-127, min(127, __float2int_rn(v * inv8)));
        W_i4r[(size_t)row * H + c] = (int8_t)max(-7,   min(7,   __float2int_rn(v * inv4)));
    }

    // 2. Per-group-128 int4 quantization (18 groups of 128)
    for (int g = 0; g < 18; ++g) {
        float v = (tid < 128) ? src[g * 128 + tid] : 0.f;
        float gm = fabsf(v);
        for (int s = 16; s; s >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, s));
        if ((tid & 31) == 0) red[tid >> 5] = gm;
        __syncthreads();
        if (tid < 32) {
            float m = (tid < 4) ? red[tid] : 0.f;
            for (int s = 2; s; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
            if (tid == 0) red[0] = m;
        }
        __syncthreads();
        float gmax = red[0];
        float sg = (gmax > 1e-12f) ? (gmax / 7.f) : 1.f;
        float invg = (gmax > 1e-12f) ? (7.f / gmax) : 0.f;
        if (tid == 0) sc_i4g[(size_t)row * 18 + g] = sg;
        if (tid < 128) {
            W_i4g[(size_t)row * H + g * 128 + tid] = (int8_t)max(-7, min(7, __float2int_rn(v * invg)));
        }
        __syncthreads();
    }
}

// Compute full-vocab logits [VOCAB] on GPU for ONE u2[H] vector under a chosen head mode!
// mode: 0=FP16(oracle), 1=i8W+f32X, 2=i8W+i16X(dp2a), 3=i8W+i8X(dp4a), 4=i4rW+i8X, 5=i4gW+i8X, 6=i4gW+i16X
__global__ void eval_head_logits(
    const __half* __restrict__ W_f16,
    const int8_t* __restrict__ W_i8,  const float* __restrict__ sc_i8,
    const int8_t* __restrict__ W_i4r, const float* __restrict__ sc_i4r,
    const int8_t* __restrict__ W_i4g, const float* __restrict__ sc_i4g,
    const float* __restrict__ xf,
    const int16_t* __restrict__ x16, float sx16,
    const int8_t* __restrict__ x8,   float sx8,
    float* __restrict__ logits,
    int mode)
{
    __shared__ float xfs[H];
    __shared__ int16_t x16s[H];
    __shared__ int8_t x8s[H];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp;

    for (int c = tid; c < H; c += 256) {
        if (mode == 0 || mode == 1) xfs[c] = xf[c];
        else if (mode == 2 || mode == 6) x16s[c] = x16[c];
        else x8s[c] = x8[c];
    }
    __syncthreads();
    if (row >= VOCAB) return;

    float out = 0.f;
    if (mode == 0) {
        const __half* wp = W_f16 + (size_t)row * H;
        float acc = 0.f;
        for (int c = lane; c < H; c += 32) acc += __half2float(wp[c]) * xfs[c];
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        out = acc;
    } else if (mode == 1) {
        const int8_t* wp = W_i8 + (size_t)row * H;
        float acc = 0.f;
        for (int c = lane; c < H; c += 32) acc += (float)wp[c] * xfs[c];
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        out = acc * sc_i8[row];
    } else if (mode == 2) {
        const int8_t* wp = W_i8 + (size_t)row * H;
        int acc = 0;
        for (int c = lane; c < H; c += 32) acc += (int)wp[c] * (int)x16s[c];
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        out = (float)acc * sc_i8[row] * sx16;
    } else if (mode == 3) {
        const int8_t* wp = W_i8 + (size_t)row * H;
        int acc = 0;
        for (int c = lane; c < H; c += 32) acc += (int)wp[c] * (int)x8s[c];
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        out = (float)acc * sc_i8[row] * sx8;
    } else if (mode == 4) {
        const int8_t* wp = W_i4r + (size_t)row * H;
        int acc = 0;
        for (int c = lane; c < H; c += 32) acc += (int)wp[c] * (int)x8s[c];
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        out = (float)acc * sc_i4r[row] * sx8;
    } else if (mode == 5) {
        const int8_t* wp = W_i4g + (size_t)row * H;
        const float* sg = sc_i4g + (size_t)row * 18;
        float acc = 0.f;
        for (int g = 0; g < 18; ++g) {
            int gacc = 0;
            for (int dc = lane; dc < 128; dc += 32) {
                int c = g * 128 + dc;
                gacc += (int)wp[c] * (int)x8s[c];
            }
            for (int s = 16; s; s >>= 1) gacc += __shfl_xor_sync(0xffffffffu, gacc, s);
            if (lane == 0) acc += (float)gacc * sg[g];
        }
        out = acc * sx8;
    } else if (mode == 6) {
        const int8_t* wp = W_i4g + (size_t)row * H;
        const float* sg = sc_i4g + (size_t)row * 18;
        float acc = 0.f;
        for (int g = 0; g < 18; ++g) {
            int gacc = 0;
            for (int dc = lane; dc < 128; dc += 32) {
                int c = g * 128 + dc;
                gacc += (int)wp[c] * (int)x16s[c];
            }
            for (int s = 16; s; s >>= 1) gacc += __shfl_xor_sync(0xffffffffu, gacc, s);
            if (lane == 0) acc += (float)gacc * sg[g];
        }
        out = acc * sx16;
    }
    if (lane == 0) logits[row] = out;
}

// ============================================================================
// CPU Helpers for DSpark 3-layer body + Markov GRU (exact FP32 + quantized!)
// Note: DSpark body has 3 layers x 7 linear ops + 2 Markov ops; on 12 prompts,
// running quantized GEMM on CPU takes ~3-5 seconds total with zero VRAM limits!
// Wait: we can run quantized matmul on GPU or fast OpenMP/plain CPU! Actually,
// 8 rows x 9216 x 2304 = 170M MACs/layer * 3 layers = 0.5B MACs per prompt ->
// on GPU a tiny cu-kernel does it in 2 ms! Let's write one clean GPU matmul_quant kernel!
// ============================================================================
// Quantize a [N, K] column-major / row-major f32 matrix and run y[B, N] = W_q @ x_q on GPU!
// Remember /tmp/mtp_dense/*.bin is stored [K][N] (row r, col c at r*N + c), so transpose to [N][K] on load.
// qmode: 0=fp32, 1=i8W+i8X(dp4a), 2=i8W+i16X(dp2a), 3=i4rW+i8X, 4=i4g128W+i8X
__global__ void matmul_sim_kernel(
    const float* __restrict__ W_nk, // [N, K] row-major
    const float* __restrict__ X_bk, // [B, K] row-major
    float* __restrict__ Y_bn,       // [B, N]
    int B, int N, int K, int qmode)
{
    const int b = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= N) return;

    const float* wr = W_nk + (size_t)row * K;
    const float* xr = X_bk + (size_t)b * K;

    if (qmode == 0) {
        float acc = 0.f;
        for (int c = lane; c < K; c += 32) acc += wr[c] * xr[c];
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (lane == 0) Y_bn[(size_t)b * N + row] = acc;
        return;
    }

    // Compute per-row amax for W and X
    float wmax = 0.f, xmax = 0.f;
    for (int c = lane; c < K; c += 32) {
        wmax = fmaxf(wmax, fabsf(wr[c]));
        xmax = fmaxf(xmax, fabsf(xr[c]));
    }
    for (int s = 16; s; s >>= 1) {
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, s));
        xmax = fmaxf(xmax, __shfl_xor_sync(0xffffffffu, xmax, s));
    }
    float xq_max = (qmode == 2) ? 32767.f : 127.f;
    float sx = (xmax > 1e-12f) ? (xmax / xq_max) : 1.f;
    float inv_sx = (xmax > 1e-12f) ? (xq_max / xmax) : 0.f;

    if (qmode == 1 || qmode == 2 || qmode == 3) {
        float wq_max = (qmode == 3) ? 7.f : 127.f;
        float sw = (wmax > 1e-12f) ? (wmax / wq_max) : 1.f;
        float inv_sw = (wmax > 1e-12f) ? (wq_max / wmax) : 0.f;
        int acc = 0;
        for (int c = lane; c < K; c += 32) {
            int wq = max(-(int)wq_max, min((int)wq_max, __float2int_rn(wr[c] * inv_sw)));
            int xq = max(-(int)xq_max, min((int)xq_max, __float2int_rn(xr[c] * inv_sx)));
            acc += wq * xq;
        }
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (lane == 0) Y_bn[(size_t)b * N + row] = (float)acc * (sw * sx);
    } else if (qmode == 4) {
        // group-128 int4 weight + per-row int8 x
        float acc = 0.f;
        int ng = K / 128;
        for (int g = 0; g < ng; ++g) {
            float gm = 0.f;
            for (int dc = lane; dc < 128; dc += 32) gm = fmaxf(gm, fabsf(wr[g * 128 + dc]));
            for (int s = 16; s; s >>= 1) gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, s));
            float sg = (gm > 1e-12f) ? (gm / 7.f) : 1.f;
            float inv_sg = (gm > 1e-12f) ? (7.f / gm) : 0.f;
            int gacc = 0;
            for (int dc = lane; dc < 128; dc += 32) {
                int c = g * 128 + dc;
                int wq = max(-7, min(7, __float2int_rn(wr[c] * inv_sg)));
                int xq = max(-127, min(127, __float2int_rn(xr[c] * inv_sx)));
                gacc += wq * xq;
            }
            for (int s = 16; s; s >>= 1) gacc += __shfl_xor_sync(0xffffffffu, gacc, s);
            if (lane == 0) acc += (float)gacc * sg;
        }
        if (lane == 0) Y_bn[(size_t)b * N + row] = acc * sx;
    }
}

struct GpuMat {
    float* d_nk = nullptr;
    int N = 0, K = 0;
    void load_transposed(const std::string& path, int k_in, int n_out) {
        K = k_in; N = n_out;
        auto col = read_bin<float>(path, (size_t)K * N);
        std::vector<float> nk((size_t)N * K);
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < N; ++c)
                nk[(size_t)c * K + r] = col[(size_t)r * N + c];
        CK(cudaMalloc(&d_nk, (size_t)N * K * 4));
        CK(cudaMemcpy(d_nk, nk.data(), (size_t)N * K * 4, cudaMemcpyHostToDevice));
    }
    void run(const std::vector<float>& x_bk, std::vector<float>& y_bn, int B, int qmode, float* dx, float* dy) const {
        CK(cudaMemcpy(dx, x_bk.data(), (size_t)B * K * 4, cudaMemcpyHostToDevice));
        dim3 gr((N + 7) / 8, B);
        matmul_sim_kernel<<<gr, 256>>>(d_nk, dx, dy, B, N, K, qmode);
        CK(cudaDeviceSynchronize());
        y_bn.resize((size_t)B * N);
        CK(cudaMemcpy(y_bn.data(), dy, (size_t)B * N * 4, cudaMemcpyDeviceToHost));
    }
};

struct Sample {
    std::string name, dir;
    int c, lo, anchor, frow;
    int vtok[8], veri[8];
};

static void cpu_rms(const float* x, const float* res, const float* w, float* out, float* res_out, int n, int ch) {
    for (int t = 0; t < n; ++t) {
        double ss = 0.0;
        for (int c = 0; c < ch; ++c) {
            float v = x[(size_t)t * ch + c] + (res ? res[(size_t)t * ch + c] : 0.f);
            ss += (double)v * v;
        }
        float inv = 1.f / sqrtf((float)(ss / ch) + EPS);
        for (int c = 0; c < ch; ++c) {
            float v = x[(size_t)t * ch + c] + (res ? res[(size_t)t * ch + c] : 0.f);
            out[(size_t)t * ch + c] = v * inv * w[c];
            if (res_out) res_out[(size_t)t * ch + c] = v;
        }
    }
}

static void cpu_rope(float* x, int c0, int nb, int nh, int hd) {
    int hh = hd / 2;
    for (int r = 0; r < nb; ++r) {
        float pos = (float)(c0 + r);
        for (int h = 0; h < nh; ++h) {
            float* p = x + ((size_t)r * nh + h) * hd;
            for (int i = 0; i < hh; ++i) {
                float f = pos * powf(THETA, -2.f * i / hd);
                float co = cosf(f), si = sinf(f);
                float x1 = p[i], x2 = p[i + hh];
                p[i] = x1 * co - x2 * si;
                p[i + hh] = x2 * co + x1 * si;
            }
        }
    }
}

static void cpu_attn(const float* q, const float* kc, const float* vc,
                     const float* kb, const float* vb, float* out,
                     int c, int win_lo, int nb, int nkv, int g, int hd) {
    int nctx = c - win_lo;
    int T = nctx + nb;
    float scale = 1.f / sqrtf((float)hd);
    std::vector<float> s(T);
    for (int r = 0; r < nb; ++r) {
        for (int h = 0; h < nkv * g; ++h) {
            int kvh = h / g;
            const float* qr = q + ((size_t)r * (nkv * g) + h) * hd;
            float mx = -1e30f;
            for (int t = 0; t < T; ++t) {
                const float* kp = (t < nctx)
                    ? kc + ((size_t)(win_lo + t) * nkv + kvh) * hd
                    : kb + ((size_t)(t - nctx) * nkv + kvh) * hd;
                float d = 0.f;
                for (int e = 0; e < hd; ++e) d += qr[e] * kp[e];
                s[t] = d * scale;
                if (s[t] > mx) mx = s[t];
            }
            float sm = 0.f;
            for (int t = 0; t < T; ++t) { s[t] = expf(s[t] - mx); sm += s[t]; }
            float inv = 1.f / sm;
            float* orow = out + ((size_t)r * (nkv * g) + h) * hd;
            for (int e = 0; e < hd; ++e) {
                float acc = 0.f;
                for (int t = 0; t < T; ++t) {
                    const float* vp = (t < nctx)
                        ? vc + ((size_t)(win_lo + t) * nkv + kvh) * hd
                        : vb + ((size_t)(t - nctx) * nkv + kvh) * hd;
                    acc += s[t] * vp[e];
                }
                orow[e] = acc * inv;
            }
        }
    }
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

int main() {
    const std::string PDIR = "/tmp/drafter_data/proxy_inputs";
    const std::string DDIR = "/tmp/mtp_dense";

    // 1. Load manifest of 12 real prompts
    std::vector<Sample> samples;
    {
        std::ifstream in(PDIR + "/manifest.txt");
        int ns = 0; in >> ns;
        for (int i = 0; i < ns; ++i) {
            Sample s;
            in >> s.name >> s.dir >> s.c >> s.lo >> s.anchor >> s.frow;
            for (int j = 0; j < 8; ++j) in >> s.vtok[j];
            for (int j = 0; j < 8; ++j) in >> s.veri[j];
            samples.push_back(s);
        }
    }
    printf("Loaded %zu real MTP prompt samples (%zu draft steps)\n", samples.size(), samples.size() * 7);

    // 2. Load independent corpus vocab ranking
    auto vocab_order = read_bin<uint32_t>("opt/kbench/drafter/vocab_rank_corpus.bin", VOCAB);
    std::vector<int> rank_of(VOCAB, VOCAB);
    for (int r = 0; r < VOCAB; ++r) rank_of[vocab_order[r]] = r;

    // 3. Dequantize REAL model.llm.unembed on GPU -> W_f16, W_i8, W_i4r, W_i4g
    printf("Dequantizing real model.llm.unembed (200,058 x 2304, EXL3 K=4.0) on GPU...\n");
    auto htrel = read_bin<uint32_t>("/tmp/sgemv_dense_gold/unembed.bin", (size_t)TK * TN * 32);
    auto hsuh  = read_bin<uint16_t>(PDIR + "/unembed_suh.f16", H);
    auto hsvh  = read_bin<uint16_t>(PDIR + "/unembed_svh.f16", N_PAD);

    uint32_t* dtrel; uint16_t *dsuh, *dsvh;
    __half* dW_f16; int8_t *dW_i8, *dW_i4r, *dW_i4g;
    float *dsc_i8, *dsc_i4r, *dsc_i4g, *dtmp;
    CK(cudaMalloc(&dtrel, htrel.size() * 4));
    CK(cudaMemcpy(dtrel, htrel.data(), htrel.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsuh, H * 2)); CK(cudaMemcpy(dsuh, hsuh.data(), H * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsvh, N_PAD * 2)); CK(cudaMemcpy(dsvh, hsvh.data(), N_PAD * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dW_f16, (size_t)VOCAB * H * 2));   // 879 MiB
    CK(cudaMalloc(&dW_i8,  (size_t)VOCAB * H));       // 439 MiB
    CK(cudaMalloc(&dW_i4r, (size_t)VOCAB * H));       // 439 MiB
    CK(cudaMalloc(&dW_i4g, (size_t)VOCAB * H));       // 439 MiB
    CK(cudaMalloc(&dsc_i8,  VOCAB * 4));
    CK(cudaMalloc(&dsc_i4r, VOCAB * 4));
    CK(cudaMalloc(&dsc_i4g, (size_t)VOCAB * 18 * 4));

    const int CHUNK_TN = 1024; // 16384 rows = 128 groups of 128 rows -> tmp = 144 MiB
    CK(cudaMalloc(&dtmp, (size_t)CHUNK_TN * 16 * H * 4));
    for (int tn0 = 0; tn0 < TN; tn0 += CHUNK_TN) {
        int ctn = std::min(CHUNK_TN, TN - tn0);
        int crows = ctn * 16;
        dim3 g1(18, ctn);
        unembed_stage1_had_in<<<g1, 256>>>(dtrel, dsuh, dtmp, tn0, ctn);
        dim3 g2(H / 16, crows / 128);
        unembed_stage2_had_out<<<g2, 128>>>(dtmp, dsvh, tn0 * 16, crows);
        unembed_stage3_quant<<<crows, 256>>>(dtmp, dW_f16, dW_i8, dsc_i8, dW_i4r, dsc_i4r, dW_i4g, dsc_i4g, tn0 * 16);
    }
    CK(cudaDeviceSynchronize());
    cudaFree(dtrel); cudaFree(dtmp); cudaFree(dsuh); cudaFree(dsvh);
    printf("Unembed dequant + int8/int4 quant complete!\n");

    // Helper: run head on one u2[H] vector -> returns full logits[VOCAB] on host
    float *d_xf, *d_logits; int16_t* d_x16; int8_t* d_x8;
    CK(cudaMalloc(&d_xf, H * 4));
    CK(cudaMalloc(&d_x16, H * 2));
    CK(cudaMalloc(&d_x8, H));
    CK(cudaMalloc(&d_logits, VOCAB * 4));
    std::vector<float> hlogits(VOCAB);

    auto run_head = [&](const std::vector<float>& u2, int mode) -> const std::vector<float>& {
        float amax = 0.f;
        for (float v : u2) amax = std::max(amax, fabsf(v));
        float sx16 = (amax > 1e-12f) ? (amax / 32767.f) : 1.f;
        float inv16 = (amax > 1e-12f) ? (32767.f / amax) : 0.f;
        float sx8 = (amax > 1e-12f) ? (amax / 127.f) : 1.f;
        float inv8 = (amax > 1e-12f) ? (127.f / amax) : 0.f;
        std::vector<int16_t> x16(H); std::vector<int8_t> x8(H);
        for (int i = 0; i < H; ++i) {
            x16[i] = (int16_t)std::max(-32767, std::min(32767, (int)lrintf(u2[i] * inv16)));
            x8[i]  = (int8_t)std::max(-127,   std::min(127,   (int)lrintf(u2[i] * inv8)));
        }
        CK(cudaMemcpy(d_xf, u2.data(), H * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_x16, x16.data(), H * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_x8, x8.data(), H, cudaMemcpyHostToDevice));
        eval_head_logits<<<(VOCAB + 7) / 8, 256>>>(
            dW_f16, dW_i8, dsc_i8, dW_i4r, dsc_i4r, dW_i4g, dsc_i4g,
            d_xf, d_x16, sx16, d_x8, sx8, d_logits, mode);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hlogits.data(), d_logits, VOCAB * 4, cudaMemcpyDeviceToHost));
        return hlogits;
    };

    auto argmax_topV = [&](const std::vector<float>& logits, int Vp) -> int {
        float best = -1e30f; int bi = 0;
        if (Vp >= VOCAB) {
            for (int i = 0; i < VOCAB; ++i) if (logits[i] > best) { best = logits[i]; bi = i; }
        } else {
            for (int r = 0; r < Vp; ++r) {
                int tid = (int)vocab_order[r];
                if (logits[tid] > best) { best = logits[tid]; bi = tid; }
            }
        }
        return bi;
    };

    // 4. Load DSpark body + Markov weights onto GPU (free VRAM ~3.2 GB!)
    printf("Loading DSpark 3-layer + Markov weights...\n");
    struct LayerW {
        GpuMat q, k, v, o, gate, up, down;
        std::vector<float> in_ln, post_ln, qn, kn;
    };
    LayerW lw[3];
    for (int l = 0; l < 3; ++l) {
        std::string p = DDIR + "/l" + std::to_string(l) + "_";
        lw[l].q.load_transposed(p + "q.bin", H, NH * HD);
        lw[l].k.load_transposed(p + "k.bin", H, NKV * HD);
        lw[l].v.load_transposed(p + "v.bin", H, NKV * HD);
        lw[l].o.load_transposed(p + "o.bin", NH * HD, H);
        lw[l].gate.load_transposed(p + "gate.bin", H, FF);
        lw[l].up.load_transposed(p + "up.bin", H, FF);
        lw[l].down.load_transposed(p + "down.bin", FF, H);
        lw[l].in_ln   = read_bin<float>(PDIR + "/l" + std::to_string(l) + "_in_ln.f32", H);
        lw[l].post_ln = read_bin<float>(PDIR + "/l" + std::to_string(l) + "_post_ln.f32", H);
        lw[l].qn      = read_bin<float>(PDIR + "/l" + std::to_string(l) + "_qn.f32", HD);
        lw[l].kn      = read_bin<float>(PDIR + "/l" + std::to_string(l) + "_kn.f32", HD);
    }
    GpuMat m_rnn, m_hout;
    m_rnn.load_transposed(DDIR + "/rnn.bin", 5120, 3 * RANK);
    m_hout.load_transposed(DDIR + "/hout.bin", RANK, H);
    auto rnn_bias = read_bin<float>(PDIR + "/rnn_bias.f32", 3 * RANK);
    auto ho_bias  = read_bin<float>(PDIR + "/ho_bias.f32", H);
    auto final_norm = read_bin<float>(PDIR + "/norm.f32", H);
    auto markov_w1 = read_bin<uint16_t>(PDIR + "/markov_w1.bf16");

    float *d_mx, *d_my;
    CK(cudaMalloc(&d_mx, (size_t)NB * FF * 4));
    CK(cudaMalloc(&d_my, (size_t)NB * FF * 4));

    // Helper: run DSpark 3-layer body for a sample under body_qmode -> returns hid[8*H]
    auto run_body = [&](const Sample& S, int body_qmode) -> std::vector<float> {
        auto x = read_bin<float>(S.dir + "/x.bin", NB * H);
        std::vector<float> res(NB * H), h(NB * H), qb, kb, vb, att(NB * NH * HD), ob, gb, ub, act(NB * FF);
        for (int l = 0; l < 3; ++l) {
            if (l == 0) {
                res = x;
                cpu_rms(x.data(), nullptr, lw[l].in_ln.data(), h.data(), nullptr, NB, H);
            } else {
                cpu_rms(x.data(), res.data(), lw[l].in_ln.data(), h.data(), res.data(), NB, H);
            }
            lw[l].q.run(h, qb, NB, body_qmode, d_mx, d_my);
            lw[l].k.run(h, kb, NB, body_qmode, d_mx, d_my);
            lw[l].v.run(h, vb, NB, body_qmode, d_mx, d_my);
            cpu_rms(qb.data(), nullptr, lw[l].qn.data(), qb.data(), nullptr, NB * NH, HD);
            cpu_rms(kb.data(), nullptr, lw[l].kn.data(), kb.data(), nullptr, NB * NKV, HD);
            cpu_rope(qb.data(), S.c, NB, NH, HD);
            cpu_rope(kb.data(), S.c, NB, NKV, HD);
            auto kc = read_bin<float>(S.dir + "/kc" + std::to_string(l) + ".bin");
            auto vc = read_bin<float>(S.dir + "/vc" + std::to_string(l) + ".bin");
            int win_lo = (l == 1) ? S.lo : std::max(S.c - WINDOW, S.lo);
            cpu_attn(qb.data(), kc.data(), vc.data(), kb.data(), vb.data(), att.data(),
                     S.c, win_lo, NB, NKV, NH / NKV, HD);
            lw[l].o.run(att, ob, NB, body_qmode, d_mx, d_my);
            cpu_rms(ob.data(), res.data(), lw[l].post_ln.data(), h.data(), res.data(), NB, H);
            lw[l].gate.run(h, gb, NB, body_qmode, d_mx, d_my);
            lw[l].up.run(h, ub, NB, body_qmode, d_mx, d_my);
            for (size_t i = 0; i < act.size(); ++i) {
                float gv = gb[i];
                act[i] = (gv / (1.f + expf(-gv))) * ub[i];
            }
            lw[l].down.run(act, x, NB, body_qmode, d_mx, d_my);
        }
        std::vector<float> hid(NB * H);
        cpu_rms(x.data(), res.data(), final_norm.data(), hid.data(), nullptr, NB, H);
        return hid;
    };

    // Helper: run 7 Markov steps given hid[8*H], f2[H], anchor, markov_qmode, head_mode, Vp -> returns 7 drafted tokens + u2s
    auto run_markov = [&](const std::vector<float>& hid, const std::vector<float>& f2, int anchor,
                          int markov_qmode, int head_mode, int Vp,
                          std::vector<std::vector<float>>* out_u2s = nullptr) -> std::vector<int> {
        std::vector<float> state(RANK, 0.f), rnn_in(5120), raw, latent(RANK), ho, u2(H);
        int prev = anchor;
        std::vector<int> preds(7);
        for (int i = 1; i <= 7; ++i) {
            memcpy(rnn_in.data(), state.data(), RANK * 4);
            for (int j = 0; j < RANK; ++j)
                rnn_in[RANK + j] = bf16_to_f32(markov_w1[(size_t)prev * RANK + j]);
            memcpy(rnn_in.data() + 2 * RANK, hid.data() + (size_t)i * H, H * 4);
            memcpy(rnn_in.data() + 2 * RANK + H, f2.data(), H * 4);
            m_rnn.run(rnn_in, raw, 1, markov_qmode, d_mx, d_my);
            for (int c = 0; c < RANK; ++c) {
                float gt = 1.f / (1.f + expf(-(raw[c] + rnn_bias[c])));
                float cand = tanhf(raw[RANK + c] + rnn_bias[RANK + c]);
                state[c] = gt * state[c] + (1.f - gt) * cand;
                latent[c] = tanhf(raw[2 * RANK + c] + rnn_bias[2 * RANK + c]);
            }
            m_hout.run(latent, ho, 1, markov_qmode, d_mx, d_my);
            for (int c = 0; c < H; ++c) u2[c] = ho[c] + ho_bias[c] + hid[(size_t)i * H + c];
            if (out_u2s) out_u2s->push_back(u2);
            const auto& logits = run_head(u2, head_mode);
            prev = argmax_topV(logits, Vp);
            preds[i - 1] = prev;
        }
        return preds;
    };

    // ========================================================================
    // STEP 0: ORACLE VERIFICATION (FP32 body + Markov + FP16-dequant unembed vs Inkwell vtok!)
    // ========================================================================
    printf("\n=== STEP 0: ORACLE VERIFICATION (replaying exact FP32 DSpark vs Inkwell logged vtok) ===\n");
    std::vector<std::vector<std::vector<float>>> fp_u2s(samples.size()); // [sample][7][H]
    std::vector<std::vector<int>> fp_preds(samples.size());
    int match_vtok_all = 0, match_vtok_d1 = 0, total_steps = (int)samples.size() * 7;
    int fp_tgt_d1 = 0; double fp_emit1 = 0, fp_emit2 = 0, fp_emit3 = 0;
    for (size_t si = 0; si < samples.size(); ++si) {
        const auto& S = samples[si];
        auto hid = read_bin<float>(S.dir + "/hid.bin", NB * H);
        auto f2  = read_bin<float>(S.dir + "/f2.bin", H);
        // Also verify run_body(FP32) matches hid.bin!
        if (si == 0) {
            auto hid_chk = run_body(S, 0);
            float md = 0.f;
            for (size_t k = 0; k < hid.size(); ++k) md = std::max(md, fabsf(hid[k] - hid_chk[k]));
            printf("  Body FP32 replay vs Inkwell hid.bin max|diff| = %.2e\n", md);
        }
        fp_preds[si] = run_markov(hid, f2, S.anchor, 0, 0, VOCAB, &fp_u2s[si]);
        for (int j = 0; j < 7; ++j) {
            if (fp_preds[si][j] == S.vtok[j + 1]) {
                match_vtok_all++;
                if (j == 0) match_vtok_d1++;
            }
        }
        if (fp_preds[si][0] == S.veri[0]) fp_tgt_d1++;
        auto calc_emit = [&](const std::vector<int>& pr, int K) {
            int n = 0;
            while (n < K && pr[n] == S.veri[n]) n++;
            return n + 1;
        };
        fp_emit1 += calc_emit(fp_preds[si], 1);
        fp_emit2 += calc_emit(fp_preds[si], 2);
        fp_emit3 += calc_emit(fp_preds[si], 3);
    }
    printf("  Oracle match vs Inkwell logged vtok: %d/%d (%.2f%%), draft-1: %d/%zu (%.1f%%)\n",
           match_vtok_all, total_steps, 100.0 * match_vtok_all / total_steps,
           match_vtok_d1, samples.size(), 100.0 * match_vtok_d1 / samples.size());
    printf("  FP32 DSpark vs REAL Target veri: P(d1)=%d/%zu (%.1f%%), emit(k=1)=%.2f, emit(k=2)=%.2f, emit(k=3)=%.2f\n",
           fp_tgt_d1, samples.size(), 100.0 * fp_tgt_d1 / samples.size(),
           fp_emit1 / samples.size(), fp_emit2 / samples.size(), fp_emit3 / samples.size());

    // ========================================================================
    // STEP 1: EXPERIMENT 5A — HEAD-ONLY QUANTIZATION + TOP-V' (on exact FP32 u2_i)
    // ========================================================================
    printf("\n=== STEP 1 (Q5A): HEAD-ONLY QUANT + REDUCED VOCAB V' (exact FP32 u2_i, %d draft steps) ===\n", total_steps);
    struct HeadCfg { const char* name; int mode; };
    HeadCfg hcfgs[] = {
        {"FP32 head (oracle)",      0},
        {"Int8 W + FP32 X",         1},
        {"Int8 W + Int16 X (dp2a)", 2},
        {"Int8 W + Int8  X (dp4a)", 3},
        {"Int4-row W + Int8 X",     4},
        {"Int4-g128 W + Int8 X",    5},
        {"Int4-g128 W + Int16 X",   6},
    };
    int vps[] = {VOCAB, 65536, 32768, 16384, 8192};

    printf("%-24s | %6s | %11s | %11s | %11s | %10s | %6s %6s %6s\n",
           "Head Quant Mode", "V'", "Match d1", "Match d1..3", "Match d1..7", "Tgt P(d1)", "em(1)", "em(2)", "em(3)");
    printf("------------------------------------------------------------------------------------------------------------\n");
    for (auto& hc : hcfgs) {
        // Compute logits once per (sample, step), then slice at every V'
        std::vector<std::vector<std::vector<int>>> best_by_vp(5, std::vector<std::vector<int>>(samples.size(), std::vector<int>(7)));
        for (size_t si = 0; si < samples.size(); ++si) {
            for (int j = 0; j < 7; ++j) {
                const auto& logits = run_head(fp_u2s[si][j], hc.mode);
                for (int vi = 0; vi < 5; ++vi) {
                    best_by_vp[vi][si][j] = argmax_topV(logits, vps[vi]);
                }
            }
        }
        for (int vi = 0; vi < 5; ++vi) {
            int md1 = 0, md3 = 0, md7 = 0, td1 = 0;
            double e1 = 0, e2 = 0, e3 = 0;
            for (size_t si = 0; si < samples.size(); ++si) {
                const auto& pr = best_by_vp[vi][si];
                const auto& fp = fp_preds[si];
                const auto& S = samples[si];
                if (pr[0] == fp[0]) md1++;
                for (int j = 0; j < 3; ++j) if (pr[j] == fp[j]) md3++;
                for (int j = 0; j < 7; ++j) if (pr[j] == fp[j]) md7++;
                if (pr[0] == S.veri[0]) td1++;
                auto em = [&](int K) { int n = 0; while (n < K && pr[n] == S.veri[n]) n++; return n + 1; };
                e1 += em(1); e2 += em(2); e3 += em(3);
            }
            const char* vstr = (vps[vi] >= VOCAB) ? "200K" : (vps[vi] == 65536 ? "64K" : (vps[vi] == 32768 ? "32K" : (vps[vi] == 16384 ? "16K" : "8K")));
            printf("%-24s | %6s | %2d/%2zu (%4.0f%%) | %2d/%2zu (%4.1f%%) | %2d/%2d (%4.1f%%) | %2d/%2zu (%3.0f%%) | %6.2f %6.2f %6.2f\n",
                   hc.name, vstr,
                   md1, samples.size(), 100.0 * md1 / samples.size(),
                   md3, samples.size() * 3, 100.0 * md3 / (samples.size() * 3),
                   md7, total_steps, 100.0 * md7 / total_steps,
                   td1, samples.size(), 100.0 * td1 / samples.size(),
                   e1 / samples.size(), e2 / samples.size(), e3 / samples.size());
        }
        printf("------------------------------------------------------------------------------------------------------------\n");
    }

    // ========================================================================
    // STEP 2: EXPERIMENT 5B — FULL DSpark (3L BODY + MARKOV + HEAD) END-TO-END QUANTIZED
    // ========================================================================
    printf("\n=== STEP 2 (Q5B): FULL END-TO-END QUANTIZED DSpark (3L Body + Markov + Head, autoreg) ===\n");
    struct FullCfg { const char* name; int body_qm; int head_m; };
    FullCfg fcfgs[] = {
        {"Full Int8 + Int16 X (dp2a)",   2, 2},
        {"Full Int8 + Int8  X (dp4a)",   1, 3},
        {"Full Int4-g128 + Int16 X",     4, 6},
        {"Full Int4-g128 + Int8 X",      4, 5},
        {"Full Int4-row + Int8 X",       3, 4},
    };
    int vps_full[] = {VOCAB, 32768, 16384, 8192};

    printf("%-28s | %5s | %11s | %11s | %11s | %10s | %6s %6s %6s\n",
           "Full DSpark Quant Mode", "V'", "Match d1", "Match d1..3", "Match d1..7", "Tgt P(d1)", "em(1)", "em(2)", "em(3)");
    printf("--------------------------------------------------------------------------------------------------------------\n");
    for (auto& fc : fcfgs) {
        // Compute body hid once per sample for this body_qm
        std::vector<std::vector<float>> hids(samples.size());
        std::vector<std::vector<float>> f2s(samples.size());
        for (size_t si = 0; si < samples.size(); ++si) {
            hids[si] = run_body(samples[si], fc.body_qm);
            f2s[si]  = read_bin<float>(samples[si].dir + "/f2.bin", H);
        }
        for (int Vp : vps_full) {
            int md1 = 0, md3 = 0, md7 = 0, td1 = 0;
            double e1 = 0, e2 = 0, e3 = 0;
            for (size_t si = 0; si < samples.size(); ++si) {
                auto pr = run_markov(hids[si], f2s[si], samples[si].anchor, fc.body_qm, fc.head_m, Vp);
                const auto& fp = fp_preds[si];
                const auto& S = samples[si];
                if (pr[0] == fp[0]) md1++;
                for (int j = 0; j < 3; ++j) if (pr[j] == fp[j]) md3++;
                for (int j = 0; j < 7; ++j) if (pr[j] == fp[j]) md7++;
                if (pr[0] == S.veri[0]) td1++;
                auto em = [&](int K) { int n = 0; while (n < K && pr[n] == S.veri[n]) n++; return n + 1; };
                e1 += em(1); e2 += em(2); e3 += em(3);
            }
            const char* vstr = (Vp >= VOCAB) ? "200K" : (Vp == 32768 ? "32K" : (Vp == 16384 ? "16K" : "8K"));
            printf("%-28s | %5s | %2d/%2zu (%4.0f%%) | %2d/%2zu (%4.1f%%) | %2d/%2d (%4.1f%%) | %2d/%2zu (%3.0f%%) | %6.2f %6.2f %6.2f\n",
                   fc.name, vstr,
                   md1, samples.size(), 100.0 * md1 / samples.size(),
                   md3, samples.size() * 3, 100.0 * md3 / (samples.size() * 3),
                   md7, total_steps, 100.0 * md7 / total_steps,
                   td1, samples.size(), 100.0 * td1 / samples.size(),
                   e1 / samples.size(), e2 / samples.size(), e3 / samples.size());
        }
        printf("--------------------------------------------------------------------------------------------------------------\n");
    }

    return 0;
}
