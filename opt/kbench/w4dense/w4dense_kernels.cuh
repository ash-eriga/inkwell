#pragma once
// Hand-tuned Pascal sm_61 (GTX 1060 6GB) group-wise INT4 / INT5 / INT6 dense GEMV kernels
// with F32 activations (exact F32 accumulation default, plus int16 dp2a comparison).
//
// Packing contract (all matrices row-major [N_out, K_in], K_in % 256 == 0):
//   - W4 weight_lo: [N_out, K_in/2] U8 (viewed as [N_out, K_in/8] U32 little-endian).
//       Each uint32 p holds 8 consecutive weights u[0..7] in [0, 15]:
//         low  nibbles (p & 0x0F0F0F0Fu)        = bytes [u0, u1, u2, u3] -> cols k..k+3
//         high nibbles ((p >> 4) & 0x0F0F0F0Fu) = bytes [u4, u5, u6, u7] -> cols k+4..k+7
//       Symmetric:  u = q + 8  (q in [-8, 7] or [-7, 7]),  w = scale * (u - 8)
//       Asymmetric: u in [0, 15],                          w = scale * u + min
//   - W5 (4+1 split plane):
//       weight_lo  [N_out, K_in/2] U8: low 4 bits of u in [0, 31], identical to W4.
//       weight_hi1 [N_out, K_in/8] U8 (viewed as [N_out, K_in/32] U32): 1 u32 H per 32 weights
//         (4 low words p[0..3]). Byte b in {0..3} of H holds:
//           bit k     (k=0..3) = (u[8k + b]     >> 4) & 1
//           bit 4 + k (k=0..3) = (u[8k + 4 + b] >> 4) & 1
//       Symmetric:  u = q + 16 (q in [-16, 15]), w = scale * (u - 16)
//       Asymmetric: u in [0, 31],                w = scale * u + min
//   - W6 (4+2 split plane):
//       weight_lo  [N_out, K_in/2] U8: low 4 bits of u in [0, 63], identical to W4.
//       weight_hi2 [N_out, K_in/4] U8 (viewed as [N_out, K_in/16] U32): 1 u32 H per 16 weights
//         (2 low words p[0..1]). Byte b in {0..3} of H holds:
//           bits [2k+1 : 2k]     (k=0,1) = (u[8k + b]     >> 4) & 3
//           bits [4+2k+1 : 4+2k] (k=0,1) = (u[8k + 4 + b] >> 4) & 3
//       Symmetric:  u = q + 32 (q in [-32, 31]), w = scale * (u - 32)
//       Asymmetric: u in [0, 63],                w = scale * u + min
//   - Scales:
//       Symmetric:  scale     [N_out, K_in/G]    F16
//       Asymmetric: scale_min [N_out, K_in/G, 2] F16 (interleaved half2 {scale, min})

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace w4d {

// ---- PTX helpers (sm_61) ----

// Opaque AND prevents ptxas from folding (p & 0x0F0F0F0F) >> 8 & 0xFF into BFE + I2F.U32,
// unlocking native dual-issued I2F.F32.U8.B{0..3} on Pascal sm_61.
__device__ __forceinline__ uint32_t and_opaque(uint32_t a, uint32_t mask) {
    uint32_t d;
    asm("and.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); // (a & b) | c
    return d;
}

__device__ __forceinline__ uint32_t lop3_or_and(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xF8;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); // a | (b & c)
    return d;
}

__device__ __forceinline__ float cvt_u8_b0(uint32_t v) { return (float)(v & 0xFFu); }
__device__ __forceinline__ float cvt_u8_b1(uint32_t v) { return (float)((v >> 8) & 0xFFu); }
__device__ __forceinline__ float cvt_u8_b2(uint32_t v) { return (float)((v >> 16) & 0xFFu); }
__device__ __forceinline__ float cvt_u8_b3(uint32_t v) { return (float)((v >> 24) & 0xFFu); }

__device__ __forceinline__ int dp2a_lo_su(int a16x2, uint32_t b8x4, int c) {
    int d;
    asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a16x2), "r"(b8x4), "r"(c));
    return d;
}
__device__ __forceinline__ int dp2a_hi_su(int a16x2, uint32_t b8x4, int c) {
    int d;
    asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a16x2), "r"(b8x4), "r"(c));
    return d;
}

// ---- Prologue: stage x slice [g_lo*G .. g_hi*G) in smem and compute per-group sum_x ----
template <int G, int NWARPS, bool STAGE_X>
__device__ __forceinline__ void stage_x_and_sum(
    const float* __restrict__ x,
    float4* __restrict__ xs4,
    float* __restrict__ sxg,
    int g_lo, int g_hi)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int local_ng = g_hi - g_lo;

    if constexpr (G == 128) {
        for (int lg = warp; lg < local_ng; lg += NWARPS) {
            const int g = g_lo + lg;
            const float4 v = __ldg(x4 + g * 32 + lane);
            if constexpr (STAGE_X) xs4[lg * 32 + lane] = v;
            float s = (v.x + v.y) + (v.z + v.w);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (lane == 0) sxg[lg] = s;
        }
    } else {
        const int hw = tid >> 4;
        const int sub = tid & 15;
        for (int lg = hw; lg < local_ng; lg += NWARPS * 2) {
            const int g = g_lo + lg;
            const float4 v = __ldg(x4 + g * 16 + sub);
            if constexpr (STAGE_X) xs4[lg * 16 + sub] = v;
            float s = (v.x + v.y) + (v.z + v.w);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (sub == 0) sxg[lg] = s;
        }
    }
    __syncthreads();
}

// ---- Split-K self-resetting epilogue (zero memset, single launch, CUDA-graph safe) ----
template <int ROWS_PER_BLK, int NTHREADS>
__device__ __forceinline__ void splitk_epilogue(
    float* __restrict__ y,
    float* __restrict__ part,
    uint32_t* __restrict__ ctr,
    int row_base, int N, int split_k)
{
    __shared__ uint32_t is_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        uint32_t done = atomicAdd(ctr + blockIdx.x, 1u) + 1u;
        is_last = (done == (uint32_t)split_k) ? 1u : 0u;
    }
    __syncthreads();
    if (is_last) {
        for (int r = threadIdx.x; r < ROWS_PER_BLK; r += NTHREADS) {
            const int row = row_base + r;
            if (row < N) {
                float s = 0.f;
                for (int z = 0; z < split_k; ++z) s += part[(size_t)z * N + row];
                y[row] = s;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) ctr[blockIdx.x] = 0u;
    }
}

// ---- Scale/min loader from pre-offset pointer ----
template <bool ASYM>
__device__ __forceinline__ void load_scale_ptr(
    const char* __restrict__ sp, float& sc, float& mn)
{
    if constexpr (ASYM) {
        const __half2 h2 = __ldg(reinterpret_cast<const __half2*>(sp));
        const float2 f2 = __half22float2(h2);
        sc = f2.x;
        mn = f2.y;
    } else {
        sc = __half2float(__ldg(reinterpret_cast<const __half*>(sp)));
        mn = 0.f;
    }
}

template <bool ASYM>
__device__ __forceinline__ void accum_group_f32(
    float dot, float sc, float mn, float neg_z, bool is_g0, float sxg_v, float& acc)
{
    if constexpr (ASYM) {
        acc = fmaf(dot, sc, acc);
        if (is_g0) acc = fmaf(mn, sxg_v, acc);
    } else {
        if (is_g0) dot = fmaf(neg_z, sxg_v, dot);
        acc = fmaf(dot, sc, acc);
    }
}

// ============================================================================
// 1. W4 UNPACK MODES
//    MODE 0: DUAL_I2F   - lo=p&0x0F0F0F0F, hi=p&0xF0F0F0F0, native I2F.F32.U8 (0 shifts!)
//    MODE 1: BFE_I2F    - shift+mask per nibble, I2F.F32.U32
//    MODE 2: LOP3_MAGIC - LOP3 0x4B000000 + FADD(-8388608.f) (0 I2F)
//    MODE 3: LOP3_FOLD  - LOP3 0x3F800000 (1 + u/16, 0 I2F, 0 FADD)
// ============================================================================
template <int MODE>
__device__ __forceinline__ float w4_dot8_f32(uint32_t p, float4 x0, float4 x1) {
    if constexpr (MODE == 0) {
        const uint32_t lo = and_opaque(p, 0x0F0F0F0Fu);
        const uint32_t hi = and_opaque(p, 0xF0F0F0F0u);
        float a_lo = cvt_u8_b0(lo) * x0.x;
        a_lo = fmaf(cvt_u8_b1(lo), x0.y, a_lo);
        a_lo = fmaf(cvt_u8_b2(lo), x0.z, a_lo);
        a_lo = fmaf(cvt_u8_b3(lo), x0.w, a_lo);
        float a_hi = cvt_u8_b0(hi) * x1.x;
        a_hi = fmaf(cvt_u8_b1(hi), x1.y, a_hi);
        a_hi = fmaf(cvt_u8_b2(hi), x1.z, a_hi);
        a_hi = fmaf(cvt_u8_b3(hi), x1.w, a_hi);
        return fmaf(0.0625f, a_hi, a_lo);
    } else if constexpr (MODE == 1) {
        float a = (float)(p & 15u) * x0.x;
        a = fmaf((float)((p >> 8) & 15u), x0.y, a);
        a = fmaf((float)((p >> 16) & 15u), x0.z, a);
        a = fmaf((float)((p >> 24) & 15u), x0.w, a);
        a = fmaf((float)((p >> 4) & 15u), x1.x, a);
        a = fmaf((float)((p >> 12) & 15u), x1.y, a);
        a = fmaf((float)((p >> 20) & 15u), x1.z, a);
        a = fmaf((float)(p >> 28), x1.w, a);
        return a;
    } else if constexpr (MODE == 2) {
        auto cvt = [](uint32_t v) { return __uint_as_float(lop3_and_or(v, 0xFu, 0x4B000000u)) - 8388608.f; };
        float a = cvt(p) * x0.x;
        a = fmaf(cvt(p >> 8), x0.y, a);
        a = fmaf(cvt(p >> 16), x0.z, a);
        a = fmaf(cvt(p >> 24), x0.w, a);
        a = fmaf(cvt(p >> 4), x1.x, a);
        a = fmaf(cvt(p >> 12), x1.y, a);
        a = fmaf(cvt(p >> 20), x1.z, a);
        a = fmaf(cvt(p >> 28), x1.w, a);
        return a;
    } else {
        auto cvt = [](uint32_t v) { return __uint_as_float(lop3_and_or(v, 0x00780000u, 0x3F800000u)); };
        float a = cvt(p << 19) * x0.x;
        a = fmaf(cvt(p << 11), x0.y, a);
        a = fmaf(cvt(p << 3),  x0.z, a);
        a = fmaf(cvt(p >> 5),  x0.w, a);
        a = fmaf(cvt(p << 15), x1.x, a);
        a = fmaf(cvt(p << 7),  x1.y, a);
        a = fmaf(cvt(p >> 1),  x1.z, a);
        a = fmaf(cvt(p >> 9),  x1.w, a);
        return a;
    }
}

// ============================================================================
// 2. W4 GEMV KERNEL (F32 activations, zero XMADs in inner loop!)
// ============================================================================
template <int G, bool ASYM, int R_PER_WARP, int NWARPS, int MODE = 0, bool STAGE_X = true>
__global__ void __launch_bounds__(NWARPS * 32) gemv_w4_f32(
    const uint32_t* __restrict__ W,
    const void* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ part,
    uint32_t* __restrict__ ctr,
    int N, int K)
{
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    static_assert(R_PER_WARP == 1 || R_PER_WARP == 2, "R_PER_WARP must be 1 or 2");
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int ROWS_PER_BLK = NWARPS * R_PER_WARP;
    constexpr int G_PER_STEP = 256 / G;
    constexpr int LANES_PER_G = G / 8;
    constexpr int SC_BYTES = ASYM ? 4 : 2;

    extern __shared__ float smem[];
    float* sxg = smem;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48);

    const int nsteps = K >> 8;
    const int split_k = gridDim.y;
    const int step_lo = (blockIdx.y * nsteps) / split_k;
    const int step_hi = ((blockIdx.y + 1) * nsteps) / split_k;
    const int n_it = step_hi - step_lo;
    const int g_lo = step_lo * G_PER_STEP;
    const int g_hi = step_hi * G_PER_STEP;

    stage_x_and_sum<G, NWARPS, STAGE_X>(x, xs4, sxg, g_lo, g_hi);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g_off = lane / LANES_PER_G;
    const bool is_g0 = ((lane & (LANES_PER_G - 1)) == 0);
    const int row_base = blockIdx.x * ROWS_PER_BLK;
    const int row0 = row_base + warp * R_PER_WARP;
    const int k_u32 = K >> 3;
    const int ng = K / G;

    // Pre-offset pointers outside inner loop -> zero XMADs in loop!
    const uint32_t* wp[R_PER_WARP];
    const char* sp[R_PER_WARP];
    float facc[R_PER_WARP];
#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        const int rr = (row0 + r < N) ? (row0 + r) : (N - 1);
        wp[r] = W + (size_t)rr * k_u32 + (size_t)step_lo * 32 + lane;
        sp[r] = reinterpret_cast<const char*>(scales) + ((size_t)rr * ng + g_lo + g_off) * SC_BYTES;
        facc[r] = 0.f;
    }

    const float4* xp = STAGE_X
        ? (xs4 + lane * 2)
        : (reinterpret_cast<const float4*>(x) + step_lo * 64 + lane * 2);
    const float* sxg_p = sxg + g_off;

    for (int ls = 0; ls < n_it; ++ls) {
        float4 x0, x1;
        if constexpr (STAGE_X) {
            x0 = xp[0]; x1 = xp[1];
        } else {
            x0 = __ldg(xp); x1 = __ldg(xp + 1);
        }
        const float sxg_v = is_g0 ? *sxg_p : 0.f;

#pragma unroll
        for (int r = 0; r < R_PER_WARP; ++r) {
            const uint32_t p = __ldg(wp[r]);
            float sc, mn;
            load_scale_ptr<ASYM>(sp[r], sc, mn);
            float dot = w4_dot8_f32<MODE>(p, x0, x1);
            if constexpr (MODE == 3) {
                const float sc16 = 16.0f * sc;
                if constexpr (ASYM) {
                    facc[r] = fmaf(dot, sc16, facc[r]);
                    if (is_g0) facc[r] = fmaf(mn - sc16, sxg_v, facc[r]);
                } else {
                    if (is_g0) dot = fmaf(-1.5f, sxg_v, dot); // 16*(dot - 1.5*sxg) = sum(u*x) - 8*sxg
                    facc[r] = fmaf(dot, sc16, facc[r]);
                }
            } else {
                accum_group_f32<ASYM>(dot, sc, mn, -8.0f, is_g0, sxg_v, facc[r]);
            }
            wp[r] += 32;
            sp[r] += G_PER_STEP * SC_BYTES;
        }
        xp += 64;
        sxg_p += G_PER_STEP;
    }

#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        float v = facc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && (row0 + r) < N) {
            if (split_k == 1) y[row0 + r] = v;
            else part[(size_t)blockIdx.y * N + (row0 + r)] = v;
        }
    }

    if (split_k > 1) {
        splitk_epilogue<ROWS_PER_BLK, NTHREADS>(y, part, ctr, row_base, N, split_k);
    }
}

// ============================================================================
// 3. W4 64-BIT LOAD VARIANT (16 lanes/row, LDG.64 = 16 weights/lane/iter, 2 rows/warp)
// ============================================================================
template <int G, bool ASYM, int NWARPS, bool STAGE_X = true>
__global__ void __launch_bounds__(NWARPS * 32) gemv_w4_u2_f32(
    const uint2* __restrict__ W2,
    const void* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ part,
    uint32_t* __restrict__ ctr,
    int N, int K)
{
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int ROWS_PER_BLK = NWARPS * 2;
    constexpr int G_PER_STEP = 256 / G;
    constexpr int LANES_PER_G = G / 16;
    constexpr int SC_BYTES = ASYM ? 4 : 2;

    extern __shared__ float smem[];
    float* sxg = smem;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48);

    const int nsteps = K >> 8;
    const int split_k = gridDim.y;
    const int step_lo = (blockIdx.y * nsteps) / split_k;
    const int step_hi = ((blockIdx.y + 1) * nsteps) / split_k;
    const int n_it = step_hi - step_lo;
    const int g_lo = step_lo * G_PER_STEP;
    const int g_hi = step_hi * G_PER_STEP;

    stage_x_and_sum<G, NWARPS, STAGE_X>(x, xs4, sxg, g_lo, g_hi);

    const int tid = threadIdx.x;
    const int hw = tid >> 4;
    const int sub = tid & 15;
    const int g_off = sub / LANES_PER_G;
    const bool is_g0 = ((sub & (LANES_PER_G - 1)) == 0);
    const int row_base = blockIdx.x * ROWS_PER_BLK;
    const int row = row_base + hw;
    const int k_u2 = K >> 4;
    const int ng = K / G;

    float facc = 0.f;
    if (row < N) {
        const uint2* wp = W2 + (size_t)row * k_u2 + (size_t)step_lo * 16 + sub;
        const char* sp = reinterpret_cast<const char*>(scales) + ((size_t)row * ng + g_lo + g_off) * SC_BYTES;
        const float4* xp = STAGE_X
            ? (xs4 + sub * 4)
            : (reinterpret_cast<const float4*>(x) + step_lo * 64 + sub * 4);
        const float* sxg_p = sxg + g_off;

        for (int ls = 0; ls < n_it; ++ls) {
            const uint2 wv = __ldg(wp);
            float4 x0, x1, x2, x3;
            if constexpr (STAGE_X) {
                x0 = xp[0]; x1 = xp[1]; x2 = xp[2]; x3 = xp[3];
            } else {
                x0 = __ldg(xp); x1 = __ldg(xp + 1); x2 = __ldg(xp + 2); x3 = __ldg(xp + 3);
            }

            const uint32_t lo0 = and_opaque(wv.x, 0x0F0F0F0Fu);
            const uint32_t hi0 = and_opaque(wv.x, 0xF0F0F0F0u);
            float a_lo = cvt_u8_b0(lo0) * x0.x;
            a_lo = fmaf(cvt_u8_b1(lo0), x0.y, a_lo);
            a_lo = fmaf(cvt_u8_b2(lo0), x0.z, a_lo);
            a_lo = fmaf(cvt_u8_b3(lo0), x0.w, a_lo);
            float a_hi = cvt_u8_b0(hi0) * x1.x;
            a_hi = fmaf(cvt_u8_b1(hi0), x1.y, a_hi);
            a_hi = fmaf(cvt_u8_b2(hi0), x1.z, a_hi);
            a_hi = fmaf(cvt_u8_b3(hi0), x1.w, a_hi);

            const uint32_t lo1 = and_opaque(wv.y, 0x0F0F0F0Fu);
            const uint32_t hi1 = and_opaque(wv.y, 0xF0F0F0F0u);
            a_lo = fmaf(cvt_u8_b0(lo1), x2.x, a_lo);
            a_lo = fmaf(cvt_u8_b1(lo1), x2.y, a_lo);
            a_lo = fmaf(cvt_u8_b2(lo1), x2.z, a_lo);
            a_lo = fmaf(cvt_u8_b3(lo1), x2.w, a_lo);
            a_hi = fmaf(cvt_u8_b0(hi1), x3.x, a_hi);
            a_hi = fmaf(cvt_u8_b1(hi1), x3.y, a_hi);
            a_hi = fmaf(cvt_u8_b2(hi1), x3.z, a_hi);
            a_hi = fmaf(cvt_u8_b3(hi1), x3.w, a_hi);

            float sc, mn;
            load_scale_ptr<ASYM>(sp, sc, mn);
            const float sxg_v = is_g0 ? *sxg_p : 0.f;
            accum_group_f32<ASYM>(fmaf(0.0625f, a_hi, a_lo), sc, mn, -8.0f, is_g0, sxg_v, facc);

            wp += 16;
            sp += G_PER_STEP * SC_BYTES;
            xp += 64;
            sxg_p += G_PER_STEP;
        }
    }

#pragma unroll
    for (int off = 8; off > 0; off >>= 1) facc += __shfl_xor_sync(0xffffffffu, facc, off);
    if (sub == 0 && row < N) {
        if (split_k == 1) y[row] = facc;
        else part[(size_t)blockIdx.y * N + row] = facc;
    }

    if (split_k > 1) {
        splitk_epilogue<ROWS_PER_BLK, NTHREADS>(y, part, ctr, row_base, N, split_k);
    }
}

// ============================================================================
// 4. W4 INT16 ACTIVATION VARIANT (dp2a.lo/hi.s32.u32, F32 input quantized in prologue)
// ============================================================================
template <int G, bool ASYM, int R_PER_WARP, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) gemv_w4_dp2a_i16(
    const uint32_t* __restrict__ W,
    const void* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ part,
    uint32_t* __restrict__ ctr,
    int N, int K)
{
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    static_assert(R_PER_WARP == 1 || R_PER_WARP == 2, "R_PER_WARP must be 1 or 2");
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int ROWS_PER_BLK = NWARPS * R_PER_WARP;
    constexpr int G_PER_STEP = 256 / G;
    constexpr int LANES_PER_G = G / 8;
    constexpr int SC_BYTES = ASYM ? 4 : 2;

    extern __shared__ int smem_i[];
    float* sx_sc = reinterpret_cast<float*>(smem_i);
    int* sx_sum = smem_i + 48;
    int4* xq4 = reinterpret_cast<int4*>(smem_i + 96);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int nsteps = K >> 8;
    const int split_k = gridDim.y;
    const int step_lo = (blockIdx.y * nsteps) / split_k;
    const int step_hi = ((blockIdx.y + 1) * nsteps) / split_k;
    const int n_it = step_hi - step_lo;
    const int g_lo = step_lo * G_PER_STEP;
    const int g_hi = step_hi * G_PER_STEP;
    const int local_ng = g_hi - g_lo;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    if constexpr (G == 128) {
        for (int lg = warp; lg < local_ng; lg += NWARPS) {
            const int g = g_lo + lg;
            const float4 v = __ldg(x4 + g * 32 + lane);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float sc = (amax > 1e-12f) ? (amax / 32767.0f) : 1.0f;
            const float inv = (amax > 1e-12f) ? (32767.0f / amax) : 0.0f;
            auto q16 = [&](float val) -> int {
                int q = __float2int_rn(val * inv);
                return max(-32767, min(32767, q));
            };
            int q0 = q16(v.x), q1 = q16(v.y), q2 = q16(v.z), q3 = q16(v.w);
            uint32_t p01 = ((uint32_t)(q0 & 0xFFFF)) | (((uint32_t)(q1 & 0xFFFF)) << 16);
            uint32_t p23 = ((uint32_t)(q2 & 0xFFFF)) | (((uint32_t)(q3 & 0xFFFF)) << 16);
            reinterpret_cast<uint2*>(xq4)[lg * 32 + lane] = make_uint2(p01, p23);
            int s = (q0 + q1) + (q2 + q3);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (lane == 0) { sx_sc[lg] = sc; sx_sum[lg] = s; }
        }
    } else {
        const int hw = tid >> 4;
        const int sub = tid & 15;
        for (int lg = hw; lg < local_ng; lg += NWARPS * 2) {
            const int g = g_lo + lg;
            const float4 v = __ldg(x4 + g * 16 + sub);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float sc = (amax > 1e-12f) ? (amax / 32767.0f) : 1.0f;
            const float inv = (amax > 1e-12f) ? (32767.0f / amax) : 0.0f;
            auto q16 = [&](float val) -> int {
                int q = __float2int_rn(val * inv);
                return max(-32767, min(32767, q));
            };
            int q0 = q16(v.x), q1 = q16(v.y), q2 = q16(v.z), q3 = q16(v.w);
            uint32_t p01 = ((uint32_t)(q0 & 0xFFFF)) | (((uint32_t)(q1 & 0xFFFF)) << 16);
            uint32_t p23 = ((uint32_t)(q2 & 0xFFFF)) | (((uint32_t)(q3 & 0xFFFF)) << 16);
            reinterpret_cast<uint2*>(xq4)[lg * 16 + sub] = make_uint2(p01, p23);
            int s = (q0 + q1) + (q2 + q3);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (sub == 0) { sx_sc[lg] = sc; sx_sum[lg] = s; }
        }
    }
    __syncthreads();

    const int g_off = lane / LANES_PER_G;
    const bool is_g0 = ((lane & (LANES_PER_G - 1)) == 0);
    const int row_base = blockIdx.x * ROWS_PER_BLK;
    const int row0 = row_base + warp * R_PER_WARP;
    const int k_u32 = K >> 3;
    const int ng = K / G;

    const uint32_t* wp[R_PER_WARP];
    const char* sp[R_PER_WARP];
    float facc[R_PER_WARP];
#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        const int rr = (row0 + r < N) ? (row0 + r) : (N - 1);
        wp[r] = W + (size_t)rr * k_u32 + (size_t)step_lo * 32 + lane;
        sp[r] = reinterpret_cast<const char*>(scales) + ((size_t)rr * ng + g_lo + g_off) * SC_BYTES;
        facc[r] = 0.f;
    }

    const int4* xqp = xq4 + lane;
    const float* xscp = sx_sc + g_off;
    const int* xsump = sx_sum + g_off;

    for (int ls = 0; ls < n_it; ++ls) {
        const int4 xv = *xqp;
        const float xsc = *xscp;
        const int xsum = is_g0 ? *xsump : 0;

#pragma unroll
        for (int r = 0; r < R_PER_WARP; ++r) {
            const uint32_t p = __ldg(wp[r]);
            const uint32_t lo = p & 0x0F0F0F0Fu;
            const uint32_t hi = (p >> 4) & 0x0F0F0F0Fu;
            int idot = dp2a_lo_su(xv.x, lo, 0);
            idot = dp2a_hi_su(xv.y, lo, idot);
            idot = dp2a_lo_su(xv.z, hi, idot);
            idot = dp2a_hi_su(xv.w, hi, idot);

            float sc, mn;
            load_scale_ptr<ASYM>(sp[r], sc, mn);
            if constexpr (ASYM) {
                facc[r] = fmaf((float)idot, sc * xsc, facc[r]);
                if (is_g0) facc[r] = fmaf(mn * xsc, (float)xsum, facc[r]);
            } else {
                if (is_g0) idot -= 8 * xsum;
                facc[r] = fmaf((float)idot, sc * xsc, facc[r]);
            }
            wp[r] += 32;
            sp[r] += G_PER_STEP * SC_BYTES;
        }
        xqp += 32;
        xscp += G_PER_STEP;
        xsump += G_PER_STEP;
    }

#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        float v = facc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && (row0 + r) < N) {
            if (split_k == 1) y[row0 + r] = v;
            else part[(size_t)blockIdx.y * N + (row0 + r)] = v;
        }
    }

    if (split_k > 1) {
        splitk_epilogue<ROWS_PER_BLK, NTHREADS>(y, part, ctr, row_base, N, split_k);
    }
}

// ============================================================================
// 5. W5 (4+1 SPLIT PLANE) GEMV KERNEL (F32 activations, zero XMADs in loop!)
// ============================================================================
template <int G, bool ASYM, int R_PER_WARP, int NWARPS, bool STAGE_X = true>
__global__ void __launch_bounds__(NWARPS * 32) gemv_w5_f32(
    const uint32_t* __restrict__ Wlo,
    const uint32_t* __restrict__ Whi1,
    const void* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ part,
    uint32_t* __restrict__ ctr,
    int N, int K)
{
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    static_assert(R_PER_WARP == 1 || R_PER_WARP == 2, "R_PER_WARP must be 1 or 2");
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int ROWS_PER_BLK = NWARPS * R_PER_WARP;
    constexpr int G_PER_STEP = 256 / G;
    constexpr int LANES_PER_G = G / 8;
    constexpr int SC_BYTES = ASYM ? 4 : 2;

    extern __shared__ float smem[];
    float* sxg = smem;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48);

    const int nsteps = K >> 8;
    const int split_k = gridDim.y;
    const int step_lo = (blockIdx.y * nsteps) / split_k;
    const int step_hi = ((blockIdx.y + 1) * nsteps) / split_k;
    const int n_it = step_hi - step_lo;
    const int g_lo = step_lo * G_PER_STEP;
    const int g_hi = step_hi * G_PER_STEP;

    stage_x_and_sum<G, NWARPS, STAGE_X>(x, xs4, sxg, g_lo, g_hi);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g_off = lane / LANES_PER_G;
    const bool is_g0 = ((lane & (LANES_PER_G - 1)) == 0);
    const int hi_idx = lane >> 2;
    const int hi_sh = lane & 3;
    const int row_base = blockIdx.x * ROWS_PER_BLK;
    const int row0 = row_base + warp * R_PER_WARP;
    const int k_lo = K >> 3;
    const int k_hi = K >> 5;
    const int ng = K / G;

    const uint32_t* wlo_p[R_PER_WARP];
    const uint32_t* whi_p[R_PER_WARP];
    const char* sp[R_PER_WARP];
    float facc[R_PER_WARP];
#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        const int rr = (row0 + r < N) ? (row0 + r) : (N - 1);
        wlo_p[r] = Wlo + (size_t)rr * k_lo + (size_t)step_lo * 32 + lane;
        whi_p[r] = Whi1 + (size_t)rr * k_hi + (size_t)step_lo * 8 + hi_idx;
        sp[r] = reinterpret_cast<const char*>(scales) + ((size_t)rr * ng + g_lo + g_off) * SC_BYTES;
        facc[r] = 0.f;
    }

    const float4* xp = STAGE_X
        ? (xs4 + lane * 2)
        : (reinterpret_cast<const float4*>(x) + step_lo * 64 + lane * 2);
    const float* sxg_p = sxg + g_off;

    for (int ls = 0; ls < n_it; ++ls) {
        float4 x0, x1;
        if constexpr (STAGE_X) { x0 = xp[0]; x1 = xp[1]; }
        else { x0 = __ldg(xp); x1 = __ldg(xp + 1); }
        const float sxg_v = is_g0 ? *sxg_p : 0.f;

#pragma unroll
        for (int r = 0; r < R_PER_WARP; ++r) {
            const uint32_t p = __ldg(wlo_p[r]);
            const uint32_t Hk = __ldg(whi_p[r]) >> hi_sh;
            const uint32_t lo = lop3_or_and(and_opaque(p, 0x0F0F0F0Fu), Hk << 4, 0x10101010u);
            const uint32_t hi = lop3_or_and(and_opaque(p >> 4, 0x0F0F0F0Fu), Hk, 0x10101010u);

            float dot = cvt_u8_b0(lo) * x0.x;
            dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
            dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
            dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
            dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
            dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
            dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
            dot = fmaf(cvt_u8_b3(hi), x1.w, dot);

            float sc, mn;
            load_scale_ptr<ASYM>(sp[r], sc, mn);
            accum_group_f32<ASYM>(dot, sc, mn, -16.0f, is_g0, sxg_v, facc[r]);

            wlo_p[r] += 32;
            whi_p[r] += 8;
            sp[r] += G_PER_STEP * SC_BYTES;
        }
        xp += 64;
        sxg_p += G_PER_STEP;
    }

#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        float v = facc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && (row0 + r) < N) {
            if (split_k == 1) y[row0 + r] = v;
            else part[(size_t)blockIdx.y * N + (row0 + r)] = v;
        }
    }

    if (split_k > 1) {
        splitk_epilogue<ROWS_PER_BLK, NTHREADS>(y, part, ctr, row_base, N, split_k);
    }
}

// ============================================================================
// 6. W6 (4+2 SPLIT PLANE) GEMV KERNEL (F32 activations, zero XMADs in loop!)
// ============================================================================
template <int G, bool ASYM, int R_PER_WARP, int NWARPS, bool STAGE_X = true>
__global__ void __launch_bounds__(NWARPS * 32) gemv_w6_f32(
    const uint32_t* __restrict__ Wlo,
    const uint32_t* __restrict__ Whi2,
    const void* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ part,
    uint32_t* __restrict__ ctr,
    int N, int K)
{
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    static_assert(R_PER_WARP == 1 || R_PER_WARP == 2, "R_PER_WARP must be 1 or 2");
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int ROWS_PER_BLK = NWARPS * R_PER_WARP;
    constexpr int G_PER_STEP = 256 / G;
    constexpr int LANES_PER_G = G / 8;
    constexpr int SC_BYTES = ASYM ? 4 : 2;

    extern __shared__ float smem[];
    float* sxg = smem;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48);

    const int nsteps = K >> 8;
    const int split_k = gridDim.y;
    const int step_lo = (blockIdx.y * nsteps) / split_k;
    const int step_hi = ((blockIdx.y + 1) * nsteps) / split_k;
    const int n_it = step_hi - step_lo;
    const int g_lo = step_lo * G_PER_STEP;
    const int g_hi = step_hi * G_PER_STEP;

    stage_x_and_sum<G, NWARPS, STAGE_X>(x, xs4, sxg, g_lo, g_hi);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g_off = lane / LANES_PER_G;
    const bool is_g0 = ((lane & (LANES_PER_G - 1)) == 0);
    const int hi_idx = lane >> 1;
    const int hi_sh = (lane & 1) << 1;
    const int row_base = blockIdx.x * ROWS_PER_BLK;
    const int row0 = row_base + warp * R_PER_WARP;
    const int k_lo = K >> 3;
    const int k_hi = K >> 4;
    const int ng = K / G;

    const uint32_t* wlo_p[R_PER_WARP];
    const uint32_t* whi_p[R_PER_WARP];
    const char* sp[R_PER_WARP];
    float facc[R_PER_WARP];
#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        const int rr = (row0 + r < N) ? (row0 + r) : (N - 1);
        wlo_p[r] = Wlo + (size_t)rr * k_lo + (size_t)step_lo * 32 + lane;
        whi_p[r] = Whi2 + (size_t)rr * k_hi + (size_t)step_lo * 16 + hi_idx;
        sp[r] = reinterpret_cast<const char*>(scales) + ((size_t)rr * ng + g_lo + g_off) * SC_BYTES;
        facc[r] = 0.f;
    }

    const float4* xp = STAGE_X
        ? (xs4 + lane * 2)
        : (reinterpret_cast<const float4*>(x) + step_lo * 64 + lane * 2);
    const float* sxg_p = sxg + g_off;

    for (int ls = 0; ls < n_it; ++ls) {
        float4 x0, x1;
        if constexpr (STAGE_X) { x0 = xp[0]; x1 = xp[1]; }
        else { x0 = __ldg(xp); x1 = __ldg(xp + 1); }
        const float sxg_v = is_g0 ? *sxg_p : 0.f;

#pragma unroll
        for (int r = 0; r < R_PER_WARP; ++r) {
            const uint32_t p = __ldg(wlo_p[r]);
            const uint32_t Hk = __ldg(whi_p[r]) >> hi_sh;
            const uint32_t lo = lop3_or_and(and_opaque(p, 0x0F0F0F0Fu), Hk << 4, 0x30303030u);
            const uint32_t hi = lop3_or_and(and_opaque(p >> 4, 0x0F0F0F0Fu), Hk, 0x30303030u);

            float dot = cvt_u8_b0(lo) * x0.x;
            dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
            dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
            dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
            dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
            dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
            dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
            dot = fmaf(cvt_u8_b3(hi), x1.w, dot);

            float sc, mn;
            load_scale_ptr<ASYM>(sp[r], sc, mn);
            accum_group_f32<ASYM>(dot, sc, mn, -32.0f, is_g0, sxg_v, facc[r]);

            wlo_p[r] += 32;
            whi_p[r] += 16;
            sp[r] += G_PER_STEP * SC_BYTES;
        }
        xp += 64;
        sxg_p += G_PER_STEP;
    }

#pragma unroll
    for (int r = 0; r < R_PER_WARP; ++r) {
        float v = facc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0 && (row0 + r) < N) {
            if (split_k == 1) y[row0 + r] = v;
            else part[(size_t)blockIdx.y * N + (row0 + r)] = v;
        }
    }

    if (split_k > 1) {
        splitk_epilogue<ROWS_PER_BLK, NTHREADS>(y, part, ctr, row_base, N, split_k);
    }
}

// ============================================================================
// 7. FUSED UNEMBED + ARGMAX (single persistent launch, 0 logit DRAM writeback!)
// ============================================================================
template <int BITS, int G, bool ASYM, int NWARPS = 8>
__global__ void __launch_bounds__(NWARPS * 32) unembed_argmax_f32(
    const uint32_t* __restrict__ Wlo,
    const uint32_t* __restrict__ Whi,
    const void* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    float* __restrict__ out_val,
    int V, int K)
{
    constexpr int G_PER_STEP = 256 / G;
    constexpr int LANES_PER_G = G / 8;
    constexpr int SC_BYTES = ASYM ? 4 : 2;
    constexpr float NEG_Z = (BITS == 4) ? -8.0f : ((BITS == 5) ? -16.0f : -32.0f);

    extern __shared__ float smem[];
    float* sxg = smem;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48);
    __shared__ float red_v[NWARPS];
    __shared__ uint32_t red_i[NWARPS];

    const int nsteps = K >> 8;
    const int ng = K / G;
    stage_x_and_sum<G, NWARPS, true>(x, xs4, sxg, 0, ng);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g_off = lane / LANES_PER_G;
    const bool is_g0 = ((lane & (LANES_PER_G - 1)) == 0);
    const int k_lo = K >> 3;
    const int k_hi = (BITS == 5) ? (K >> 5) : (K >> 4);
    const int hi_idx = (BITS == 5) ? (lane >> 2) : (lane >> 1);
    const int hi_sh  = (BITS == 5) ? (lane & 3) : ((lane & 1) << 1);
    constexpr int HI_STEP = (BITS == 5) ? 8 : 16;

    float best_v = -1e30f;
    uint32_t best_i = 0;

    const int stride = gridDim.x * NWARPS * 2;
    for (int row0 = (blockIdx.x * NWARPS + warp) * 2; row0 < V; row0 += stride) {
        const int r0 = row0;
        const int r1 = (row0 + 1 < V) ? (row0 + 1) : r0;
        const uint32_t* wlo0 = Wlo + (size_t)r0 * k_lo + lane;
        const uint32_t* wlo1 = Wlo + (size_t)r1 * k_lo + lane;
        const uint32_t* whi0 = Whi ? (Whi + (size_t)r0 * k_hi + hi_idx) : nullptr;
        const uint32_t* whi1 = Whi ? (Whi + (size_t)r1 * k_hi + hi_idx) : nullptr;
        const char* sp0 = reinterpret_cast<const char*>(scales) + ((size_t)r0 * ng + g_off) * SC_BYTES;
        const char* sp1 = reinterpret_cast<const char*>(scales) + ((size_t)r1 * ng + g_off) * SC_BYTES;
        const float4* xp = xs4 + lane * 2;
        const float* sxg_p = sxg + g_off;
        float a0 = 0.f, a1 = 0.f;

        for (int s = 0; s < nsteps; ++s) {
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            const float sxg_v = is_g0 ? *sxg_p : 0.f;

            auto eval_row = [&](const uint32_t* wl, const uint32_t* wh, const char* sp, float& acc) {
                const uint32_t p = __ldg(wl);
                float dot;
                if constexpr (BITS == 4) {
                    dot = w4_dot8_f32<0>(p, x0, x1);
                } else if constexpr (BITS == 5) {
                    const uint32_t Hk = __ldg(wh) >> hi_sh;
                    const uint32_t lo = lop3_or_and(and_opaque(p, 0x0F0F0F0Fu), Hk << 4, 0x10101010u);
                    const uint32_t hi = lop3_or_and(and_opaque(p >> 4, 0x0F0F0F0Fu), Hk, 0x10101010u);
                    dot = cvt_u8_b0(lo) * x0.x;
                    dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
                    dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
                    dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
                    dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
                    dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
                    dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
                    dot = fmaf(cvt_u8_b3(hi), x1.w, dot);
                } else {
                    const uint32_t Hk = __ldg(wh) >> hi_sh;
                    const uint32_t lo = lop3_or_and(and_opaque(p, 0x0F0F0F0Fu), Hk << 4, 0x30303030u);
                    const uint32_t hi = lop3_or_and(and_opaque(p >> 4, 0x0F0F0F0Fu), Hk, 0x30303030u);
                    dot = cvt_u8_b0(lo) * x0.x;
                    dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
                    dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
                    dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
                    dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
                    dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
                    dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
                    dot = fmaf(cvt_u8_b3(hi), x1.w, dot);
                }
                float sc, mn;
                load_scale_ptr<ASYM>(sp, sc, mn);
                accum_group_f32<ASYM>(dot, sc, mn, NEG_Z, is_g0, sxg_v, acc);
            };
            eval_row(wlo0, whi0, sp0, a0);
            eval_row(wlo1, whi1, sp1, a1);
            wlo0 += 32; wlo1 += 32;
            if constexpr (BITS != 4) { whi0 += HI_STEP; whi1 += HI_STEP; }
            sp0 += G_PER_STEP * SC_BYTES; sp1 += G_PER_STEP * SC_BYTES;
            xp += 64; sxg_p += G_PER_STEP;
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a0 += __shfl_xor_sync(0xffffffffu, a0, off);
            a1 += __shfl_xor_sync(0xffffffffu, a1, off);
        }
        if (lane == 0) {
            if (a0 > best_v) { best_v = a0; best_i = (uint32_t)r0; }
            if (row0 + 1 < V && a1 > best_v) { best_v = a1; best_i = (uint32_t)r1; }
        }
    }

    if (lane == 0) { red_v[warp] = best_v; red_i[warp] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = -1e30f; uint32_t bi = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            if (red_v[w] > bv) { bv = red_v[w]; bi = red_i[w]; }
        }
        blk_val[blockIdx.x] = bv;
        blk_id[blockIdx.x] = bi;
        __threadfence();
        uint32_t done = atomicAdd(ticket, 1u) + 1u;
        red_i[0] = (done == (uint32_t)gridDim.x) ? 1u : 0u;
    }
    __syncthreads();
    if (red_i[0]) {
        const int G_blks = gridDim.x;
        float bv = -1e30f; uint32_t bi = 0;
        for (int g = tid; g < G_blks; g += NWARPS * 32) {
            float v = __ldcg(blk_val + g);
            uint32_t id = __ldcg(blk_id + g);
            if (v > bv) { bv = v; bi = id; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            float fbv = -1e30f; uint32_t fbi = 0;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                if (red_v[w] > fbv) { fbv = red_v[w]; fbi = red_i[w]; }
            }
            *out_tok = fbi;
            if (out_val) *out_val = fbv;
            *ticket = 0u;
        }
    }
}

// ============================================================================
// 8. N32-TILED K-OUTER KERNELS (32 rows/warp, LDG.128, 0 bank conflicts, 0 SHFL)
//
//    Layout (load-time 32-row transpose of row-major; 0 extra bytes on all 6 attn shapes,
//    +6 padded rows = +6.9 KB on unembed 200,058 -> 200,064):
//      - Wlo_t:  [N_tiles, K/32, 32] uint4  (each group of 128w = 4 consecutive uint4 slices)
//      - Whi1_t: [N_tiles, K/32, 32] uint32 (W5: 1 u32 per 32w; 4 u32s = 1 uint4 per 128w)
//      - Whi2_t: [N_tiles, K/32, 32] uint2  (W6: 1 uint2 per 32w = 2 u32s per 32w)
//      - S_t:    [N_tiles, K/G,  32] F16 (sym) or half2 {scale, min} (asym)
//
//    Each lane l in [0..31] owns output row (tile * 32 + l).
//    All 32 lanes read the EXACT SAME float4 x from smem (1-cycle hardware broadcast,
//    ZERO bank conflicts), perform ZERO warp shuffles inside the K loop, and load ONE
//    scale per G weights (16x fewer scale LDGs than row-major).
//    For small N (wr_du=384, wk_dv=1024), WPT warps collaborate in-block on the same
//    32-row tile and reduce in smem (ZERO global atomics, ZERO __threadfence!).
// ============================================================================

template <int BITS, int G, bool ASYM, int NWARPS, int WPT, bool STAGE_X = true>
__global__ void __launch_bounds__(NWARPS * 32, (NWARPS == 8 ? 6 : (NWARPS == 16 ? 3 : 8))) gemv_wn_n32_f32(
    const uint4* __restrict__ Wlo_t,
    const void* __restrict__ Whi_t,
    const void* __restrict__ S_t,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K)
{
    static_assert(BITS == 4 || BITS == 5 || BITS == 6, "BITS must be 4, 5, or 6");
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    static_assert(NWARPS % WPT == 0, "NWARPS must be divisible by WPT");
    constexpr int TILES_PER_BLK = NWARPS / WPT;
    constexpr int U4_PER_G = G / 32; // 4 for G=128, 2 for G=64
    constexpr int SC_BYTES = ASYM ? 4 : 2;
    constexpr float NEG_Z = (BITS == 4) ? -8.0f : ((BITS == 5) ? -16.0f : -32.0f);

    extern __shared__ float smem[];
    float* sxg = smem;                                       // [48]
    float* red = smem + 48;                                  // [WPT > 1 ? TILES_PER_BLK * WPT * 32 : 0]
    constexpr int RED_FLOATS = (WPT > 1) ? (TILES_PER_BLK * WPT * 32) : 0;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48 + RED_FLOATS);

    const int ng = K / G;
    stage_x_and_sum<G, NWARPS, STAGE_X>(x, xs4, sxg, 0, ng);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t_in_blk = warp / WPT;
    const int w_part = warp % WPT;
    const int n_tiles = (N + 31) >> 5;

    const int g_lo = (w_part * ng) / WPT;
    const int g_hi = ((w_part + 1) * ng) / WPT;
    const int n_blks_iter = (n_tiles + TILES_PER_BLK - 1) / TILES_PER_BLK;

    for (int blk = blockIdx.x; blk < n_blks_iter; blk += gridDim.x) {
        const int tile = blk * TILES_PER_BLK + t_in_blk;
        const int row = tile * 32 + lane;
        float facc = 0.f;

        if (tile < n_tiles) {
            const size_t base_u4 = ((size_t)tile * ng + g_lo) * U4_PER_G * 32 + lane;
            const uint4* wp = Wlo_t + base_u4;
            const uint32_t* wh5 = (BITS == 5) ? (reinterpret_cast<const uint32_t*>(Whi_t) + base_u4) : nullptr;
            const uint2*    wh6 = (BITS == 6) ? (reinterpret_cast<const uint2*>(Whi_t) + base_u4) : nullptr;
            const char* sp = reinterpret_cast<const char*>(S_t) + (((size_t)tile * ng + g_lo) * 32 + lane) * SC_BYTES;
            const float4* xp = (STAGE_X ? xs4 : reinterpret_cast<const float4*>(x)) + g_lo * (G / 4);

            for (int g = g_lo; g < g_hi; ++g) {
                float dot;
                if constexpr (BITS == 4) {
                    float a_lo = 0.f, a_hi = 0.f;
#pragma unroll
                    for (int c = 0; c < U4_PER_G; ++c) {
                        const uint4 wv = __ldg(wp + c * 32);
                        const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint32_t lo = and_opaque(ps[k], 0x0F0F0F0Fu);
                            const uint32_t hi = and_opaque(ps[k], 0xF0F0F0F0u);
                            const float4 x0 = STAGE_X ? xp[c * 8 + k * 2 + 0] : __ldg(xp + c * 8 + k * 2 + 0);
                            const float4 x1 = STAGE_X ? xp[c * 8 + k * 2 + 1] : __ldg(xp + c * 8 + k * 2 + 1);
                            a_lo = fmaf(cvt_u8_b0(lo), x0.x, a_lo);
                            a_lo = fmaf(cvt_u8_b1(lo), x0.y, a_lo);
                            a_lo = fmaf(cvt_u8_b2(lo), x0.z, a_lo);
                            a_lo = fmaf(cvt_u8_b3(lo), x0.w, a_lo);
                            a_hi = fmaf(cvt_u8_b0(hi), x1.x, a_hi);
                            a_hi = fmaf(cvt_u8_b1(hi), x1.y, a_hi);
                            a_hi = fmaf(cvt_u8_b2(hi), x1.z, a_hi);
                            a_hi = fmaf(cvt_u8_b3(hi), x1.w, a_hi);
                        }
                    }
                    dot = fmaf(0.0625f, a_hi, a_lo);
                } else if constexpr (BITS == 5) {
                    dot = 0.f;
#pragma unroll
                    for (int c = 0; c < U4_PER_G; ++c) {
                        const uint4 wv = __ldg(wp + c * 32);
                        const uint32_t H = __ldg(wh5 + c * 32);
                        const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint32_t Hk = H >> k;
                            const uint32_t lo = lop3_or_and(and_opaque(ps[k], 0x0F0F0F0Fu), Hk << 4, 0x10101010u);
                            const uint32_t hi = lop3_or_and(and_opaque(ps[k] >> 4, 0x0F0F0F0Fu), Hk, 0x10101010u);
                            const float4 x0 = STAGE_X ? xp[c * 8 + k * 2 + 0] : __ldg(xp + c * 8 + k * 2 + 0);
                            const float4 x1 = STAGE_X ? xp[c * 8 + k * 2 + 1] : __ldg(xp + c * 8 + k * 2 + 1);
                            dot = fmaf(cvt_u8_b0(lo), x0.x, dot);
                            dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
                            dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
                            dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
                            dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
                            dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
                            dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
                            dot = fmaf(cvt_u8_b3(hi), x1.w, dot);
                        }
                    }
                } else {
                    dot = 0.f;
#pragma unroll
                    for (int c = 0; c < U4_PER_G; ++c) {
                        const uint4 wv = __ldg(wp + c * 32);
                        const uint2 Hv = __ldg(wh6 + c * 32);
                        const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
                        const uint32_t hs[4] = {Hv.x, Hv.x >> 2, Hv.y, Hv.y >> 2};
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint32_t Hk = hs[k];
                            const uint32_t lo = lop3_or_and(and_opaque(ps[k], 0x0F0F0F0Fu), Hk << 4, 0x30303030u);
                            const uint32_t hi = lop3_or_and(and_opaque(ps[k] >> 4, 0x0F0F0F0Fu), Hk, 0x30303030u);
                            const float4 x0 = STAGE_X ? xp[c * 8 + k * 2 + 0] : __ldg(xp + c * 8 + k * 2 + 0);
                            const float4 x1 = STAGE_X ? xp[c * 8 + k * 2 + 1] : __ldg(xp + c * 8 + k * 2 + 1);
                            dot = fmaf(cvt_u8_b0(lo), x0.x, dot);
                            dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
                            dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
                            dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
                            dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
                            dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
                            dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
                            dot = fmaf(cvt_u8_b3(hi), x1.w, dot);
                        }
                    }
                }

                float sc, mn;
                load_scale_ptr<ASYM>(sp, sc, mn);
                if constexpr (ASYM) {
                    facc = fmaf(dot, sc, facc);
                    facc = fmaf(mn, sxg[g], facc);
                } else {
                    facc = fmaf(fmaf(NEG_Z, sxg[g], dot), sc, facc);
                }
                wp += U4_PER_G * 32;
                if constexpr (BITS == 5) wh5 += U4_PER_G * 32;
                if constexpr (BITS == 6) wh6 += U4_PER_G * 32;
                sp += 32 * SC_BYTES;
                xp += (G / 4);
            }
        }

        if constexpr (WPT == 1) {
            if (row < N) y[row] = facc;
        } else {
            red[(t_in_blk * WPT + w_part) * 32 + lane] = facc;
            __syncthreads();
            if (w_part == 0 && row < N) {
                float s = 0.f;
#pragma unroll
                for (int p = 0; p < WPT; ++p) s += red[(t_in_blk * WPT + p) * 32 + lane];
                y[row] = s;
            }
            __syncthreads();
        }
    }
}

// N32-tiled W4 int16 dp2a kernel (for comparison: ~1.2 ins/w, half smem, 0 bank conflicts)
template <int G, bool ASYM, int NWARPS, int WPT>
__global__ void __launch_bounds__(NWARPS * 32) gemv_w4_n32_dp2a_i16(
    const uint4* __restrict__ Wlo_t,
    const void* __restrict__ S_t,
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int K)
{
    static_assert(G == 128 || G == 64, "G must be 64 or 128");
    static_assert(NWARPS % WPT == 0);
    constexpr int TILES_PER_BLK = NWARPS / WPT;
    constexpr int U4_PER_G = G / 32;
    constexpr int SC_BYTES = ASYM ? 4 : 2;
    constexpr int RED_FLOATS = (WPT > 1) ? (TILES_PER_BLK * WPT * 32) : 0;

    extern __shared__ int smem_i[];
    float* sx_sc = reinterpret_cast<float*>(smem_i);          // [48]
    int* sx_sum = smem_i + 48;                                // [48]
    float* red = reinterpret_cast<float*>(smem_i + 96);       // [RED_FLOATS]
    int4* xq4 = reinterpret_cast<int4*>(smem_i + 96 + RED_FLOATS);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int ng = K / G;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    if constexpr (G == 128) {
        for (int g = warp; g < ng; g += NWARPS) {
            const float4 v = __ldg(x4 + g * 32 + lane);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float sc = (amax > 1e-12f) ? (amax / 32767.0f) : 1.0f;
            const float inv = (amax > 1e-12f) ? (32767.0f / amax) : 0.0f;
            auto q16 = [&](float val) -> int {
                int q = __float2int_rn(val * inv);
                return max(-32767, min(32767, q));
            };
            int q0 = q16(v.x), q1 = q16(v.y), q2 = q16(v.z), q3 = q16(v.w);
            uint32_t p01 = ((uint32_t)(q0 & 0xFFFF)) | (((uint32_t)(q1 & 0xFFFF)) << 16);
            uint32_t p23 = ((uint32_t)(q2 & 0xFFFF)) | (((uint32_t)(q3 & 0xFFFF)) << 16);
            reinterpret_cast<uint2*>(xq4)[g * 32 + lane] = make_uint2(p01, p23);
            int s = (q0 + q1) + (q2 + q3);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (lane == 0) { sx_sc[g] = sc; sx_sum[g] = s; }
        }
    } else {
        const int hw = tid >> 4;
        const int sub = tid & 15;
        for (int g = hw; g < ng; g += NWARPS * 2) {
            const float4 v = __ldg(x4 + g * 16 + sub);
            float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float sc = (amax > 1e-12f) ? (amax / 32767.0f) : 1.0f;
            const float inv = (amax > 1e-12f) ? (32767.0f / amax) : 0.0f;
            auto q16 = [&](float val) -> int {
                int q = __float2int_rn(val * inv);
                return max(-32767, min(32767, q));
            };
            int q0 = q16(v.x), q1 = q16(v.y), q2 = q16(v.z), q3 = q16(v.w);
            uint32_t p01 = ((uint32_t)(q0 & 0xFFFF)) | (((uint32_t)(q1 & 0xFFFF)) << 16);
            uint32_t p23 = ((uint32_t)(q2 & 0xFFFF)) | (((uint32_t)(q3 & 0xFFFF)) << 16);
            reinterpret_cast<uint2*>(xq4)[g * 16 + sub] = make_uint2(p01, p23);
            int s = (q0 + q1) + (q2 + q3);
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (sub == 0) { sx_sc[g] = sc; sx_sum[g] = s; }
        }
    }
    __syncthreads();

    const int t_in_blk = warp / WPT;
    const int w_part = warp % WPT;
    const int n_tiles = (N + 31) >> 5;
    const int g_lo = (w_part * ng) / WPT;
    const int g_hi = ((w_part + 1) * ng) / WPT;
    const int n_blks_iter = (n_tiles + TILES_PER_BLK - 1) / TILES_PER_BLK;

    for (int blk = blockIdx.x; blk < n_blks_iter; blk += gridDim.x) {
        const int tile = blk * TILES_PER_BLK + t_in_blk;
        const int row = tile * 32 + lane;
        float facc = 0.f;
        if (tile < n_tiles) {
            const uint4* wp = Wlo_t + ((size_t)tile * ng + g_lo) * U4_PER_G * 32 + lane;
            const char* sp = reinterpret_cast<const char*>(S_t) + (((size_t)tile * ng + g_lo) * 32 + lane) * SC_BYTES;
            const int4* xqp = xq4 + g_lo * (G / 8);

            for (int g = g_lo; g < g_hi; ++g) {
                int idot = 0;
#pragma unroll
                for (int c = 0; c < U4_PER_G; ++c) {
                    const uint4 wv = __ldg(wp + c * 32);
                    const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int4 xv = xqp[c * 4 + k];
                        const uint32_t lo = ps[k] & 0x0F0F0F0Fu;
                        const uint32_t hi = (ps[k] >> 4) & 0x0F0F0F0Fu;
                        idot = dp2a_lo_su(xv.x, lo, idot);
                        idot = dp2a_hi_su(xv.y, lo, idot);
                        idot = dp2a_lo_su(xv.z, hi, idot);
                        idot = dp2a_hi_su(xv.w, hi, idot);
                    }
                }
                float sc, mn;
                load_scale_ptr<ASYM>(sp, sc, mn);
                const float xsc = sx_sc[g];
                if constexpr (ASYM) {
                    facc = fmaf((float)idot, sc * xsc, facc);
                    facc = fmaf(mn * xsc, (float)sx_sum[g], facc);
                } else {
                    idot -= 8 * sx_sum[g];
                    facc = fmaf((float)idot, sc * xsc, facc);
                }
                wp += U4_PER_G * 32;
                sp += 32 * SC_BYTES;
                xqp += (G / 8);
            }
        }
        if constexpr (WPT == 1) {
            if (row < N) y[row] = facc;
        } else {
            red[(t_in_blk * WPT + w_part) * 32 + lane] = facc;
            __syncthreads();
            if (w_part == 0 && row < N) {
                float s = 0.f;
#pragma unroll
                for (int p = 0; p < WPT; ++p) s += red[(t_in_blk * WPT + p) * 32 + lane];
                y[row] = s;
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// 9. FUSED N32-TILED UNEMBED + ARGMAX (2.68 ins/w, 0 bank conflicts, 0 logit DRAM write!)
// ============================================================================
template <int BITS, int G, bool ASYM, int NWARPS = 8>
__global__ void __launch_bounds__(NWARPS * 32) unembed_argmax_n32_f32(
    const uint4* __restrict__ Wlo_t,
    const void* __restrict__ Whi_t,
    const void* __restrict__ S_t,
    const float* __restrict__ x,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    float* __restrict__ out_val,
    int V, int K)
{
    constexpr int U4_PER_G = G / 32;
    constexpr int SC_BYTES = ASYM ? 4 : 2;
    constexpr float NEG_Z = (BITS == 4) ? -8.0f : ((BITS == 5) ? -16.0f : -32.0f);

    extern __shared__ float smem[];
    float* sxg = smem;
    float4* xs4 = reinterpret_cast<float4*>(smem + 48);
    __shared__ float red_v[NWARPS];
    __shared__ uint32_t red_i[NWARPS];

    const int ng = K / G;
    stage_x_and_sum<G, NWARPS, true>(x, xs4, sxg, 0, ng);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_tiles = (V + 31) >> 5;
    const int stride = gridDim.x * NWARPS;

    float best_v = -1e30f;
    uint32_t best_i = 0;

    for (int tile = blockIdx.x * NWARPS + warp; tile < n_tiles; tile += stride) {
        const int row = tile * 32 + lane;
        const size_t base_u4 = (size_t)tile * ng * U4_PER_G * 32 + lane;
        const uint4* wp = Wlo_t + base_u4;
        const uint32_t* wh5 = (BITS == 5) ? (reinterpret_cast<const uint32_t*>(Whi_t) + base_u4) : nullptr;
        const uint2*    wh6 = (BITS == 6) ? (reinterpret_cast<const uint2*>(Whi_t) + base_u4) : nullptr;
        const char* sp = reinterpret_cast<const char*>(S_t) + ((size_t)tile * ng * 32 + lane) * SC_BYTES;
        const float4* xp = xs4;

        float facc = 0.f;
        for (int g = 0; g < ng; ++g) {
            float dot;
            if constexpr (BITS == 4) {
                float a_lo = 0.f, a_hi = 0.f;
#pragma unroll
                for (int c = 0; c < U4_PER_G; ++c) {
                    const uint4 wv = __ldg(wp + c * 32);
                    const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t lo = and_opaque(ps[k], 0x0F0F0F0Fu);
                        const uint32_t hi = and_opaque(ps[k], 0xF0F0F0F0u);
                        const float4 x0 = xp[c * 8 + k * 2 + 0];
                        const float4 x1 = xp[c * 8 + k * 2 + 1];
                        a_lo = fmaf(cvt_u8_b0(lo), x0.x, a_lo);
                        a_lo = fmaf(cvt_u8_b1(lo), x0.y, a_lo);
                        a_lo = fmaf(cvt_u8_b2(lo), x0.z, a_lo);
                        a_lo = fmaf(cvt_u8_b3(lo), x0.w, a_lo);
                        a_hi = fmaf(cvt_u8_b0(hi), x1.x, a_hi);
                        a_hi = fmaf(cvt_u8_b1(hi), x1.y, a_hi);
                        a_hi = fmaf(cvt_u8_b2(hi), x1.z, a_hi);
                        a_hi = fmaf(cvt_u8_b3(hi), x1.w, a_hi);
                    }
                }
                dot = fmaf(0.0625f, a_hi, a_lo);
            } else if constexpr (BITS == 5) {
                dot = 0.f;
#pragma unroll
                for (int c = 0; c < U4_PER_G; ++c) {
                    const uint4 wv = __ldg(wp + c * 32);
                    const uint32_t H = __ldg(wh5 + c * 32);
                    const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t Hk = H >> k;
                        const uint32_t lo = lop3_or_and(and_opaque(ps[k], 0x0F0F0F0Fu), Hk << 4, 0x10101010u);
                        const uint32_t hi = lop3_or_and(and_opaque(ps[k] >> 4, 0x0F0F0F0Fu), Hk, 0x10101010u);
                        const float4 x0 = xp[c * 8 + k * 2 + 0];
                        const float4 x1 = xp[c * 8 + k * 2 + 1];
                        dot = fmaf(cvt_u8_b0(lo), x0.x, dot);
                        dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
                        dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
                        dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
                        dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
                        dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
                        dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
                        dot = fmaf(cvt_u8_b3(hi), x1.w, dot);
                    }
                }
            } else {
                dot = 0.f;
#pragma unroll
                for (int c = 0; c < U4_PER_G; ++c) {
                    const uint4 wv = __ldg(wp + c * 32);
                    const uint2 Hv = __ldg(wh6 + c * 32);
                    const uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
                    const uint32_t hs[4] = {Hv.x, Hv.x >> 2, Hv.y, Hv.y >> 2};
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t Hk = hs[k];
                        const uint32_t lo = lop3_or_and(and_opaque(ps[k], 0x0F0F0F0Fu), Hk << 4, 0x30303030u);
                        const uint32_t hi = lop3_or_and(and_opaque(ps[k] >> 4, 0x0F0F0F0Fu), Hk, 0x30303030u);
                        const float4 x0 = xp[c * 8 + k * 2 + 0];
                        const float4 x1 = xp[c * 8 + k * 2 + 1];
                        dot = fmaf(cvt_u8_b0(lo), x0.x, dot);
                        dot = fmaf(cvt_u8_b1(lo), x0.y, dot);
                        dot = fmaf(cvt_u8_b2(lo), x0.z, dot);
                        dot = fmaf(cvt_u8_b3(lo), x0.w, dot);
                        dot = fmaf(cvt_u8_b0(hi), x1.x, dot);
                        dot = fmaf(cvt_u8_b1(hi), x1.y, dot);
                        dot = fmaf(cvt_u8_b2(hi), x1.z, dot);
                        dot = fmaf(cvt_u8_b3(hi), x1.w, dot);
                    }
                }
            }

            float sc, mn;
            load_scale_ptr<ASYM>(sp, sc, mn);
            if constexpr (ASYM) {
                facc = fmaf(dot, sc, facc);
                facc = fmaf(mn, sxg[g], facc);
            } else {
                facc = fmaf(fmaf(NEG_Z, sxg[g], dot), sc, facc);
            }
            wp += U4_PER_G * 32;
            if constexpr (BITS == 5) wh5 += U4_PER_G * 32;
            if constexpr (BITS == 6) wh6 += U4_PER_G * 32;
            sp += 32 * SC_BYTES;
            xp += (G / 4);
        }

        if (row < V && facc > best_v) { best_v = facc; best_i = (uint32_t)row; }
    }

    // Warp reduction
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
        uint32_t oi = __shfl_xor_sync(0xffffffffu, best_i, off);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { red_v[warp] = best_v; red_i[warp] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = -1e30f; uint32_t bi = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            if (red_v[w] > bv) { bv = red_v[w]; bi = red_i[w]; }
        }
        blk_val[blockIdx.x] = bv;
        blk_id[blockIdx.x] = bi;
        __threadfence();
        uint32_t done = atomicAdd(ticket, 1u) + 1u;
        red_i[0] = (done == (uint32_t)gridDim.x) ? 1u : 0u;
    }
    __syncthreads();
    if (red_i[0]) {
        const int G_blks = gridDim.x;
        float bv = -1e30f; uint32_t bi = 0;
        for (int g = tid; g < G_blks; g += NWARPS * 32) {
            float v = __ldcg(blk_val + g);
            uint32_t id = __ldcg(blk_id + g);
            if (v > bv) { bv = v; bi = id; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
        __syncthreads();
        if (tid == 0) {
            float fbv = -1e30f; uint32_t fbi = 0;
#pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                if (red_v[w] > fbv) { fbv = red_v[w]; fbi = red_i[w]; }
            }
            *out_tok = fbi;
            if (out_val) *out_val = fbv;
            *ticket = 0u;
        }
    }
}

} // namespace w4d
