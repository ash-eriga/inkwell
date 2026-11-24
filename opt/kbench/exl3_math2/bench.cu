// Round 2 EXL3 decode-math GPU microbenchmark vs shipped sgemv_run (inkwell_exl3_gemv_moe_s).
//
// Enforces:
//   - <= 40 registers/thread (0 spills), >= 6 blocks/SM (256 threads/block)
//   - Exact same 8-tiles/block grid parallelism as shipped sgemv_run (144 blocks GU, 144 blocks down)
//   - Verified against CPU FP64 ref_gemv on the exact same synthetic data
//   - Interleaved (shipped vs candidate) min-of-31 (and median-of-31) timing and G weights/s
//   - Drop-in float* xh interface (in-prologue int16/int8 quant for C4/C6 & per-tile sxt for C3b)
//
// Build:
//   ~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 --use_fast_math -Xptxas -v \
//     -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
//     -gencode arch=compute_61,code=sm_61 -Icrates/inkwell-cuda/src \
//     opt/kbench/exl3_math2/bench.cu -o /tmp/exl3_math2_bench
// Run (ALWAYS under flock):
//   flock /tmp/inkwell-gpu.lock /tmp/exl3_math2_bench

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <type_traits>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../../crates/inkwell-cuda/src/kernels.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace r2 {

// ---- PTX building blocks (sm_61) ----
__device__ __forceinline__ uint32_t dp4a_uu(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d; asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ int dp4a_su(uint32_t a, uint32_t b, int c) {
    int d; asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ int dp2a_lo_su(uint32_t a, uint32_t b, int c) {
    int d; asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ int xmad_s16_h0(uint32_t xpair, int bs_c, int acc) {
    int d;
    asm("{\n\t.reg .s16 x0, x1, b0;\n\t.reg .s32 p;\n\tmov.b32 {x0, x1}, %1;\n\tcvt.s16.s32 b0, %2;\n\tmul.wide.s16 p, x0, b0;\n\tadd.s32 %0, p, %3;\n\t}"
        : "=r"(d) : "r"(xpair), "r"(bs_c), "r"(acc));
    return d;
}
__device__ __forceinline__ int xmad_s16_h1(uint32_t xpair, int bs_c, int acc) {
    int d;
    asm("{\n\t.reg .s16 x0, x1, b0;\n\t.reg .s32 p;\n\tmov.b32 {x0, x1}, %1;\n\tcvt.s16.s32 b0, %2;\n\tmul.wide.s16 p, x1, b0;\n\tadd.s32 %0, p, %3;\n\t}"
        : "=r"(d) : "r"(xpair), "r"(bs_c), "r"(acc));
    return d;
}
__device__ __forceinline__ uint32_t mul16_h0(uint32_t U, uint32_t CM) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, cl;\n\t}"
        : "=r"(t) : "r"(U), "r"(CM));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, lo, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(U), "r"(CM), "r"(t));
    return p;
}
__device__ __forceinline__ uint32_t mul16_h1(uint32_t U, uint32_t CM) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, cl;\n\t}"
        : "=r"(t) : "r"(U), "r"(CM));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, hi, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(U), "r"(CM), "r"(t));
    return p;
}
__device__ __forceinline__ int mac_lea_dp2a_h0(uint32_t U, uint32_t x16s, int acc) {
    uint32_t rt, rhi;
    asm("{\n\t.reg .u16 lo, hi;\n\t.reg .u32 thi, prod;\n\tmov.b32 {lo, hi}, %2;\n\t"
        "mul.wide.u16 %0, lo, 0xD12D;\n\tshf.r.clamp.b32 thi, %0, 0, 16;\n\t"
        "mul.wide.u16 prod, lo, 0x83DC;\n\tadd.u32 %1, prod, thi;\n\t}" : "=r"(rt), "=r"(rhi) : "r"(U));
    acc = dp2a_lo_su(x16s, rt, acc);
    acc = dp2a_lo_su(x16s, rhi, acc);
    return acc;
}
__device__ __forceinline__ int mac_lea_dp2a_h1(uint32_t U, uint32_t x16s, int acc) {
    uint32_t rt, rhi;
    asm("{\n\t.reg .u16 lo, hi;\n\t.reg .u32 thi, prod;\n\tmov.b32 {lo, hi}, %2;\n\t"
        "mul.wide.u16 %0, hi, 0xD12D;\n\tshf.r.clamp.b32 thi, %0, 0, 16;\n\t"
        "mul.wide.u16 prod, hi, 0x83DC;\n\tadd.u32 %1, prod, thi;\n\t}" : "=r"(rt), "=r"(rhi) : "r"(U));
    acc = dp2a_lo_su(x16s, rt, acc);
    acc = dp2a_lo_su(x16s, rhi, acc);
    return acc;
}

static constexpr int MAXK = 2304;
static constexpr float KINV_F = 0.00676727294921875f;  // fp16 0x1eee
static constexpr float KC_F   = -10.3828125f;          // fp16 0xc931
static constexpr float KOFF_F = 1024.f * KINV_F + KC_F; // -3.453125f
static constexpr float KOFF_C = -0.0018157958984375f;  // KOFF + 510 * KINV
static constexpr float KINV_S = 222.62571716308594f;
static constexpr float KOFF_S = -226.07884216308594f;
static constexpr float BETA_S = 1.01551091670990f;

__device__ __forceinline__ float reduce_block_sum(float v, float* red8) {
#pragma unroll
    for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
    if ((threadIdx.x & 31) == 0) red8[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += red8[i];
    return s;
}

template <int PERM, bool WANT_SXT>
__device__ __forceinline__ float stage_f32(const float* __restrict__ xg, int k, float* xs, float* sxt, float* red8) {
    float sxl = 0.f;
    for (int e = threadIdx.x; e < k; e += 256) {
        const float v = xg[e];
        sxl += v;
        if constexpr (PERM == 0) {
            xs[e] = v;
        } else if constexpr (PERM == 1) {
            int r = e & 15;
            xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = v;
        } else {
            int r = e & 15;
            xs[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3)] = v;
        }
    }
    if constexpr (WANT_SXT) {
        const int tk = k >> 4;
        for (int a = threadIdx.x >> 4; a < tk; a += 16) {
            float v = xg[a * 16 + (threadIdx.x & 15)];
#pragma unroll
            for (int s = 8; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if ((threadIdx.x & 15) == 0) sxt[a] = -BETA_S * v;
        }
    }
    return reduce_block_sum(sxl, red8);
}

template <int BLK, int PERM>
__device__ __forceinline__ float stage_q16(const float* __restrict__ xg, int k, uint32_t* xq, float* qs, float* red8) {
    float sxq = 0.f;
    if constexpr (BLK == 128) {
        const int nblk = k >> 7;
        for (int b = threadIdx.x >> 5; b < nblk; b += 8) {
            const int lane = threadIdx.x & 31;
            const int base = b * 128 + lane * 4;
            const float4 v = *(const float4*)&xg[base];
            float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
            for (int s = 16; s; s >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, s));
            const float sc = fmaxf(am, 1e-30f) * (1.f / 32767.f);
            const float inv = 32767.f / fmaxf(am, 1e-30f);
            if (lane == 0) qs[b] = sc;
            int q0 = __float2int_rn(v.x * inv), q1 = __float2int_rn(v.y * inv);
            int q2 = __float2int_rn(v.z * inv), q3 = __float2int_rn(v.w * inv);
            sxq += (float)(q0 + q1 + q2 + q3) * sc;
            uint32_t p0 = ((uint32_t)(q0 & 0xFFFF)) | ((uint32_t)(q1 & 0xFFFF) << 16);
            uint32_t p1 = ((uint32_t)(q2 & 0xFFFF)) | ((uint32_t)(q3 & 0xFFFF) << 16);
            int pair0 = (base >> 1), pair1 = pair0 + 1;
            if constexpr (PERM == 0) {
                xq[pair0] = p0; xq[pair1] = p1;
            } else if constexpr (PERM == 1) {
                int i0 = pair0 & 7, i1 = pair1 & 7;
                xq[(pair0 & ~7) + ((i0 & 2) ? 4 : 0) + ((i0 & 4) ? 2 : 0) + (i0 & 1)] = p0;
                xq[(pair1 & ~7) + ((i1 & 2) ? 4 : 0) + ((i1 & 4) ? 2 : 0) + (i1 & 1)] = p1;
            } else {
                int i0 = pair0 & 7, i1 = pair1 & 7;
                xq[(pair0 & ~7) + (i0 & 3) * 2 + (i0 >> 2)] = p0;
                xq[(pair1 & ~7) + (i1 & 3) * 2 + (i1 >> 2)] = p1;
            }
        }
    } else {
        const int nblk = k >> 4;
        for (int b = threadIdx.x >> 3; b < nblk; b += 32) {
            const int sub = threadIdx.x & 7;
            const float2 v = *(const float2*)&xg[b * 16 + sub * 2];
            float am = fmaxf(fabsf(v.x), fabsf(v.y));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
            const float sc = fmaxf(am, 1e-30f) * (1.f / 32767.f);
            const float inv = 32767.f / fmaxf(am, 1e-30f);
            if (sub == 0) qs[b] = sc;
            int q0 = __float2int_rn(v.x * inv), q1 = __float2int_rn(v.y * inv);
            sxq += (float)(q0 + q1) * sc;
            uint32_t p = ((uint32_t)(q0 & 0xFFFF)) | ((uint32_t)(q1 & 0xFFFF) << 16);
            if constexpr (PERM == 0) xq[b * 8 + sub] = p;
            else if constexpr (PERM == 1) xq[b * 8 + ((sub & 2) ? 4 : 0) + ((sub & 4) ? 2 : 0) + (sub & 1)] = p;
            else xq[b * 8 + (sub & 3) * 2 + (sub >> 2)] = p;
        }
    }
    return reduce_block_sum(sxq, red8);
}

__device__ __forceinline__ float stage_q16_splat(const float* __restrict__ xg, int k, uint32_t* xq, float* qs, float* red8) {
    float sxq = 0.f;
    const int nblk = k >> 7;
    for (int b = threadIdx.x >> 5; b < nblk; b += 8) {
        const int lane = threadIdx.x & 31;
        const int base = b * 128 + lane * 4;
        const float4 v = *(const float4*)&xg[base];
        float am = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
        for (int s = 16; s; s >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, s));
        const float sc = fmaxf(am, 1e-30f) * (1.f / 32767.f);
        const float inv = 32767.f / fmaxf(am, 1e-30f);
        if (lane == 0) qs[b] = sc;
        float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int q = __float2int_rn(vals[j] * inv);
            sxq += (float)q * sc;
            int e = base + j, r = e & 15;
            xq[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = (uint32_t)(q & 0xFFFF) * 0x00010001u;
        }
    }
    return reduce_block_sum(sxq, red8);
}

template <int PERM>
__device__ __forceinline__ float stage_q8(const float* __restrict__ xg, int k, uint32_t* xq, float* qs, float* red8) {
    float sxq = 0.f;
    const int nblk = k >> 4;
    for (int b = threadIdx.x >> 4; b < nblk; b += 16) {
        const int sub = threadIdx.x & 15;
        const float v = xg[b * 16 + sub];
        float am = fabsf(v);
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        const float sc = fmaxf(am, 1e-30f) * (1.f / 127.f);
        const float inv = 127.f / fmaxf(am, 1e-30f);
        if (sub == 0) qs[b] = sc;
        int q = __float2int_rn(v * inv);
        sxq += (float)q * sc;
        uint32_t sp = (uint32_t)(q & 0xFF) * 0x01010101u;
        int e = b * 16 + sub, r = sub;
        if constexpr (PERM == 0) xq[e] = sp;
        else if constexpr (PERM == 1) xq[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = sp;
        else xq[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3)] = sp;
    }
    return reduce_block_sum(sxq, red8);
}

// ============================================================================
// KERNEL FAMILY A: M=8 UPGRADED (32 lanes/tile, exact shipped grid, unroll-2/4)
//   MODE: 0=C0c (exact I2F), 1=C3b (folded FP32), 2=C4 (blk=128), 3=C4 (blk=16), 4=C6 (blk=16)
// ============================================================================
template <int MODE>
__device__ __forceinline__ void m8_k2_run(
    const uint32_t* __restrict__ tb, int tn, int tn_i, int tk,
    const float* xs, const float* sxt, const uint32_t* xq, const float* qs,
    uint32_t CM, float& a0, float& a1) {
    const int t = threadIdx.x & 31;
    const int w0 = t >> 1, o = (t & 1) << 4, w1 = (w0 + 1) & 15;
    const int r0 = 2 * (t & 3);
    const size_t stride = (size_t)tn * 16;
    const uint32_t* p0 = tb + (size_t)tn_i * 16 + w0;
    const uint32_t* p1 = tb + (size_t)tn_i * 16 + w1;

    if constexpr (MODE <= 1) {
#pragma unroll 4
        for (int a = 0; a < tk; ++a, p0 += stride, p1 += stride) {
            const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
            const uint32_t hi = __funnelshift_l(W1, W0, o);
            uint32_t pp[8];
            pp[0] = mul16_h1(hi, CM);      pp[1] = mul16_h1(hi << 2, CM);
            pp[2] = mul16_h1(hi << 4, CM); pp[3] = mul16_h1(hi << 6, CM);
            pp[4] = mul16_h1(hi << 8, CM); pp[5] = mul16_h1(hi << 10, CM);
            pp[6] = mul16_h1(hi << 12, CM);pp[7] = mul16_h1(hi << 14, CM);
            const float2 xa = *(const float2*)&xs[a * 16 + r0], xc = *(const float2*)&xs[a * 16 + r0 + 8];
            float f[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (MODE == 0) f[j] = (float)(int)dp4a_uu(pp[j], 0x01010101u, (uint32_t)(-510));
                else f[j] = __uint_as_float(dp4a_uu(pp[j], 0xFFFFFFFFu, 0x3F800000u));
            }
            if constexpr (MODE == 1) {
                float t0 = ((t & 3) == 0) ? sxt[a] : 0.f, t1 = t0;
                t0 = fmaf(xa.x, f[0], t0); t0 = fmaf(xa.y, f[1], t0); t0 = fmaf(xc.x, f[2], t0); t0 = fmaf(xc.y, f[3], t0);
                t1 = fmaf(xa.x, f[4], t1); t1 = fmaf(xa.y, f[5], t1); t1 = fmaf(xc.x, f[6], t1); t1 = fmaf(xc.y, f[7], t1);
                a0 += t0; a1 += t1;
            } else {
                a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0); a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
                a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1); a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
            }
        }
    } else if constexpr (MODE == 2) {
        for (int b = 0; b < tk; b += 8) {
            int ia0 = 0, ia1 = 0;
#pragma unroll 2
            for (int u = 0; u < 8; ++u, p0 += stride, p1 += stride) {
                const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                const uint32_t hi = __funnelshift_l(W1, W0, o);
                const uint32_t xp0 = xq[(b + u) * 8 + (r0 >> 1)];
                const uint32_t xp1 = xq[(b + u) * 8 + (r0 >> 1) + 4];
                ia0 = xmad_s16_h0(xp0, (int)dp4a_uu(mul16_h1(hi, CM),      0x01010101u, (uint32_t)(-510)), ia0);
                ia0 = xmad_s16_h1(xp0, (int)dp4a_uu(mul16_h1(hi << 2, CM), 0x01010101u, (uint32_t)(-510)), ia0);
                ia0 = xmad_s16_h0(xp1, (int)dp4a_uu(mul16_h1(hi << 4, CM), 0x01010101u, (uint32_t)(-510)), ia0);
                ia0 = xmad_s16_h1(xp1, (int)dp4a_uu(mul16_h1(hi << 6, CM), 0x01010101u, (uint32_t)(-510)), ia0);
                ia1 = xmad_s16_h0(xp0, (int)dp4a_uu(mul16_h1(hi << 8, CM), 0x01010101u, (uint32_t)(-510)), ia1);
                ia1 = xmad_s16_h1(xp0, (int)dp4a_uu(mul16_h1(hi << 10, CM),0x01010101u, (uint32_t)(-510)), ia1);
                ia1 = xmad_s16_h0(xp1, (int)dp4a_uu(mul16_h1(hi << 12, CM),0x01010101u, (uint32_t)(-510)), ia1);
                ia1 = xmad_s16_h1(xp1, (int)dp4a_uu(mul16_h1(hi << 14, CM),0x01010101u, (uint32_t)(-510)), ia1);
            }
            const float sc = qs[b >> 3];
            a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
        }
    } else if constexpr (MODE == 3) {
#pragma unroll 2
        for (int a = 0; a < tk; ++a, p0 += stride, p1 += stride) {
            const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
            const uint32_t hi = __funnelshift_l(W1, W0, o);
            const uint32_t xp0 = xq[a * 8 + (r0 >> 1)], xp1 = xq[a * 8 + (r0 >> 1) + 4];
            int ia0 = 0, ia1 = 0;
            ia0 = xmad_s16_h0(xp0, (int)dp4a_uu(mul16_h1(hi, CM),      0x01010101u, (uint32_t)(-510)), ia0);
            ia0 = xmad_s16_h1(xp0, (int)dp4a_uu(mul16_h1(hi << 2, CM), 0x01010101u, (uint32_t)(-510)), ia0);
            ia0 = xmad_s16_h0(xp1, (int)dp4a_uu(mul16_h1(hi << 4, CM), 0x01010101u, (uint32_t)(-510)), ia0);
            ia0 = xmad_s16_h1(xp1, (int)dp4a_uu(mul16_h1(hi << 6, CM), 0x01010101u, (uint32_t)(-510)), ia0);
            ia1 = xmad_s16_h0(xp0, (int)dp4a_uu(mul16_h1(hi << 8, CM), 0x01010101u, (uint32_t)(-510)), ia1);
            ia1 = xmad_s16_h1(xp0, (int)dp4a_uu(mul16_h1(hi << 10, CM),0x01010101u, (uint32_t)(-510)), ia1);
            ia1 = xmad_s16_h0(xp1, (int)dp4a_uu(mul16_h1(hi << 12, CM),0x01010101u, (uint32_t)(-510)), ia1);
            ia1 = xmad_s16_h1(xp1, (int)dp4a_uu(mul16_h1(hi << 14, CM),0x01010101u, (uint32_t)(-510)), ia1);
            const float sc = qs[a];
            a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
        }
    } else {
#pragma unroll 4
        for (int a = 0; a < tk; ++a, p0 += stride, p1 += stride) {
            const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
            const uint32_t hi = __funnelshift_l(W1, W0, o);
            const uint2 xa = *(const uint2*)&xq[a * 16 + r0], xc = *(const uint2*)&xq[a * 16 + r0 + 8];
            int ia0 = 0, ia1 = 0;
            ia0 = dp4a_su(xa.x, mul16_h1(hi, CM), ia0);       ia0 = dp4a_su(xa.y, mul16_h1(hi << 2, CM), ia0);
            ia0 = dp4a_su(xc.x, mul16_h1(hi << 4, CM), ia0);  ia0 = dp4a_su(xc.y, mul16_h1(hi << 6, CM), ia0);
            ia1 = dp4a_su(xa.x, mul16_h1(hi << 8, CM), ia1);  ia1 = dp4a_su(xa.y, mul16_h1(hi << 10, CM), ia1);
            ia1 = dp4a_su(xc.x, mul16_h1(hi << 12, CM), ia1); ia1 = dp4a_su(xc.y, mul16_h1(hi << 14, CM), ia1);
            const float sc = qs[a];
            a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
        }
    }
}

template <int MODE>
__global__ void __launch_bounds__(256, 6) gemv_m8_k2(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const float* __restrict__ xh, const float* __restrict__ xh2,
    float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, uint32_t CM) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1, zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const float* xg = (zsel ? xh2 : xh) + (size_t)slot * k;
    __shared__ __align__(16) uint32_t xbuf[MAXK];
    __shared__ float aux[144];
    __shared__ float red[8];
    float sx = 0.f;
    if constexpr (MODE == 0) sx = stage_f32<0, false>(xg, k, (float*)xbuf, aux, red);
    else if constexpr (MODE == 1) sx = stage_f32<0, true>(xg, k, (float*)xbuf, aux, red);
    else if constexpr (MODE == 2) sx = stage_q16<128, 0>(xg, k, xbuf, aux, red);
    else if constexpr (MODE == 3) sx = stage_q16<16, 0>(xg, k, xbuf, aux, red);
    else sx = stage_q8<0>(xg, k, xbuf, aux, red);
    const int tn_i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tn_i >= tn) return;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    float a0 = 0.f, a1 = 0.f;
    m8_k2_run<MODE>(tb, tn, tn_i, tk, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1);
    a0 += __shfl_xor_sync(0xffffffffu, a0, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    a0 += __shfl_xor_sync(0xffffffffu, a0, 2); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    const int lane = threadIdx.x & 31;
    if ((lane & 3) == 0) {
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (lane >> 2)];
        if constexpr (MODE == 1) { dst[0] = KINV_S * a0; dst[8] = KINV_S * a1; }
        else if constexpr (MODE == 4) { dst[0] = fmaf(KINV_F, a0, KOFF_F * sx); dst[8] = fmaf(KINV_F, a1, KOFF_F * sx); }
        else { dst[0] = fmaf(KINV_F, a0, KOFF_C * sx); dst[8] = fmaf(KINV_F, a1, KOFF_C * sx); }
    }
}

// ============================================================================
// KERNEL FAMILY B: M=16 PAIRED (16 lanes/tile, 2-way in-block tk-split -> EXACT 8 TILES/BLOCK!)
//   FIXES vs v1:
//     - Vectorized LDS.128 for ALL 7 K (2x LDS.128 for FP32/C6, 1x LDS.128 for C4!)
//     - Fixed 8-tile aligned blocks for C4 blk=128 (0 spills, unroll-2!)
//   MODE: 0=C0c (exact I2F), 1=C1c (exact Magic), 2=C3b (folded per-tile),
//         3=C3a (folded global), 4=C4 (blk=128), 5=C4 (blk=16), 6=C6 (blk=16), 7=LEA+2*DP2A
// ============================================================================
template <int MODE>
__device__ __forceinline__ void m16_pair_f(uint32_t U, uint32_t CM, float x1, float x0, float& a1, float& a0) {
    uint32_t p1 = mul16_h1(U, CM), p0 = mul16_h0(U, CM);
    if constexpr (MODE == 0) {
        a1 = fmaf(x1, (float)(int)dp4a_uu(p1, 0x01010101u, (uint32_t)(-510)), a1);
        a0 = fmaf(x0, (float)(int)dp4a_uu(p0, 0x01010101u, (uint32_t)(-510)), a0);
    } else if constexpr (MODE == 1) {
        a1 = fmaf(x1, __uint_as_float(dp4a_uu(p1, 0x01010101u, 0x4B000000u)) - 8389118.f, a1);
        a0 = fmaf(x0, __uint_as_float(dp4a_uu(p0, 0x01010101u, 0x4B000000u)) - 8389118.f, a0);
    } else {
        a1 = fmaf(x1, __uint_as_float(dp4a_uu(p1, 0xFFFFFFFFu, 0x3F800000u)), a1);
        a0 = fmaf(x0, __uint_as_float(dp4a_uu(p0, 0xFFFFFFFFu, 0x3F800000u)), a0);
    }
}
__device__ __forceinline__ void m16_pair_c4(uint32_t U, uint32_t CM, uint32_t xp1, bool hi1, uint32_t xp0, bool hi0, int& a1, int& a0) {
    int b1 = (int)dp4a_uu(mul16_h1(U, CM), 0x01010101u, (uint32_t)(-510));
    int b0 = (int)dp4a_uu(mul16_h0(U, CM), 0x01010101u, (uint32_t)(-510));
    a1 = hi1 ? xmad_s16_h1(xp1, b1, a1) : xmad_s16_h0(xp1, b1, a1);
    a0 = hi0 ? xmad_s16_h1(xp0, b0, a0) : xmad_s16_h0(xp0, b0, a0);
}
__device__ __forceinline__ void m16_pair_c6(uint32_t U, uint32_t CM, uint32_t x1, uint32_t x0, int& a1, int& a0) {
    a1 = dp4a_su(x1, mul16_h1(U, CM), a1);
    a0 = dp4a_su(x0, mul16_h0(U, CM), a0);
}

template <int KA, int H, int MODE>
__device__ __forceinline__ void m16_run_anyk(
    const uint32_t* __restrict__ tb, int tn, int tn_i, int blk_lo, int blk_hi,
    const float* xs, const float* sxt, const uint32_t* xq, const float* qs,
    uint32_t CM, float& a0, float& a1) {
    // blk_lo..blk_hi are in units of 8-tile (128-element) blocks! So a_lo = blk_lo*8, a_hi = blk_hi*8.
    constexpr int NW = 8 * KA + 4 * H;
    const int sub = threadIdx.x & 15;
    const int hx_f = (sub & 1) ? 8 : 0;
    const int hx_16 = (sub & 1) ? 4 : 0;
    const size_t stride = (size_t)tn * NW;
    const int a_lo = blk_lo * 8, a_hi = blk_hi * 8;

    if constexpr (KA == 2 && H == 0) {
        const uint32_t* p0 = tb + ((size_t)a_lo * tn + tn_i) * 16 + sub;
        const uint32_t* p1 = tb + ((size_t)a_lo * tn + tn_i) * 16 + ((sub + 1) & 15);
        if constexpr (MODE <= 3) {
#pragma unroll 2
            for (int a = a_lo; a < a_hi; ++a, p0 += stride, p1 += stride) {
                const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                const float4 xA = *(const float4*)&xs[a * 16 + hx_f + 0];
                const float4 xB = *(const float4*)&xs[a * 16 + hx_f + 4];
                float t0 = (MODE == 2 && (sub & 1) == 0) ? sxt[a] : ((MODE == 2) ? 0.f : a0);
                float t1 = (MODE == 2 && (sub & 1) == 0) ? sxt[a] : ((MODE == 2) ? 0.f : a1);
                m16_pair_f<MODE>(W0,                         CM, xA.x, xA.z, t0, t0);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0, 2), CM, xA.y, xA.w, t0, t0);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0, 4), CM, xB.x, xB.z, t0, t0);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0, 6), CM, xB.y, xB.w, t0, t0);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0, 8), CM, xA.x, xA.z, t1, t1);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0,10), CM, xA.y, xA.w, t1, t1);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0,12), CM, xB.x, xB.z, t1, t1);
                m16_pair_f<MODE>(__funnelshift_l(W1, W0,14), CM, xB.y, xB.w, t1, t1);
                if constexpr (MODE == 2) { a0 += t0; a1 += t1; } else { a0 = t0; a1 = t1; }
            }
        } else if constexpr (MODE == 4) {
            // Fixed 8-tile blocks (compile-time 8, zero min(), 0 spills!)
            for (int b = blk_lo; b < blk_hi; ++b) {
                int ia0 = 0, ia1 = 0;
#pragma unroll 2
                for (int u = 0; u < 8; ++u, p0 += stride, p1 += stride) {
                    const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                    const uint4 xp = *(const uint4*)&xq[(b * 8 + u) * 8 + hx_16];
                    m16_pair_c4(W0,                         CM, xp.x, false, xp.y, false, ia0, ia0);
                    m16_pair_c4(__funnelshift_l(W1, W0, 2), CM, xp.x, true,  xp.y, true,  ia0, ia0);
                    m16_pair_c4(__funnelshift_l(W1, W0, 4), CM, xp.z, false, xp.w, false, ia0, ia0);
                    m16_pair_c4(__funnelshift_l(W1, W0, 6), CM, xp.z, true,  xp.w, true,  ia0, ia0);
                    m16_pair_c4(__funnelshift_l(W1, W0, 8), CM, xp.x, false, xp.y, false, ia1, ia1);
                    m16_pair_c4(__funnelshift_l(W1, W0,10), CM, xp.x, true,  xp.y, true,  ia1, ia1);
                    m16_pair_c4(__funnelshift_l(W1, W0,12), CM, xp.z, false, xp.w, false, ia1, ia1);
                    m16_pair_c4(__funnelshift_l(W1, W0,14), CM, xp.z, true,  xp.w, true,  ia1, ia1);
                }
                const float sc = qs[b];
                a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
            }
        } else if constexpr (MODE == 5) {
#pragma unroll 2
            for (int a = a_lo; a < a_hi; ++a, p0 += stride, p1 += stride) {
                const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                const uint4 xp = *(const uint4*)&xq[a * 8 + hx_16];
                int ia0 = 0, ia1 = 0;
                m16_pair_c4(W0,                         CM, xp.x, false, xp.y, false, ia0, ia0);
                m16_pair_c4(__funnelshift_l(W1, W0, 2), CM, xp.x, true,  xp.y, true,  ia0, ia0);
                m16_pair_c4(__funnelshift_l(W1, W0, 4), CM, xp.z, false, xp.w, false, ia0, ia0);
                m16_pair_c4(__funnelshift_l(W1, W0, 6), CM, xp.z, true,  xp.w, true,  ia0, ia0);
                m16_pair_c4(__funnelshift_l(W1, W0, 8), CM, xp.x, false, xp.y, false, ia1, ia1);
                m16_pair_c4(__funnelshift_l(W1, W0,10), CM, xp.x, true,  xp.y, true,  ia1, ia1);
                m16_pair_c4(__funnelshift_l(W1, W0,12), CM, xp.z, false, xp.w, false, ia1, ia1);
                m16_pair_c4(__funnelshift_l(W1, W0,14), CM, xp.z, true,  xp.w, true,  ia1, ia1);
                const float sc = qs[a];
                a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
            }
        } else if constexpr (MODE == 6) {
#pragma unroll 2
            for (int a = a_lo; a < a_hi; ++a, p0 += stride, p1 += stride) {
                const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                const uint4 xA = *(const uint4*)&xq[a * 16 + hx_f + 0];
                const uint4 xB = *(const uint4*)&xq[a * 16 + hx_f + 4];
                int ia0 = 0, ia1 = 0;
                m16_pair_c6(W0,                         CM, xA.x, xA.z, ia0, ia0);
                m16_pair_c6(__funnelshift_l(W1, W0, 2), CM, xA.y, xA.w, ia0, ia0);
                m16_pair_c6(__funnelshift_l(W1, W0, 4), CM, xB.x, xB.z, ia0, ia0);
                m16_pair_c6(__funnelshift_l(W1, W0, 6), CM, xB.y, xB.w, ia0, ia0);
                m16_pair_c6(__funnelshift_l(W1, W0, 8), CM, xA.x, xA.z, ia1, ia1);
                m16_pair_c6(__funnelshift_l(W1, W0,10), CM, xA.y, xA.w, ia1, ia1);
                m16_pair_c6(__funnelshift_l(W1, W0,12), CM, xB.x, xB.z, ia1, ia1);
                m16_pair_c6(__funnelshift_l(W1, W0,14), CM, xB.y, xB.w, ia1, ia1);
                const float sc = qs[a];
                a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
            }
        } else {
            for (int b = blk_lo; b < blk_hi; ++b) {
                int ia0 = 0, ia1 = 0;
#pragma unroll 2
                for (int u = 0; u < 8; ++u, p0 += stride, p1 += stride) {
                    const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                    const uint4 xA = *(const uint4*)&xq[(b * 8 + u) * 16 + hx_f + 0];
                    const uint4 xB = *(const uint4*)&xq[(b * 8 + u) * 16 + hx_f + 4];
                    auto mp = [&](uint32_t U, uint32_t x1, uint32_t x0, int& acc) {
                        acc = mac_lea_dp2a_h1(U, x1, acc); acc = mac_lea_dp2a_h0(U, x0, acc);
                    };
                    mp(W0,                         xA.x, xA.z, ia0);
                    mp(__funnelshift_l(W1, W0, 2), xA.y, xA.w, ia0);
                    mp(__funnelshift_l(W1, W0, 4), xB.x, xB.z, ia0);
                    mp(__funnelshift_l(W1, W0, 6), xB.y, xB.w, ia0);
                    mp(__funnelshift_l(W1, W0, 8), xA.x, xA.z, ia1);
                    mp(__funnelshift_l(W1, W0,10), xA.y, xA.w, ia1);
                    mp(__funnelshift_l(W1, W0,12), xB.x, xB.z, ia1);
                    mp(__funnelshift_l(W1, W0,14), xB.y, xB.w, ia1);
                }
                const float sc = qs[b];
                a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
            }
        }
        return;
    }

    // All other K in {1.5, 2.5, 3.0, 3.5, 4.0, 5.0} at M=16 with VECTORIZED LDS.128 & 0-XMAD stride!
    const int base_bit = sub * (16 * KA + 8 * H);
    const int w0 = base_bit >> 5, o = base_bit & 31;
    constexpr int MAX_SPAN = 15 * KA + 8 * H + 16;
    constexpr int NA = (MAX_SPAN + 31) / 32;
    const uint32_t stride_b = (uint32_t)(tn * NW * 4);
    const char* qb = (const char*)(tb + ((size_t)a_lo * tn + tn_i) * NW);
    // Precompute lane word indices (0..NW-1) ONCE outside loop -> 0 XMADs inside loop!
    int wi[NA + 1];
#pragma unroll
    for (int m = 0; m <= NA; ++m) { int t = w0 + m; wi[m] = (t >= NW) ? (t - NW) : t; }

    auto eval_one_tile = [&](const char* basep, int a, int& ia0, int& ia1) {
        const uint32_t* ptr = (const uint32_t*)basep;
        uint32_t W[NA];
        if constexpr (KA == 4 && H == 0) {
            W[0] = __ldg(ptr + wi[0]);
            W[1] = __ldg(ptr + wi[1]);
            W[2] = __ldg(ptr + wi[2]);
        } else {
            uint32_t rprev = __ldg(ptr + wi[0]);
#pragma unroll
            for (int m = 0; m < NA; ++m) {
                uint32_t rcur = __ldg(ptr + wi[m + 1]);
                W[m] = __funnelshift_l(rcur, rprev, o);
                rprev = rcur;
            }
        }
        // Vectorized LDS.128 once per tile (2x LDS.128 for FP32/C6, 1x LDS.128 for C4!)
        float xv_f[8];
        uint32_t xv_i[8];
        uint32_t xp_i[4];
        if constexpr (MODE <= 3) {
            const float4 xA = *(const float4*)&xs[a * 16 + hx_f + 0];
            const float4 xB = *(const float4*)&xs[a * 16 + hx_f + 4];
            xv_f[0]=xA.x; xv_f[1]=xA.y; xv_f[2]=xA.z; xv_f[3]=xA.w;
            xv_f[4]=xB.x; xv_f[5]=xB.y; xv_f[6]=xB.z; xv_f[7]=xB.w;
        } else if constexpr (MODE == 4 || MODE == 5) {
            const uint4 xp = *(const uint4*)&xq[a * 8 + hx_16];
            xp_i[0]=xp.x; xp_i[1]=xp.y; xp_i[2]=xp.z; xp_i[3]=xp.w;
        } else {
            const uint4 xA = *(const uint4*)&xq[a * 16 + hx_f + 0];
            const uint4 xB = *(const uint4*)&xq[a * 16 + hx_f + 4];
            xv_i[0]=xA.x; xv_i[1]=xA.y; xv_i[2]=xA.z; xv_i[3]=xA.w;
            xv_i[4]=xB.x; xv_i[5]=xB.y; xv_i[6]=xB.z; xv_i[7]=xB.w;
        }

        auto get_u = [&](int d) -> uint32_t {
            const int wi = d >> 5, bi = d & 31;
            if (bi == 0) return W[wi];
            if (bi <= 16) return W[wi] << bi;
            return __funnelshift_l(W[wi + 1], W[wi], bi);
        };
        auto do_mac_j = [&](int j, uint32_t U, bool use_h1, float& t0, float& t1) {
            const int s = j >> 3, k = j & 7;
            const int r_loc = ((k & 2) ? 4 : 0) + 2 * s + (k & 1);
            if constexpr (MODE <= 3) {
                float x = xv_f[r_loc];
                float& acc = (k & 4) ? t1 : t0;
                uint32_t p = use_h1 ? mul16_h1(U, CM) : mul16_h0(U, CM);
                if constexpr (MODE == 0) acc = fmaf(x, (float)(int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510)), acc);
                else if constexpr (MODE == 1) acc = fmaf(x, __uint_as_float(dp4a_uu(p, 0x01010101u, 0x4B000000u)) - 8389118.f, acc);
                else acc = fmaf(x, __uint_as_float(dp4a_uu(p, 0xFFFFFFFFu, 0x3F800000u)), acc);
            } else if constexpr (MODE == 4 || MODE == 5) {
                uint32_t xp = xp_i[r_loc >> 1];
                int& acc = (k & 4) ? ia1 : ia0;
                uint32_t p = use_h1 ? mul16_h1(U, CM) : mul16_h0(U, CM);
                int b = (int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510));
                acc = (r_loc & 1) ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
            } else if constexpr (MODE == 6) {
                uint32_t xv = xv_i[r_loc];
                int& acc = (k & 4) ? ia1 : ia0;
                uint32_t p = use_h1 ? mul16_h1(U, CM) : mul16_h0(U, CM);
                acc = dp4a_su(xv, p, acc);
            } else {
                uint32_t xv = xv_i[r_loc];
                int& acc = (k & 4) ? ia1 : ia0;
                acc = use_h1 ? mac_lea_dp2a_h1(U, xv, acc) : mac_lea_dp2a_h0(U, xv, acc);
            }
        };

        float t0 = (MODE == 2 && (sub & 1) == 0) ? sxt[a] : ((MODE == 2) ? 0.f : a0);
        float t1 = (MODE == 2 && (sub & 1) == 0) ? sxt[a] : ((MODE == 2) ? 0.f : a1);
        if constexpr (KA == 4 && H == 0) {
            do_mac_j(0,  W[0], true, t0, t1); do_mac_j(4,  W[0], false, t0, t1);
            do_mac_j(8,  W[1], true, t0, t1); do_mac_j(12, W[1], false, t0, t1);
            { uint32_t U = __funnelshift_l(W[1], W[0],  4); do_mac_j(1, U, true, t0, t1); do_mac_j(5, U, false, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  8); do_mac_j(2, U, true, t0, t1); do_mac_j(6, U, false, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 12); do_mac_j(3, U, true, t0, t1); do_mac_j(7, U, false, t0, t1); }
            { uint32_t U = __funnelshift_l(W[2], W[1],  4); do_mac_j(9, U, true, t0, t1); do_mac_j(13,U, false, t0, t1); }
            { uint32_t U = __funnelshift_l(W[2], W[1],  8); do_mac_j(10,U, true, t0, t1); do_mac_j(14,U, false, t0, t1); }
            { uint32_t U = __funnelshift_l(W[2], W[1], 12); do_mac_j(11,U, true, t0, t1); do_mac_j(15,U, false, t0, t1); }
        } else if constexpr (KA == 1 && H == 1) {
            do_mac_j(0, W[0], true, t0, t1);
            { uint32_t U = __funnelshift_l(W[1], W[0], 2); do_mac_j(1, U, true, t0, t1); do_mac_j(12, U, false, t0, t1); do_mac_j(2, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 5); do_mac_j(3, U, true, t0, t1); do_mac_j(14, U, false, t0, t1); do_mac_j(4, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 8); do_mac_j(5, U, true, t0, t1); do_mac_j(6, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],11); do_mac_j(7, U, true, t0, t1); do_mac_j(8, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],14); do_mac_j(9, U, true, t0, t1); do_mac_j(10,U + U, true, t0, t1); }
            do_mac_j(11, __funnelshift_l(W[1], W[0], 17), true, t0, t1);
            do_mac_j(13, __funnelshift_l(W[1], W[0], 20), true, t0, t1);
            do_mac_j(15, __funnelshift_l(W[1], W[0], 23), true, t0, t1);
        } else if constexpr (KA == 2 && H == 1) {
            // K=2.5: (D-1) SHF + IADD(U+U) pairing -> 6 SHF + 2 SHL + 7 IADD (halves shifter ops!)
            do_mac_j(0, W[0], true, t0, t1);
            { uint32_t U = __funnelshift_l(W[1], W[0],  2); do_mac_j(7,  U, false, t0, t1); do_mac_j(1, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  4); do_mac_j(8,  U, false, t0, t1); do_mac_j(2, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  7); do_mac_j(9,  U, false, t0, t1); do_mac_j(3, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  9); do_mac_j(10, U, false, t0, t1); do_mac_j(4, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 12); do_mac_j(11, U, false, t0, t1); do_mac_j(5, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 14); do_mac_j(12, U, false, t0, t1); do_mac_j(6, U + U, true, t0, t1); }
            do_mac_j(13, W[1] + W[1], true, t0, t1);
            do_mac_j(14, W[1] << 3, true, t0, t1);
            do_mac_j(15, W[1] << 6, true, t0, t1);
        } else if constexpr (KA == 3 && H == 0) {
            // K=3.0: (D-1) SHF + IADD(U+U) pairing -> 5 SHF + 4 SHL + 6 IADD
            do_mac_j(0, W[0], true, t0, t1);
            { uint32_t U = __funnelshift_l(W[1], W[0],  2); do_mac_j(6,  U, false, t0, t1); do_mac_j(1, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  5); do_mac_j(7,  U, false, t0, t1); do_mac_j(2, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  8); do_mac_j(8,  U, false, t0, t1); do_mac_j(3, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 11); do_mac_j(9,  U, false, t0, t1); do_mac_j(4, U + U, true, t0, t1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 14); do_mac_j(10, U, false, t0, t1); do_mac_j(5, U + U, true, t0, t1); }
            do_mac_j(11, W[1] + W[1], true, t0, t1);
            do_mac_j(12, W[1] << 4, true, t0, t1);
            do_mac_j(13, W[1] << 7, true, t0, t1);
            do_mac_j(14, W[1] << 10, true, t0, t1);
            do_mac_j(15, W[1] << 13, true, t0, t1);
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int D = j * KA + H * ((j + 1) >> 1);
                if ((D & 15) == 0) do_mac_j(j, W[D >> 5], (D & 16) == 0, t0, t1);
                else do_mac_j(j, get_u(D), true, t0, t1);
            }
        }
        if constexpr (MODE <= 3) {
            if constexpr (MODE == 2) { a0 += t0; a1 += t1; } else { a0 = t0; a1 = t1; }
        }
    };

    if constexpr (MODE <= 3) {
#pragma unroll 1
        for (int a = a_lo; a < a_hi; ++a, qb += stride_b) { int d0 = 0, d1 = 0; eval_one_tile(qb, a, d0, d1); }
    } else if constexpr (MODE == 4 || MODE == 7) {
        for (int b = blk_lo; b < blk_hi; ++b) {
            int ia0 = 0, ia1 = 0;
#pragma unroll 1
            for (int u = 0; u < 8; ++u, qb += stride_b) eval_one_tile(qb, b * 8 + u, ia0, ia1);
            const float sc = qs[b];
            a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
        }
    } else {
#pragma unroll 1
        for (int a = a_lo; a < a_hi; ++a, qb += stride_b) {
            int ia0 = 0, ia1 = 0;
            eval_one_tile(qb, a, ia0, ia1);
            const float sc = qs[a];
            a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
        }
    }
}

template <int MODE, bool ALL_K>
__global__ void __launch_bounds__(256, 6) gemv_m16(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const float* __restrict__ xh, const float* __restrict__ xh2,
    float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, uint32_t CM) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1, zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const float* xg = (zsel ? xh2 : xh) + (size_t)slot * k;
    __shared__ __align__(16) uint32_t xbuf[MAXK];
    __shared__ float aux[144];
    __shared__ float red[8];
    float sx = 0.f;
    if constexpr (MODE == 0 || MODE == 1 || MODE == 3) sx = stage_f32<1, false>(xg, k, (float*)xbuf, aux, red);
    else if constexpr (MODE == 2) sx = stage_f32<1, true>(xg, k, (float*)xbuf, aux, red);
    else if constexpr (MODE == 4) sx = stage_q16<128, 1>(xg, k, xbuf, aux, red);
    else if constexpr (MODE == 5) sx = stage_q16<16, 1>(xg, k, xbuf, aux, red);
    else if constexpr (MODE == 6) sx = stage_q8<1>(xg, k, xbuf, aux, red);
    else sx = stage_q16_splat(xg, k, xbuf, aux, red);

    const int part = threadIdx.x >> 7;
    const int local_tid = threadIdx.x & 127;
    const int t_in_blk = local_tid >> 4;
    const int tn_i = blockIdx.x * 8 + t_in_blk;
    // Align split to 8-tile (128-elem) block boundaries!
    const int nblk = tk >> 3;
    const int blk_lo = (part * nblk) >> 1;
    const int blk_hi = ((part + 1) * nblk) >> 1;
    float a0 = 0.f, a1 = 0.f;
    if (tn_i < tn) {
        const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
        if constexpr (!ALL_K) {
            m16_run_anyk<2, 0, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1);
        } else {
            const int ka = (int)(d[3] & 0xff), hk = (int)((d[3] >> 8) & 1);
            switch (ka * 2 + hk) {
                case 3:  m16_run_anyk<1, 1, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                case 4:  m16_run_anyk<2, 0, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                case 5:  m16_run_anyk<2, 1, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                case 6:  m16_run_anyk<3, 0, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                case 7:  m16_run_anyk<3, 1, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                case 8:  m16_run_anyk<4, 0, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                case 10: m16_run_anyk<5, 0, MODE>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
                default: __trap();
            }
        }
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    }
    __syncthreads();
    float* red_out = (float*)xbuf;
    const int sub = threadIdx.x & 15;
    if (part == 1 && (sub & 1) == 0 && tn_i < tn) {
        red_out[t_in_blk * 16 + (sub >> 1)] = a0;
        red_out[t_in_blk * 16 + (sub >> 1) + 8] = a1;
    }
    __syncthreads();
    if (part == 0 && (sub & 1) == 0 && tn_i < tn) {
        a0 += red_out[t_in_blk * 16 + (sub >> 1)];
        a1 += red_out[t_in_blk * 16 + (sub >> 1) + 8];
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (sub >> 1)];
        if constexpr (MODE == 2) { dst[0] = KINV_S * a0; dst[8] = KINV_S * a1; }
        else if constexpr (MODE == 3) { dst[0] = fmaf(KINV_S, a0, KOFF_S * sx); dst[8] = fmaf(KINV_S, a1, KOFF_S * sx); }
        else if constexpr (MODE == 6 || MODE == 7) { dst[0] = fmaf(KINV_F, a0, KOFF_F * sx); dst[8] = fmaf(KINV_F, a1, KOFF_F * sx); }
        else { dst[0] = fmaf(KINV_F, a0, KOFF_C * sx); dst[8] = fmaf(KINV_F, a1, KOFF_C * sx); }
    }
}

// ============================================================================
// KERNEL FAMILY C: M=32 LOSSLESS STRIDED-CRUMB + SHFL + ISCADD (K=2.0, 0% bloat!)
//   4-way in-block tk-split -> EXACT 8 tiles/block, 2 accs, <=40 regs, 0 spills!
// ============================================================================
template <int MODE>
__global__ void __launch_bounds__(256, 6) gemv_strided_m32_k2(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const float* __restrict__ xh, const float* __restrict__ xh2,
    float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, uint32_t CM) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1, zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const float* xg = (zsel ? xh2 : xh) + (size_t)slot * k;
    __shared__ __align__(16) uint32_t xbuf[MAXK];
    __shared__ float aux[144];
    __shared__ float red[8];
    float sx = 0.f;
    if constexpr (MODE == 0) sx = stage_f32<2, false>(xg, k, (float*)xbuf, aux, red);
    else if constexpr (MODE == 1) sx = stage_q16<128, 2>(xg, k, xbuf, aux, red);
    else sx = stage_q8<2>(xg, k, xbuf, aux, red);

    const int part = threadIdx.x >> 6;
    const int local_tid = threadIdx.x & 63;
    const int t_in_blk = local_tid >> 3;
    const int sub = local_tid & 7;
    const int tn_i = blockIdx.x * 8 + t_in_blk;
    const int a_lo = (part * tk) >> 2;
    const int a_hi = ((part + 1) * tk) >> 2;

    const int lane = threadIdx.x & 31;
    const uint32_t v_tbl = (lane >> 2) & 3u, u_tbl = lane & 3u;
    const uint32_t D_reg = v_tbl * 0x83DCD12Du - ((u_tbl * 0xD12Du) << 16);

    float a0 = 0.f, a1 = 0.f;
    if (tn_i < tn) {
        const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
        const size_t stride = (size_t)tn * 16;
        const uint32_t* p0 = tb + ((size_t)a_lo * tn + tn_i) * 16 + sub * 2;
        const uint32_t* pn = tb + ((size_t)a_lo * tn + tn_i) * 16 + ((sub + 1) & 7) * 2;

        auto run_tile = [&](const uint32_t* q0, const uint32_t* qn, int a, int& ia0, int& ia1) {
            const uint2 w01 = __ldg((const uint2*)q0);
            const uint2 w23 = __ldg((const uint2*)qn);
            const uint32_t W0 = w01.x, W1 = w01.y, W2 = w23.x, W3 = w23.y;
            const uint32_t h0_hi = dp4a_uu(W0 & 0x03030303u, 0x01041040u, 0u);
            const uint32_t h0_lo = dp4a_uu(W1 & 0x03030303u, 0x01041040u, 0u);
            uint32_t P = mul16_h0((h0_hi << 8) + h0_lo, CM);
            auto step = [&](uint32_t idx) { P = (P << 2) + __shfl_sync(0xFFFFFFFFu, D_reg, idx); };
            auto do_mac = [&](uint4 xv, int k) {
                if constexpr (MODE == 0) {
                    float x = (k&3)==0 ? __uint_as_float(xv.x) : ((k&3)==1 ? __uint_as_float(xv.y) : ((k&3)==2 ? __uint_as_float(xv.z) : __uint_as_float(xv.w)));
                    float& acc = (k & 4) ? a1 : a0;
                    acc = fmaf(x, (float)(int)dp4a_uu(P, 0x01010101u, (uint32_t)(-510)), acc);
                } else if constexpr (MODE == 1) {
                    uint32_t xp = (k & 2) ? xv.y : xv.x; int& acc = (k & 4) ? ia1 : ia0;
                    int b = (int)dp4a_uu(P, 0x01010101u, (uint32_t)(-510));
                    acc = (k & 1) ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
                } else {
                    uint32_t x = (k&3)==0 ? xv.x : ((k&3)==1 ? xv.y : ((k&3)==2 ? xv.z : xv.w));
                    int& acc = (k & 4) ? ia1 : ia0; acc = dp4a_su(x, P, acc);
                }
            };
            {
                const uint4 xv = (MODE == 1) ? make_uint4(xbuf[a*8+0], xbuf[a*8+1], 0, 0) : *(const uint4*)&xbuf[a*16+0];
                do_mac(xv, 0);
                step(W0); do_mac(xv, 1); step(W0>>8); do_mac(xv, 2); step(W0>>16); do_mac(xv, 3); step(W0>>24); do_mac(xv, 4);
                step(W1); do_mac(xv, 5); step(W1>>8); do_mac(xv, 6); step(W1>>16); do_mac(xv, 7);
            }
            {
                const uint4 xv = (MODE == 1) ? make_uint4(xbuf[a*8+2], xbuf[a*8+3], 0, 0) : *(const uint4*)&xbuf[a*16+4];
                step(W1>>24); do_mac(xv, 0);
                step(W0>>2);  do_mac(xv, 1); step(W0>>10); do_mac(xv, 2); step(W0>>18); do_mac(xv, 3); step(W0>>26); do_mac(xv, 4);
                step(W1>>2);  do_mac(xv, 5); step(W1>>10); do_mac(xv, 6); step(W1>>18); do_mac(xv, 7);
            }
            {
                const uint4 xv = (MODE == 1) ? make_uint4(xbuf[a*8+4], xbuf[a*8+5], 0, 0) : *(const uint4*)&xbuf[a*16+8];
                step(W1>>26); do_mac(xv, 0);
                step(W0>>4);  do_mac(xv, 1); step(W0>>12); do_mac(xv, 2); step(W0>>20); do_mac(xv, 3); step(W0>>28); do_mac(xv, 4);
                step(W1>>4);  do_mac(xv, 5); step(W1>>12); do_mac(xv, 6); step(W1>>20); do_mac(xv, 7);
            }
            {
                const uint4 xv = (MODE == 1) ? make_uint4(xbuf[a*8+6], xbuf[a*8+7], 0, 0) : *(const uint4*)&xbuf[a*16+12];
                const uint32_t W02 = ((W0 >> 6) & 0x03030303u) | ((W2 & 0x03030303u) << 2);
                const uint32_t W13 = ((W1 >> 6) & 0x03030303u) | ((W3 & 0x03030303u) << 2);
                step(W1>>28); do_mac(xv, 0);
                step(W02);    do_mac(xv, 1); step(W02>>8);  do_mac(xv, 2); step(W02>>16); do_mac(xv, 3); step(W02>>24); do_mac(xv, 4);
                step(W13);    do_mac(xv, 5); step(W13>>8);  do_mac(xv, 6); step(W13>>16); do_mac(xv, 7);
            }
        };

        if constexpr (MODE == 0) {
#pragma unroll 1
            for (int a = a_lo; a < a_hi; ++a, p0 += stride, pn += stride) { int d0 = 0, d1 = 0; run_tile(p0, pn, a, d0, d1); }
        } else if constexpr (MODE == 1) {
            for (int a = a_lo; a < a_hi; ) {
                const int nb = min(8 - (a & 7), a_hi - a);
                int ia0 = 0, ia1 = 0;
                for (int u = 0; u < nb; ++u, p0 += stride, pn += stride) run_tile(p0, pn, a + u, ia0, ia1);
                const float sc = aux[a >> 3];
                a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
                a += nb;
            }
        } else {
#pragma unroll 1
            for (int a = a_lo; a < a_hi; ++a, p0 += stride, pn += stride) {
                int ia0 = 0, ia1 = 0;
                run_tile(p0, pn, a, ia0, ia1);
                const float sc = aux[a];
                a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
            }
        }
    }
    __syncthreads();
    float* red_out = (float*)xbuf;
    if (tn_i < tn) {
        red_out[(part * 8 + t_in_blk) * 16 + sub] = a0;
        red_out[(part * 8 + t_in_blk) * 16 + sub + 8] = a1;
    }
    __syncthreads();
    if (part == 0 && tn_i < tn) {
        float s0 = red_out[(0 * 8 + t_in_blk) * 16 + sub] + red_out[(1 * 8 + t_in_blk) * 16 + sub]
                 + red_out[(2 * 8 + t_in_blk) * 16 + sub] + red_out[(3 * 8 + t_in_blk) * 16 + sub];
        float s1 = red_out[(0 * 8 + t_in_blk) * 16 + sub + 8] + red_out[(1 * 8 + t_in_blk) * 16 + sub + 8]
                 + red_out[(2 * 8 + t_in_blk) * 16 + sub + 8] + red_out[(3 * 8 + t_in_blk) * 16 + sub + 8];
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + sub];
        if constexpr (MODE == 2) { dst[0] = fmaf(KINV_F, s0, KOFF_F * sx); dst[8] = fmaf(KINV_F, s1, KOFF_F * sx); }
        else { dst[0] = fmaf(KINV_F, s0, KOFF_C * sx); dst[8] = fmaf(KINV_F, s1, KOFF_C * sx); }
    }
}

// ============================================================================
// KERNEL FAMILY C2: HYBRID ALL-7-K (M=32-Strided on K=2.0 + M=16-Paired on K!=2.0)
//   CTA-uniform dispatch on K -> zero warp divergence, exact 8 tiles/block, 0% bloat!
//   HYB_MODE: 0 = Exact C0c (I2F), 1 = C4 (M=32-Strided blk=128 on K2 + M=16-Paired blk=16 on K!=2)
// ============================================================================
template <int HYB_MODE>
__global__ void __launch_bounds__(256, 6) gemv_hybrid_allk(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const float* __restrict__ xh, const float* __restrict__ xh2,
    float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, uint32_t CM) {
    const int slot = blockIdx.y;
    const int zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const float* xg = (zsel ? xh2 : xh) + (size_t)slot * k;
    const int ka = (int)(d[3] & 0xff), hk = (int)((d[3] >> 8) & 1);
    __shared__ __align__(16) uint32_t xbuf[MAXK];
    __shared__ float aux[144];
    __shared__ float red[8];

    if (ka == 2 && hk == 0) {
        // 4-way tk-split M=32-Strided path (40% of SAGE MoE weights!)
        float sx = (HYB_MODE == 0) ? stage_f32<2, false>(xg, k, (float*)xbuf, aux, red)
                                   : stage_q16<128, 2>(xg, k, xbuf, aux, red);
        const int part = threadIdx.x >> 6;
        const int local_tid = threadIdx.x & 63;
        const int t_in_blk = local_tid >> 3;
        const int sub = local_tid & 7;
        const int tn_i = blockIdx.x * 8 + t_in_blk;
        const int a_lo = (part * tk) >> 2, a_hi = ((part + 1) * tk) >> 2;
        const int lane = threadIdx.x & 31;
        const uint32_t D_reg = ((uint32_t)(lane >> 2) & 3u) * 0x83DCD12Du - (((uint32_t)(lane & 3u) * 0xD12Du) << 16);
        float a0 = 0.f, a1 = 0.f;
        if (tn_i < tn) {
            const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
            const size_t stride = (size_t)tn * 16;
            const uint32_t* p0 = tb + ((size_t)a_lo * tn + tn_i) * 16 + sub * 2;
            const uint32_t* pn = tb + ((size_t)a_lo * tn + tn_i) * 16 + ((sub + 1) & 7) * 2;
            auto run_tile = [&](const uint32_t* q0, const uint32_t* qn, int a, int& ia0, int& ia1) {
                const uint2 w01 = __ldg((const uint2*)q0), w23 = __ldg((const uint2*)qn);
                const uint32_t W0 = w01.x, W1 = w01.y, W2 = w23.x, W3 = w23.y;
                const uint32_t h0_hi = dp4a_uu(W0 & 0x03030303u, 0x01041040u, 0u);
                const uint32_t h0_lo = dp4a_uu(W1 & 0x03030303u, 0x01041040u, 0u);
                uint32_t P = mul16_h0((h0_hi << 8) + h0_lo, CM);
                auto step = [&](uint32_t idx) { P = (P << 2) + __shfl_sync(0xFFFFFFFFu, D_reg, idx); };
                auto do_mac = [&](uint4 xv, int kk) {
                    if constexpr (HYB_MODE == 0) {
                        float x = (kk&3)==0 ? __uint_as_float(xv.x) : ((kk&3)==1 ? __uint_as_float(xv.y) : ((kk&3)==2 ? __uint_as_float(xv.z) : __uint_as_float(xv.w)));
                        float& acc = (kk & 4) ? a1 : a0;
                        acc = fmaf(x, (float)(int)dp4a_uu(P, 0x01010101u, (uint32_t)(-510)), acc);
                    } else {
                        uint32_t xp = (kk & 2) ? xv.y : xv.x; int& acc = (kk & 4) ? ia1 : ia0;
                        int b = (int)dp4a_uu(P, 0x01010101u, (uint32_t)(-510));
                        acc = (kk & 1) ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
                    }
                };
                {
                    const uint4 xv = (HYB_MODE == 1) ? make_uint4(xbuf[a*8+0], xbuf[a*8+1], 0, 0) : *(const uint4*)&xbuf[a*16+0];
                    do_mac(xv, 0);
                    step(W0); do_mac(xv, 1); step(W0>>8); do_mac(xv, 2); step(W0>>16); do_mac(xv, 3); step(W0>>24); do_mac(xv, 4);
                    step(W1); do_mac(xv, 5); step(W1>>8); do_mac(xv, 6); step(W1>>16); do_mac(xv, 7);
                }
                {
                    const uint4 xv = (HYB_MODE == 1) ? make_uint4(xbuf[a*8+2], xbuf[a*8+3], 0, 0) : *(const uint4*)&xbuf[a*16+4];
                    step(W1>>24); do_mac(xv, 0);
                    step(W0>>2);  do_mac(xv, 1); step(W0>>10); do_mac(xv, 2); step(W0>>18); do_mac(xv, 3); step(W0>>26); do_mac(xv, 4);
                    step(W1>>2);  do_mac(xv, 5); step(W1>>10); do_mac(xv, 6); step(W1>>18); do_mac(xv, 7);
                }
                {
                    const uint4 xv = (HYB_MODE == 1) ? make_uint4(xbuf[a*8+4], xbuf[a*8+5], 0, 0) : *(const uint4*)&xbuf[a*16+8];
                    step(W1>>26); do_mac(xv, 0);
                    step(W0>>4);  do_mac(xv, 1); step(W0>>12); do_mac(xv, 2); step(W0>>20); do_mac(xv, 3); step(W0>>28); do_mac(xv, 4);
                    step(W1>>4);  do_mac(xv, 5); step(W1>>12); do_mac(xv, 6); step(W1>>20); do_mac(xv, 7);
                }
                {
                    const uint4 xv = (HYB_MODE == 1) ? make_uint4(xbuf[a*8+6], xbuf[a*8+7], 0, 0) : *(const uint4*)&xbuf[a*16+12];
                    const uint32_t W02 = ((W0 >> 6) & 0x03030303u) | ((W2 & 0x03030303u) << 2);
                    const uint32_t W13 = ((W1 >> 6) & 0x03030303u) | ((W3 & 0x03030303u) << 2);
                    step(W1>>28); do_mac(xv, 0);
                    step(W02);    do_mac(xv, 1); step(W02>>8);  do_mac(xv, 2); step(W02>>16); do_mac(xv, 3); step(W02>>24); do_mac(xv, 4);
                    step(W13);    do_mac(xv, 5); step(W13>>8);  do_mac(xv, 6); step(W13>>16); do_mac(xv, 7);
                }
            };
            if constexpr (HYB_MODE == 0) {
#pragma unroll 1
                for (int a = a_lo; a < a_hi; ++a, p0 += stride, pn += stride) { int d0 = 0, d1 = 0; run_tile(p0, pn, a, d0, d1); }
            } else {
                for (int a = a_lo; a < a_hi; ) {
                    const int nb = min(8 - (a & 7), a_hi - a);
                    int ia0 = 0, ia1 = 0;
                    for (int u = 0; u < nb; ++u, p0 += stride, pn += stride) run_tile(p0, pn, a + u, ia0, ia1);
                    const float sc = aux[a >> 3];
                    a0 = fmaf(sc, (float)ia0, a0); a1 = fmaf(sc, (float)ia1, a1);
                    a += nb;
                }
            }
        }
        __syncthreads();
        float* red_out = (float*)xbuf;
        if (tn_i < tn) { red_out[(part * 8 + t_in_blk) * 16 + sub] = a0; red_out[(part * 8 + t_in_blk) * 16 + sub + 8] = a1; }
        __syncthreads();
        if (part == 0 && tn_i < tn) {
            float s0 = red_out[(0 * 8 + t_in_blk) * 16 + sub] + red_out[(1 * 8 + t_in_blk) * 16 + sub]
                     + red_out[(2 * 8 + t_in_blk) * 16 + sub] + red_out[(3 * 8 + t_in_blk) * 16 + sub];
            float s1 = red_out[(0 * 8 + t_in_blk) * 16 + sub + 8] + red_out[(1 * 8 + t_in_blk) * 16 + sub + 8]
                     + red_out[(2 * 8 + t_in_blk) * 16 + sub + 8] + red_out[(3 * 8 + t_in_blk) * 16 + sub + 8];
            float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + sub];
            dst[0] = fmaf(KINV_F, s0, KOFF_C * sx); dst[8] = fmaf(KINV_F, s1, KOFF_C * sx);
        }
        return;
    }

    // 2-way tk-split M=16-Paired path for K != 2.0
    constexpr int M16M = (HYB_MODE == 0) ? 0 : 5;
    float sx = (HYB_MODE == 0) ? stage_f32<1, false>(xg, k, (float*)xbuf, aux, red)
                               : stage_q16<16, 1>(xg, k, xbuf, aux, red);
    const int part = threadIdx.x >> 7, local_tid = threadIdx.x & 127;
    const int t_in_blk = local_tid >> 4, tn_i = blockIdx.x * 8 + t_in_blk;
    const int nblk = tk >> 3, blk_lo = (part * nblk) >> 1, blk_hi = ((part + 1) * nblk) >> 1;
    float a0 = 0.f, a1 = 0.f;
    if (tn_i < tn) {
        const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
        switch (ka * 2 + hk) {
            case 3:  m16_run_anyk<1, 1, M16M>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
            case 5:  m16_run_anyk<2, 1, M16M>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
            case 6:  m16_run_anyk<3, 0, M16M>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
            case 7:  m16_run_anyk<3, 1, M16M>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
            case 8:  m16_run_anyk<4, 0, M16M>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
            case 10: m16_run_anyk<5, 0, M16M>(tb, tn, tn_i, blk_lo, blk_hi, (const float*)xbuf, aux, xbuf, aux, CM, a0, a1); break;
            default: __trap();
        }
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    }
    __syncthreads();
    float* red_out = (float*)xbuf;
    const int sub = threadIdx.x & 15;
    if (part == 1 && (sub & 1) == 0 && tn_i < tn) { red_out[t_in_blk * 16 + (sub >> 1)] = a0; red_out[t_in_blk * 16 + (sub >> 1) + 8] = a1; }
    __syncthreads();
    if (part == 0 && (sub & 1) == 0 && tn_i < tn) {
        a0 += red_out[t_in_blk * 16 + (sub >> 1)];
        a1 += red_out[t_in_blk * 16 + (sub >> 1) + 8];
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (sub >> 1)];
        dst[0] = fmaf(KINV_F, a0, KOFF_C * sx); dst[8] = fmaf(KINV_F, a1, KOFF_C * sx);
    }
}

// ============================================================================
// KERNEL FAMILY D: NEGATIVE LUT PROOFS (128 KB int16 exact & 64 KB int8 approx)
// ============================================================================
template <typename T_LUT, int SCALE>
__global__ void __launch_bounds__(256, 6) gemv_lut_m16_k2(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const float* __restrict__ xh, const float* __restrict__ xh2,
    const T_LUT* __restrict__ lut,
    float* __restrict__ t, int tzoff, int tk, int tn, int k, int n) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1, zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const float* xg = (zsel ? xh2 : xh) + (size_t)slot * k;
    __shared__ __align__(16) float xs[MAXK];
    __shared__ float red[8];
    const float sx = stage_f32<1, false>(xg, k, xs, nullptr, red);

    const int part = threadIdx.x >> 7;
    const int local_tid = threadIdx.x & 127;
    const int t_in_blk = local_tid >> 4, sub = local_tid & 15;
    const int tn_i = blockIdx.x * 8 + t_in_blk;
    const int hx = (sub & 1) ? 8 : 0;
    const int a_lo = (part * tk) >> 1, a_hi = ((part + 1) * tk) >> 1;
    float a0 = 0.f, a1 = 0.f;
    if (tn_i < tn) {
        const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
        const size_t stride = (size_t)tn * 16;
        const uint32_t* p0 = tb + ((size_t)a_lo * tn + tn_i) * 16 + sub;
        const uint32_t* p1 = tb + ((size_t)a_lo * tn + tn_i) * 16 + ((sub + 1) & 15);
#pragma unroll 2
        for (int a = a_lo; a < a_hi; ++a, p0 += stride, p1 += stride) {
            const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
            const float4 xA = *(const float4*)&xs[a * 16 + hx], xB = *(const float4*)&xs[a * 16 + hx + 4];
            auto lp = [&](uint32_t U, float x1, float x0, float& acc) {
                acc = fmaf(x1, (float)__ldg(lut + (U >> 16)), acc);
                acc = fmaf(x0, (float)__ldg(lut + (U & 0xFFFFu)), acc);
            };
            lp(W0, xA.x, xA.z, a0); lp(__funnelshift_l(W1, W0, 2), xA.y, xA.w, a0);
            lp(__funnelshift_l(W1, W0, 4), xB.x, xB.z, a0); lp(__funnelshift_l(W1, W0, 6), xB.y, xB.w, a0);
            lp(__funnelshift_l(W1, W0, 8), xA.x, xA.z, a1); lp(__funnelshift_l(W1, W0,10), xA.y, xA.w, a1);
            lp(__funnelshift_l(W1, W0,12), xB.x, xB.z, a1); lp(__funnelshift_l(W1, W0,14), xB.y, xB.w, a1);
        }
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
        a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    }
    __syncthreads();
    if (part == 1 && (sub & 1) == 0 && tn_i < tn) {
        xs[t_in_blk * 16 + (sub >> 1)] = a0;
        xs[t_in_blk * 16 + (sub >> 1) + 8] = a1;
    }
    __syncthreads();
    if (part == 0 && (sub & 1) == 0 && tn_i < tn) {
        a0 += xs[t_in_blk * 16 + (sub >> 1)];
        a1 += xs[t_in_blk * 16 + (sub >> 1) + 8];
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (sub >> 1)];
        dst[0] = fmaf(KINV_F * (float)SCALE, a0, KOFF_C * sx);
        dst[8] = fmaf(KINV_F * (float)SCALE, a1, KOFF_C * sx);
    }
}

// ============================================================================
// KERNEL FAMILY E: DIRECTION 3 MULTI-TOKEN BATCHING (B = 1, 2, 3, 4) AT M=16!
// ============================================================================
template <int B>
__global__ void __launch_bounds__(256, 6) gemv_batch_m16_c0c(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const float* __restrict__ xgb, float* __restrict__ t,
    int tzoff, int tk, int tn, int k, int n, int nslots, uint32_t CM) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1, zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const float* xg = xgb + (size_t)(slot * nsel + zsel) * k * B;

    __shared__ __align__(16) float xs[32 * 16 * B];
    __shared__ float sx_sh[B];
    __shared__ float red[8];

    for (int b = 0; b < B; ++b) {
        float s = 0.f;
        for (int e = threadIdx.x; e < k; e += 256) s += xg[e * B + b];
        float tot = reduce_block_sum(s, red);
        if (threadIdx.x == 0) sx_sh[b] = tot;
        __syncthreads();
    }

    const int part = threadIdx.x >> 7;
    const int local_tid = threadIdx.x & 127;
    const int t_in_blk = local_tid >> 4, sub = local_tid & 15;
    const int tn_i = blockIdx.x * 8 + t_in_blk;
    const int r_off = (sub & 1) ? 4 : 0;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    const size_t stride = (size_t)tn * 16;
    const uint32_t* p0 = tb + ((size_t)part * tn + tn_i) * 16 + sub;
    const uint32_t* p1 = tb + ((size_t)part * tn + tn_i) * 16 + ((sub + 1) & 15);
    const size_t step2 = 2 * stride;

    float a0[B], a1[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { a0[b] = 0.f; a1[b] = 0.f; }

    for (int tbase = 0; tbase < tk; tbase += 32) {
        const int nloc = min(32, tk - tbase);
        for (int e = threadIdx.x; e < nloc * 16 * B; e += 256) xs[e] = xg[(size_t)tbase * 16 * B + e];
        __syncthreads();
        if (tn_i < tn) {
#pragma unroll 1
            for (int ti = part; ti < nloc; ti += 2, p0 += step2, p1 += step2) {
                const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                const float* xt = &xs[ti * 16 * B];
                auto do_cols = [&](uint32_t U0, uint32_t U1, int r_hi, int r_lo) {
                    {
                        float f0 = (float)(int)dp4a_uu(mul16_h1(U0, CM), 0x01010101u, (uint32_t)(-510));
                        float f1 = (float)(int)dp4a_uu(mul16_h1(U1, CM), 0x01010101u, (uint32_t)(-510));
                        if constexpr (B == 1) {
                            float x = xt[r_hi]; a0[0] = fmaf(x, f0, a0[0]); a1[0] = fmaf(x, f1, a1[0]);
                        } else if constexpr (B == 2) {
                            float2 x = *(const float2*)&xt[r_hi * 2];
                            a0[0] = fmaf(x.x, f0, a0[0]); a0[1] = fmaf(x.y, f0, a0[1]);
                            a1[0] = fmaf(x.x, f1, a1[0]); a1[1] = fmaf(x.y, f1, a1[1]);
                        } else if constexpr (B == 3) {
                            float x0 = xt[r_hi*3+0], x1 = xt[r_hi*3+1], x2 = xt[r_hi*3+2];
                            a0[0] = fmaf(x0, f0, a0[0]); a0[1] = fmaf(x1, f0, a0[1]); a0[2] = fmaf(x2, f0, a0[2]);
                            a1[0] = fmaf(x0, f1, a1[0]); a1[1] = fmaf(x1, f1, a1[1]); a1[2] = fmaf(x2, f1, a1[2]);
                        } else {
                            float4 x = *(const float4*)&xt[r_hi * 4];
                            a0[0] = fmaf(x.x, f0, a0[0]); a0[1] = fmaf(x.y, f0, a0[1]); a0[2] = fmaf(x.z, f0, a0[2]); a0[3] = fmaf(x.w, f0, a0[3]);
                            a1[0] = fmaf(x.x, f1, a1[0]); a1[1] = fmaf(x.y, f1, a1[1]); a1[2] = fmaf(x.z, f1, a1[2]); a1[3] = fmaf(x.w, f1, a1[3]);
                        }
                    }
                    {
                        float f0 = (float)(int)dp4a_uu(mul16_h0(U0, CM), 0x01010101u, (uint32_t)(-510));
                        float f1 = (float)(int)dp4a_uu(mul16_h0(U1, CM), 0x01010101u, (uint32_t)(-510));
                        if constexpr (B == 1) {
                            float x = xt[r_lo]; a0[0] = fmaf(x, f0, a0[0]); a1[0] = fmaf(x, f1, a1[0]);
                        } else if constexpr (B == 2) {
                            float2 x = *(const float2*)&xt[r_lo * 2];
                            a0[0] = fmaf(x.x, f0, a0[0]); a0[1] = fmaf(x.y, f0, a0[1]);
                            a1[0] = fmaf(x.x, f1, a1[0]); a1[1] = fmaf(x.y, f1, a1[1]);
                        } else if constexpr (B == 3) {
                            float x0 = xt[r_lo*3+0], x1 = xt[r_lo*3+1], x2 = xt[r_lo*3+2];
                            a0[0] = fmaf(x0, f0, a0[0]); a0[1] = fmaf(x1, f0, a0[1]); a0[2] = fmaf(x2, f0, a0[2]);
                            a1[0] = fmaf(x0, f1, a1[0]); a1[1] = fmaf(x1, f1, a1[1]); a1[2] = fmaf(x2, f1, a1[2]);
                        } else {
                            float4 x = *(const float4*)&xt[r_lo * 4];
                            a0[0] = fmaf(x.x, f0, a0[0]); a0[1] = fmaf(x.y, f0, a0[1]); a0[2] = fmaf(x.z, f0, a0[2]); a0[3] = fmaf(x.w, f0, a0[3]);
                            a1[0] = fmaf(x.x, f1, a1[0]); a1[1] = fmaf(x.y, f1, a1[1]); a1[2] = fmaf(x.z, f1, a1[2]); a1[3] = fmaf(x.w, f1, a1[3]);
                        }
                    }
                };
                do_cols(W0,                         __funnelshift_l(W1, W0,  8), r_off + 0, r_off + 2);
                do_cols(__funnelshift_l(W1, W0, 2), __funnelshift_l(W1, W0, 10), r_off + 1, r_off + 3);
                do_cols(__funnelshift_l(W1, W0, 4), __funnelshift_l(W1, W0, 12), r_off + 8, r_off + 10);
                do_cols(__funnelshift_l(W1, W0, 6), __funnelshift_l(W1, W0, 14), r_off + 9, r_off + 11);
            }
        }
        __syncthreads();
    }
    if (tn_i < tn) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            a0[b] += __shfl_xor_sync(0xffffffffu, a0[b], 1);
            a1[b] += __shfl_xor_sync(0xffffffffu, a1[b], 1);
        }
    }
    if (part == 1 && (sub & 1) == 0 && tn_i < tn) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            xs[(b * 8 + t_in_blk) * 16 + (sub >> 1)] = a0[b];
            xs[(b * 8 + t_in_blk) * 16 + (sub >> 1) + 8] = a1[b];
        }
    }
    __syncthreads();
    if (part == 0 && (sub & 1) == 0 && tn_i < tn) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float s0 = a0[b] + xs[(b * 8 + t_in_blk) * 16 + (sub >> 1)];
            float s1 = a1[b] + xs[(b * 8 + t_in_blk) * 16 + (sub >> 1) + 8];
            float off = KOFF_C * sx_sh[b];
            float* dst = &t[(size_t)b * nslots * n + (size_t)slot * n + zsel * tzoff + tn_i * 16 + (sub >> 1)];
            dst[0] = fmaf(KINV_F, s0, off);
            dst[8] = fmaf(KINV_F, s1, off);
        }
    }
}

template <int B>
__global__ void __launch_bounds__(256, 6) gemv_batch_m16_c4(
    const uint64_t* __restrict__ descs, const uint64_t* __restrict__ descs2,
    const uint32_t* __restrict__ x16b, const float* __restrict__ qsb, const float* __restrict__ sxb,
    float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, int nslots, uint32_t CM) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1, zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + slot * 4;
    const int js = slot * nsel + zsel;
    const uint32_t* x16g = x16b + (size_t)js * k * (B / 2);
    const float* qsg = qsb + (size_t)js * (k / 128) * B;

    __shared__ __align__(16) uint32_t xs[32 * 16 * (B / 2)];
    __shared__ float qs[24 * B];
    // For B=4, keep outer float accumulators in tiny per-thread smem (or regs for B=2) to guarantee 0 spills!
    __shared__ float acc_s[256 * 2 * B];
    if (threadIdx.x < (k / 128) * B) qs[threadIdx.x] = qsg[threadIdx.x];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        acc_s[threadIdx.x * 2 * B + b] = 0.f;
        acc_s[threadIdx.x * 2 * B + B + b] = 0.f;
    }
    __syncthreads();

    const int part = threadIdx.x >> 7;
    const int local_tid = threadIdx.x & 127;
    const int t_in_blk = local_tid >> 4, sub = local_tid & 15;
    const int tn_i = blockIdx.x * 8 + t_in_blk;
    const int r_off = (sub & 1) ? 4 : 0;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    const size_t stride = (size_t)tn * 16;

    for (int tbase = 0; tbase < tk; tbase += 32) {
        const int nloc = min(32, tk - tbase);
        for (int e = threadIdx.x; e < nloc * 16 * (B / 2); e += 256) xs[e] = x16g[(size_t)tbase * 16 * (B / 2) + e];
        __syncthreads();
        if (tn_i < tn) {
            for (int blk = part; blk * 8 < nloc; blk += 2) {
                const int t0 = blk * 8, t1 = min(t0 + 8, nloc);
                const uint32_t* p0 = tb + ((size_t)(tbase + t0) * tn + tn_i) * 16 + sub;
                const uint32_t* p1 = tb + ((size_t)(tbase + t0) * tn + tn_i) * 16 + ((sub + 1) & 15);
                int ia0[B], ia1[B];
#pragma unroll
                for (int b = 0; b < B; ++b) { ia0[b] = 0; ia1[b] = 0; }
#pragma unroll 1
                for (int ti = t0; ti < t1; ++ti, p0 += stride, p1 += stride) {
                    const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
                    const uint32_t* xt = &xs[ti * 16 * (B / 2)];
                    auto do_cols = [&](uint32_t U0, uint32_t U1, int r_hi, int r_lo) {
                        {
                            int b0 = (int)dp4a_uu(mul16_h1(U0, CM), 0x01010101u, (uint32_t)(-510));
                            int b1 = (int)dp4a_uu(mul16_h1(U1, CM), 0x01010101u, (uint32_t)(-510));
                            if constexpr (B == 2) {
                                uint32_t xp = xt[r_hi];
                                ia0[0] = xmad_s16_h0(xp, b0, ia0[0]); ia0[1] = xmad_s16_h1(xp, b0, ia0[1]);
                                ia1[0] = xmad_s16_h0(xp, b1, ia1[0]); ia1[1] = xmad_s16_h1(xp, b1, ia1[1]);
                            } else {
                                uint2 xp = *(const uint2*)&xt[r_hi * 2];
                                ia0[0] = xmad_s16_h0(xp.x, b0, ia0[0]); ia0[1] = xmad_s16_h1(xp.x, b0, ia0[1]);
                                ia0[2] = xmad_s16_h0(xp.y, b0, ia0[2]); ia0[3] = xmad_s16_h1(xp.y, b0, ia0[3]);
                                ia1[0] = xmad_s16_h0(xp.x, b1, ia1[0]); ia1[1] = xmad_s16_h1(xp.x, b1, ia1[1]);
                                ia1[2] = xmad_s16_h0(xp.y, b1, ia1[2]); ia1[3] = xmad_s16_h1(xp.y, b1, ia1[3]);
                            }
                        }
                        {
                            int b0 = (int)dp4a_uu(mul16_h0(U0, CM), 0x01010101u, (uint32_t)(-510));
                            int b1 = (int)dp4a_uu(mul16_h0(U1, CM), 0x01010101u, (uint32_t)(-510));
                            if constexpr (B == 2) {
                                uint32_t xp = xt[r_lo];
                                ia0[0] = xmad_s16_h0(xp, b0, ia0[0]); ia0[1] = xmad_s16_h1(xp, b0, ia0[1]);
                                ia1[0] = xmad_s16_h0(xp, b1, ia1[0]); ia1[1] = xmad_s16_h1(xp, b1, ia1[1]);
                            } else {
                                uint2 xp = *(const uint2*)&xt[r_lo * 2];
                                ia0[0] = xmad_s16_h0(xp.x, b0, ia0[0]); ia0[1] = xmad_s16_h1(xp.x, b0, ia0[1]);
                                ia0[2] = xmad_s16_h0(xp.y, b0, ia0[2]); ia0[3] = xmad_s16_h1(xp.y, b0, ia0[3]);
                                ia1[0] = xmad_s16_h0(xp.x, b1, ia1[0]); ia1[1] = xmad_s16_h1(xp.x, b1, ia1[1]);
                                ia1[2] = xmad_s16_h0(xp.y, b1, ia1[2]); ia1[3] = xmad_s16_h1(xp.y, b1, ia1[3]);
                            }
                        }
                    };
                    do_cols(W0,                         __funnelshift_l(W1, W0,  8), r_off + 0, r_off + 2);
                    do_cols(__funnelshift_l(W1, W0, 2), __funnelshift_l(W1, W0, 10), r_off + 1, r_off + 3);
                    do_cols(__funnelshift_l(W1, W0, 4), __funnelshift_l(W1, W0, 12), r_off + 8, r_off + 10);
                    do_cols(__funnelshift_l(W1, W0, 6), __funnelshift_l(W1, W0, 14), r_off + 9, r_off + 11);
                }
                const int gblk = (tbase >> 3) + blk;
#pragma unroll
                for (int b = 0; b < B; ++b) {
                    const float sc = qs[gblk * B + b];
                    acc_s[threadIdx.x * 2 * B + b]     = fmaf(sc, (float)ia0[b], acc_s[threadIdx.x * 2 * B + b]);
                    acc_s[threadIdx.x * 2 * B + B + b] = fmaf(sc, (float)ia1[b], acc_s[threadIdx.x * 2 * B + B + b]);
                }
            }
        }
        __syncthreads();
    }
    float a0[B], a1[B];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        a0[b] = acc_s[threadIdx.x * 2 * B + b];
        a1[b] = acc_s[threadIdx.x * 2 * B + B + b];
    }
    if (tn_i < tn) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            a0[b] += __shfl_xor_sync(0xffffffffu, a0[b], 1);
            a1[b] += __shfl_xor_sync(0xffffffffu, a1[b], 1);
        }
    }
    float* red_out = (float*)xs;
    if (part == 1 && (sub & 1) == 0 && tn_i < tn) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            red_out[(b * 8 + t_in_blk) * 16 + (sub >> 1)] = a0[b];
            red_out[(b * 8 + t_in_blk) * 16 + (sub >> 1) + 8] = a1[b];
        }
    }
    __syncthreads();
    if (part == 0 && (sub & 1) == 0 && tn_i < tn) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float s0 = a0[b] + red_out[(b * 8 + t_in_blk) * 16 + (sub >> 1)];
            float s1 = a1[b] + red_out[(b * 8 + t_in_blk) * 16 + (sub >> 1) + 8];
            float off = KOFF_C * sxb[js * B + b];
            float* dst = &t[(size_t)b * nslots * n + (size_t)slot * n + zsel * tzoff + tn_i * 16 + (sub >> 1)];
            dst[0] = fmaf(KINV_F, s0, off);
            dst[8] = fmaf(KINV_F, s1, off);
        }
    }
}

} // namespace r2

// ---------------- host helpers ----------------
static uint32_t rs = 12345u;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; float v = e ? ldexpf(1.f + m / 1024.f, e - 15) : ldexpf(m / 1024.f, -14); return (h & 0x8000) ? -v : v; }

struct KH { int ka, hk; };
static int nwords(KH k) { return 8 * k.ka + 4 * k.hk; }

static void rotate_ring_left(const uint32_t* src, uint32_t* dst, int nw, int shift_bits) {
    const int R = nw * 32;
    shift_bits = ((shift_bits % R) + R) % R;
    const int ws = shift_bits >> 5, bs = shift_bits & 31;
    for (int w = 0; w < nw; ++w) {
        int i0 = (w + ws) % nw, i1 = (i0 + 1) % nw;
        dst[w] = bs == 0 ? src[i0] : ((src[i0] << bs) | (src[i1] >> (32 - bs)));
    }
}

static void encode_strided_k2_m32(const uint32_t* raw_ring, uint32_t* out16) {
    uint32_t rot[16];
    rotate_ring_left(raw_ring, rot, 16, 512 - 14);
    uint8_t c[256];
    for (int k = 0; k < 256; ++k) c[k] = (uint8_t)((rot[k >> 4] >> (30 - 2 * (k & 15))) & 3u);
    for (int l = 0; l < 8; ++l) {
        uint32_t W0 = 0, W1 = 0;
        for (int ch = 0; ch < 4; ++ch) {
            uint32_t b0 = (uint32_t)c[(32 * l + ch +  0) & 255]
                        | ((uint32_t)c[(32 * l + ch +  8) & 255] << 2)
                        | ((uint32_t)c[(32 * l + ch + 16) & 255] << 4)
                        | ((uint32_t)c[(32 * l + ch + 24) & 255] << 6);
            uint32_t b1 = (uint32_t)c[(32 * l + ch +  4) & 255]
                        | ((uint32_t)c[(32 * l + ch + 12) & 255] << 2)
                        | ((uint32_t)c[(32 * l + ch + 20) & 255] << 4)
                        | ((uint32_t)c[(32 * l + ch + 28) & 255] << 6);
            W0 |= b0 << (8 * ch);
            W1 |= b1 << (8 * ch);
        }
        out16[l * 2 + 0] = W0; out16[l * 2 + 1] = W1;
    }
}

static void ref_gemv(const uint32_t* tr, KH kh, const float* x, int tk, int tn, double* y) {
    const int nw = nwords(kh), R = nw * 32;
    const double KINV = h2f(0x1eee), KC = h2f(0xc931);
    for (int c = 0; c < tn * 16; ++c) y[c] = 0;
    for (int a = 0; a < tk; ++a)
        for (int b = 0; b < tn; ++b) {
            const uint32_t* ring = tr + ((size_t)a * tn + b) * nw;
            for (int i = 0; i < 256; ++i) {
                const int s = kh.hk ? (i + 1) * kh.ka + ((i + 1) >> 1) : (i + 1) * kh.ka;
                const int b0 = s - 16 + (s < 16 ? R : 0);
                const int i0 = b0 >> 5, s0 = 48 - (b0 & 31), i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
                const uint64_t v = ((uint64_t)ring[i0] << 32) | ring[i1];
                uint32_t p = (uint32_t)((v >> s0) & 0xffff) * 0x83DCD12Du;
                const int bs = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
                const int tt = i >> 3, j = i & 7;
                const int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
                const int cc = (tt >> 2) + ((j & 4) ? 8 : 0);
                y[b * 16 + cc] += (double)x[a * 16 + r] * ((1024.0 + bs) * KINV + KC);
            }
        }
}

struct Timing { float min_us, med_us; };
// Interleaved old/new timing (alternating base/cand every iteration -> zero clock/thermal drift bias!)
template <typename FB, typename FC>
static void timeit_interleaved(FB&& fbase, FC&& fcand, Timing& tb, Timing& tc, int reps = 31) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 4; ++i) { fbase(); fcand(); }
    CK(cudaDeviceSynchronize());
    std::vector<float> vb, vc; vb.reserve(reps); vc.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        float ms;
        if ((i & 1) == 0) {
            CK(cudaEventRecord(a)); fbase(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b)); vb.push_back(ms * 1e3f);
            CK(cudaEventRecord(a)); fcand(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b)); vc.push_back(ms * 1e3f);
        } else {
            CK(cudaEventRecord(a)); fcand(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b)); vc.push_back(ms * 1e3f);
            CK(cudaEventRecord(a)); fbase(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms, a, b)); vb.push_back(ms * 1e3f);
        }
    }
    std::sort(vb.begin(), vb.end()); std::sort(vc.begin(), vc.end());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    tb = {vb[0], vb[vb.size() / 2]};
    tc = {vc[0], vc[vc.size() / 2]};
}

int main(int argc, char** argv) {
    const int mask = (argc > 1) ? atoi(argv[1]) : 7;
    const int pwmask = (argc > 2) ? atoi(argv[2]) : 0;
    const int nslots = 8;
    const uint32_t CM = 0x83DCD12Du;
    struct Stage { const char* name; int k, nout, nsel; };
    const Stage st[2] = {{"GU  2304->2x1152", 2304, 1152, 2}, {"down 1152->2304 ", 1152, 2304, 1}};

    std::vector<int16_t> h_lut16(65536);
    std::vector<int8_t> h_lut8(65536);
    for (uint32_t h = 0; h < 65536; ++h) {
        uint32_t p = h * CM;
        int bs = (int)((p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24));
        h_lut16[h] = (int16_t)(bs - 510);
        int q = ((bs + 2) >> 2) - 128;
        h_lut8[h] = (int8_t)std::max(-128, std::min(127, q));
    }
    int16_t* d_lut16; int8_t* d_lut8;
    CK(cudaMalloc(&d_lut16, 65536 * 2)); CK(cudaMemcpy(d_lut16, h_lut16.data(), 65536 * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_lut8, 65536));      CK(cudaMemcpy(d_lut8, h_lut8.data(), 65536, cudaMemcpyHostToDevice));

    // =========================================================================
    // BENCHMARK 1: PURE K=2.0 HEAD-TO-HEAD (INTERLEAVED vs SHIPPED sgemv_run!)
    // =========================================================================
    if (mask & 1) {
    printf("==========================================================================================================\n");
    printf("[BENCH 1] Pure K=2.0 Interleaved Old/New vs Shipped sgemv_run (8 slots, GU+down, 8 tiles/blk, min-of-31)\n");
    printf("==========================================================================================================\n");
    struct Row { std::string name; double gu_b, dn_b, gu_c, dn_c, gu_cm, dn_cm, gu_err, dn_err; };
    std::vector<Row> b1_rows;
    double b1_nw_tot = 0;
    {
        for (int si = 0; si < 2; ++si) {
            const Stage S = st[si];
            const int tk = S.k / 16, tn = S.nout / 16, njobs = nslots * S.nsel, n = S.nout * S.nsel;
            const int tzoff = S.nsel == 2 ? S.nout : 0;
            b1_nw_tot += (double)njobs * S.k * S.nout;
            std::vector<std::vector<uint32_t>> tr_raw(njobs), tr_rot(njobs), tr_str(njobs);
            std::vector<uint64_t> d1_raw(nslots*4,0), d2_raw(nslots*4,0), d1_rot(nslots*4,0), d2_rot(nslots*4,0), d1_str(nslots*4,0), d2_str(nslots*4,0);
            std::vector<uint32_t*> dr_raw(njobs), dr_rot(njobs), dr_str(njobs);
            for (int j = 0; j < njobs; ++j) {
                size_t sz = (size_t)tk * tn * 16;
                tr_raw[j].resize(sz); tr_rot[j].resize(sz); tr_str[j].resize(sz);
                for (auto& v : tr_raw[j]) v = rnd();
                for (size_t ti = 0; ti < (size_t)tk * tn; ++ti) {
                    rotate_ring_left(&tr_raw[j][ti * 16], &tr_rot[j][ti * 16], 16, 512 - 14);
                    encode_strided_k2_m32(&tr_raw[j][ti * 16], &tr_str[j][ti * 16]);
                }
                CK(cudaMalloc(&dr_raw[j], sz*4)); CK(cudaMemcpy(dr_raw[j], tr_raw[j].data(), sz*4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dr_rot[j], sz*4)); CK(cudaMemcpy(dr_rot[j], tr_rot[j].data(), sz*4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dr_str[j], sz*4)); CK(cudaMemcpy(dr_str[j], tr_str[j].data(), sz*4, cudaMemcpyHostToDevice));
                int s = j / S.nsel, z = j % S.nsel;
                uint64_t meta = 2ULL;
                ((z ? d2_raw : d1_raw).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_raw[j]; ((z ? d2_raw : d1_raw).data() + s*4)[3] = meta;
                ((z ? d2_rot : d1_rot).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_rot[j]; ((z ? d2_rot : d1_rot).data() + s*4)[3] = meta;
                ((z ? d2_str : d1_str).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_str[j]; ((z ? d2_str : d1_str).data() + s*4)[3] = meta;
            }
            std::vector<float> x((size_t)njobs * S.k);
            for (auto& v : x) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
            for (int j = 0; j < njobs; ++j) x[(size_t)j * S.k + (rnd() % S.k)] *= 6.f;
            std::vector<float> xg((size_t)nslots * S.k), xu((size_t)nslots * S.k);
            for (int s = 0; s < nslots; ++s)
                for (int e = 0; e < S.k; ++e) {
                    xg[(size_t)s * S.k + e] = x[(size_t)(s * S.nsel) * S.k + e];
                    if (S.nsel == 2) xu[(size_t)s * S.k + e] = x[(size_t)(s * 2 + 1) * S.k + e];
                }
            std::vector<double> ref((size_t)nslots * n);
            for (int j = 0; j < njobs; ++j) {
                int s = j / S.nsel, z = j % S.nsel;
                ref_gemv(tr_raw[j].data(), {2, 0}, &x[(size_t)j * S.k], tk, tn, &ref[(size_t)s * n + z * S.nout]);
            }
            double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));

            uint64_t *dd1_raw, *dd2_raw=nullptr, *dd1_rot, *dd2_rot=nullptr, *dd1_str, *dd2_str=nullptr;
            int* dids; float *dxg, *dxu, *dt;
            CK(cudaMalloc(&dd1_raw, d1_raw.size()*8)); CK(cudaMemcpy(dd1_raw, d1_raw.data(), d1_raw.size()*8, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dd1_rot, d1_rot.size()*8)); CK(cudaMemcpy(dd1_rot, d1_rot.data(), d1_rot.size()*8, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dd1_str, d1_str.size()*8)); CK(cudaMemcpy(dd1_str, d1_str.data(), d1_str.size()*8, cudaMemcpyHostToDevice));
            if (S.nsel == 2) {
                CK(cudaMalloc(&dd2_raw, d2_raw.size()*8)); CK(cudaMemcpy(dd2_raw, d2_raw.data(), d2_raw.size()*8, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dd2_rot, d2_rot.size()*8)); CK(cudaMemcpy(dd2_rot, d2_rot.data(), d2_rot.size()*8, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dd2_str, d2_str.size()*8)); CK(cudaMemcpy(dd2_str, d2_str.data(), d2_str.size()*8, cudaMemcpyHostToDevice));
            }
            std::vector<int> ids(nslots); for (int i = 0; i < nslots; ++i) ids[i] = i;
            CK(cudaMalloc(&dids, nslots*4)); CK(cudaMemcpy(dids, ids.data(), nslots*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dxg, xg.size()*4)); CK(cudaMemcpy(dxg, xg.data(), xg.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dxu, xu.size()*4)); CK(cudaMemcpy(dxu, xu.data(), xu.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dt, (size_t)nslots * n * 4));
            std::vector<float> out((size_t)nslots * n);
            auto check_err = [&]() {
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                CK(cudaMemcpy(out.data(), dt, out.size()*4, cudaMemcpyDeviceToHost));
                double e = 0; for (size_t i = 0; i < out.size(); ++i) e = std::max(e, fabs((double)out[i] - ref[i]));
                return e / ymax;
            };
            const dim3 g((tn + 7) / 8, nslots, S.nsel);
            auto fship = [&]() {
                inkwell_exl3_gemv_moe_s<<<g, 256>>>(dd1_raw, dd2_raw, dids, dxg, dxu, S.k, dt, n, tzoff, tk, tn, S.k, n, 1, nullptr, 0, pwmask);
            };
            int idx = 0;
            auto run_one = [&](const char* name, auto fn) {
                CK(cudaMemset(dt, 0, (size_t)nslots * n * 4));
                fn();
                double er = check_err();
                Timing tb, tc;
                timeit_interleaved(fship, fn, tb, tc);
                if (si == 0) b1_rows.push_back({name, tb.min_us, 0, tc.min_us, 0, tc.med_us, 0, er, 0});
                else { b1_rows[idx].dn_b = tb.min_us; b1_rows[idx].dn_c = tc.min_us; b1_rows[idx].dn_cm = tc.med_us; b1_rows[idx].dn_err = er; }
                idx++;
            };

            run_one("Shipped sgemv_run (nky=1, exact)", fship);
            run_one("PW16-port C0c (ship kern, exact)", [&]() { inkwell_exl3_gemv_moe_s<<<g, 256>>>(dd1_raw, dd2_raw, dids, dxg, dxu, S.k, dt, n, tzoff, tk, tn, S.k, n, 1, nullptr, 0, 16); });
            run_one("M=8-Opt C0c (zero-ctr I2F, exact)", [&]() { r2::gemv_m8_k2<0><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=8-Opt C3b (folded FP32 per-tile)", [&]() { r2::gemv_m8_k2<1><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=8-Opt C4 (IDP4A+XMAD.S16, blk=128)", [&]() { r2::gemv_m8_k2<2><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=8-Opt C4 (IDP4A+XMAD.S16, blk=16)", [&]() { r2::gemv_m8_k2<3><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=8-Opt C6 (dp4a-8, blk=16)", [&]() { r2::gemv_m8_k2<4><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C0c (exact I2F)", [&]() { r2::gemv_m16<0, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C1c (exact Magic FADD)", [&]() { r2::gemv_m16<1, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C3b (folded FP32 tile)", [&]() { r2::gemv_m16<2, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C3a (folded FP32 glob)", [&]() { r2::gemv_m16<3, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C4 (XMAD.S16, blk=128)", [&]() { r2::gemv_m16<4, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C4 (XMAD.S16, blk=16)", [&]() { r2::gemv_m16<5, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C6 (dp4a-8, blk=16)", [&]() { r2::gemv_m16<6, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired LEA+2*DP2A.LO (blk=128)", [&]() { r2::gemv_m16<7, false><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=32-Strided C0c (SHFL+ISCADD, exact)", [&]() { r2::gemv_strided_m32_k2<0><<<g, 256>>>(dd1_str, dd2_str, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=32-Strided C4 (SHFL+ISCADD+S16)", [&]() { r2::gemv_strided_m32_k2<1><<<g, 256>>>(dd1_str, dd2_str, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=32-Strided C6 (SHFL+ISCADD+dp4a8)", [&]() { r2::gemv_strided_m32_k2<2><<<g, 256>>>(dd1_str, dd2_str, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-LUT128K-Exact (128KB LDG LUT)", [&]() { r2::gemv_lut_m16_k2<int16_t, 1><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, d_lut16, dt, tzoff, tk, tn, S.k, n); });
            run_one("M=16-LUT64K-BS8 (64KB LDG LUT)", [&]() { r2::gemv_lut_m16_k2<int8_t, 4><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, d_lut8, dt, tzoff, tk, tn, S.k, n); });

            for (int j = 0; j < njobs; ++j) { cudaFree(dr_raw[j]); cudaFree(dr_rot[j]); cudaFree(dr_str[j]); }
            cudaFree(dd1_raw); cudaFree(dd1_rot); cudaFree(dd1_str);
            if (dd2_raw) { cudaFree(dd2_raw); cudaFree(dd2_rot); cudaFree(dd2_str); }
            cudaFree(dids); cudaFree(dxg); cudaFree(dxu); cudaFree(dt);
        }
        printf("%-38s %7s %7s %7s %7s %7s %7s %8s %9s\n",
               "Design (K=2.0, interleaved)", "Ship_us", "Cand_us", "CandMed", "ShipGw", "CandGw", "Spdup", "max_rel", "GU/DN");
        for (const auto& r : b1_rows) {
            double bmin = r.gu_b + r.dn_b, cmin = r.gu_c + r.dn_c, cmed = r.gu_cm + r.dn_cm;
            double bgw = b1_nw_tot / (bmin * 1e3), cgw = b1_nw_tot / (cmin * 1e3);
            printf("%-38s %7.1f %7.1f %7.1f %7.1f %7.1f %6.2fx %8.2e %.0f/%.0f\n",
                   r.name.c_str(), bmin, cmin, cmed, bgw, cgw, bmin / cmin, std::max(r.gu_err, r.dn_err), r.gu_c, r.dn_c);
        }
    }
    } // end mask & 1

    // =========================================================================
    // BENCHMARK 2: DIRECTION 3 — MULTI-TOKEN BATCHING (B = 1, 2, 3, 4) ON K=2.0
    // =========================================================================
    if (mask & 2) {
    printf("\n==========================================================================================================\n");
    printf("[BENCH 2] Multi-Token Batching (B = 1, 2, 3, 4) on K=2.0 (GU 2304->2x1152, 8 slots, interleaved min-of-31)\n");
    printf("==========================================================================================================\n");
    {
        const Stage S = st[0];
        const int tk = S.k / 16, tn = S.nout / 16, njobs = nslots * S.nsel, n = S.nout * S.nsel;
        const int tzoff = S.nout;
        const double nw = (double)njobs * S.k * S.nout;
        std::vector<std::vector<uint32_t>> tr_raw(njobs), tr_rot(njobs);
        std::vector<uint64_t> d1_raw(nslots*4, 0), d2_raw(nslots*4, 0), d1_rot(nslots*4, 0), d2_rot(nslots*4, 0);
        std::vector<uint32_t*> dr_raw(njobs), dr_rot(njobs);
        for (int j = 0; j < njobs; ++j) {
            size_t sz = (size_t)tk * tn * 16;
            tr_raw[j].resize(sz); tr_rot[j].resize(sz);
            for (auto& v : tr_raw[j]) v = rnd();
            for (size_t ti = 0; ti < (size_t)tk * tn; ++ti)
                rotate_ring_left(&tr_raw[j][ti * 16], &tr_rot[j][ti * 16], 16, 512 - 14);
            CK(cudaMalloc(&dr_raw[j], sz*4)); CK(cudaMemcpy(dr_raw[j], tr_raw[j].data(), sz*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dr_rot[j], sz*4)); CK(cudaMemcpy(dr_rot[j], tr_rot[j].data(), sz*4, cudaMemcpyHostToDevice));
            int s = j / 2, z = j % 2;
            ((z ? d2_raw : d1_raw).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_raw[j]; ((z ? d2_raw : d1_raw).data() + s*4)[3] = 2ULL;
            ((z ? d2_rot : d1_rot).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_rot[j]; ((z ? d2_rot : d1_rot).data() + s*4)[3] = 2ULL;
        }
        uint64_t *dd1_raw, *dd2_raw, *dd1, *dd2; int* dids;
        CK(cudaMalloc(&dd1_raw, d1_raw.size()*8)); CK(cudaMemcpy(dd1_raw, d1_raw.data(), d1_raw.size()*8, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dd2_raw, d2_raw.size()*8)); CK(cudaMemcpy(dd2_raw, d2_raw.data(), d2_raw.size()*8, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dd1, d1_rot.size()*8)); CK(cudaMemcpy(dd1, d1_rot.data(), d1_rot.size()*8, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dd2, d2_rot.size()*8)); CK(cudaMemcpy(dd2, d2_rot.data(), d2_rot.size()*8, cudaMemcpyHostToDevice));
        std::vector<int> ids(nslots); for (int i = 0; i < nslots; ++i) ids[i] = i;
        CK(cudaMalloc(&dids, nslots*4)); CK(cudaMemcpy(dids, ids.data(), nslots*4, cudaMemcpyHostToDevice));
        std::vector<float> dummy_x((size_t)nslots * S.k, 0.5f);
        float *dx_d1, *dx_d2, *dt_d;
        CK(cudaMalloc(&dx_d1, dummy_x.size()*4)); CK(cudaMemcpy(dx_d1, dummy_x.data(), dummy_x.size()*4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx_d2, dummy_x.size()*4)); CK(cudaMemcpy(dx_d2, dummy_x.data(), dummy_x.size()*4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dt_d, (size_t)nslots * n * 4));
        const dim3 g((tn + 7) / 8, nslots, S.nsel);
        auto fship_b1 = [&]() {
            inkwell_exl3_gemv_moe_s<<<g, 256>>>(dd1_raw, dd2_raw, dids, dx_d1, dx_d2, S.k, dt_d, n, tzoff, tk, tn, S.k, n, 1, nullptr, 0, pwmask);
        };
        printf("%-26s %2s %8s %8s %8s %9s %9s %7s %9s\n",
               "Kernel", "B", "Ship1_us", "Cand_us", "CandMed", "Rd_Gw/s", "Eff_GMAC", "vsShip", "max_rel");

        auto test_batch_c0c = [&](auto Btag) {
            constexpr int B = decltype(Btag)::value;
            std::vector<float> xgb((size_t)njobs * S.k * B);
            for (auto& v : xgb) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
            std::vector<double> ref((size_t)B * nslots * n);
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < njobs; ++j) {
                    std::vector<float> xb(S.k);
                    for (int e = 0; e < S.k; ++e) xb[e] = xgb[(size_t)j * S.k * B + e * B + b];
                    int s = j / 2, z = j % 2;
                    ref_gemv(tr_raw[j].data(), {2, 0}, xb.data(), tk, tn, &ref[(size_t)b * nslots * n + s * n + z * S.nout]);
                }
            }
            double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));
            float *dxgb, *dt;
            CK(cudaMalloc(&dxgb, xgb.size()*4)); CK(cudaMemcpy(dxgb, xgb.data(), xgb.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dt, ref.size()*4));
            auto fn = [&]() { r2::gemv_batch_m16_c0c<B><<<g, 256>>>(dd1, dd2, dxgb, dt, tzoff, tk, tn, S.k, n, nslots, CM); };
            fn(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            std::vector<float> out(ref.size());
            CK(cudaMemcpy(out.data(), dt, out.size()*4, cudaMemcpyDeviceToHost));
            double em = 0; for (size_t i = 0; i < out.size(); ++i) em = std::max(em, fabs((double)out[i] - ref[i]));
            Timing tb, tc; timeit_interleaved(fship_b1, fn, tb, tc);
            double rd = nw / (tc.min_us * 1e3), b_gw = nw / (tb.min_us * 1e3);
            printf("%-26s %2d %8.1f %8.1f %8.1f %9.1f %9.1f %6.2fx %9.2e\n",
                   "M=16-Batch C0c (exact)", B, tb.min_us, tc.min_us, tc.med_us, rd, rd * B, (rd * B) / b_gw, em / ymax);
            cudaFree(dxgb); cudaFree(dt);
        };
        auto test_batch_c4 = [&](auto Btag) {
            constexpr int B = decltype(Btag)::value;
            std::vector<float> xgb((size_t)njobs * S.k * B);
            for (auto& v : xgb) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
            std::vector<uint32_t> x16b((size_t)njobs * S.k * (B / 2), 0);
            std::vector<float> qsb((size_t)njobs * (S.k / 128) * B);
            std::vector<float> sxb((size_t)njobs * B, 0.f);
            for (int j = 0; j < njobs; ++j) {
                for (int blk = 0; blk < S.k / 128; ++blk) {
                    for (int b = 0; b < B; ++b) {
                        float am = 1e-30f;
                        for (int i = 0; i < 128; ++i) am = fmaxf(am, fabsf(xgb[(size_t)j*S.k*B + (blk*128+i)*B + b]));
                        float sc = am / 32767.f; qsb[(size_t)j*(S.k/128)*B + blk*B + b] = sc;
                        for (int i = 0; i < 128; ++i) {
                            int q = (int)lrintf(xgb[(size_t)j*S.k*B + (blk*128+i)*B + b] / sc);
                            sxb[j*B + b] += (float)q * sc;
                            uint32_t& dst = x16b[(size_t)j*S.k*(B/2) + (blk*128+i)*(B/2) + (b/2)];
                            if ((b & 1) == 0) dst |= (uint32_t)(q & 0xFFFF);
                            else dst |= (uint32_t)(q & 0xFFFF) << 16;
                        }
                    }
                }
            }
            std::vector<double> ref((size_t)B * nslots * n);
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < njobs; ++j) {
                    std::vector<float> xb(S.k);
                    for (int e = 0; e < S.k; ++e) xb[e] = xgb[(size_t)j * S.k * B + e * B + b];
                    int s = j / 2, z = j % 2;
                    ref_gemv(tr_raw[j].data(), {2, 0}, xb.data(), tk, tn, &ref[(size_t)b * nslots * n + s * n + z * S.nout]);
                }
            }
            double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));
            uint32_t* dx16; float *dqs, *dsx, *dt;
            CK(cudaMalloc(&dx16, x16b.size()*4)); CK(cudaMemcpy(dx16, x16b.data(), x16b.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dqs, qsb.size()*4));   CK(cudaMemcpy(dqs, qsb.data(), qsb.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dsx, sxb.size()*4));   CK(cudaMemcpy(dsx, sxb.data(), sxb.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dt, ref.size()*4));
            auto fn = [&]() { r2::gemv_batch_m16_c4<B><<<g, 256>>>(dd1, dd2, dx16, dqs, dsx, dt, tzoff, tk, tn, S.k, n, nslots, CM); };
            fn(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            std::vector<float> out(ref.size());
            CK(cudaMemcpy(out.data(), dt, out.size()*4, cudaMemcpyDeviceToHost));
            double em = 0; for (size_t i = 0; i < out.size(); ++i) em = std::max(em, fabs((double)out[i] - ref[i]));
            Timing tb, tc; timeit_interleaved(fship_b1, fn, tb, tc);
            double rd = nw / (tc.min_us * 1e3), b_gw = nw / (tb.min_us * 1e3);
            printf("%-26s %2d %8.1f %8.1f %8.1f %9.1f %9.1f %6.2fx %9.2e\n",
                   "M=16-Batch C4 (int16 x)", B, tb.min_us, tc.min_us, tc.med_us, rd, rd * B, (rd * B) / b_gw, em / ymax);
            cudaFree(dx16); cudaFree(dqs); cudaFree(dsx); cudaFree(dt);
        };
        test_batch_c0c(std::integral_constant<int, 1>{});
        test_batch_c0c(std::integral_constant<int, 2>{});
        test_batch_c0c(std::integral_constant<int, 3>{});
        test_batch_c0c(std::integral_constant<int, 4>{});
        test_batch_c4(std::integral_constant<int, 2>{});
        test_batch_c4(std::integral_constant<int, 4>{});
        for (int j = 0; j < njobs; ++j) { cudaFree(dr_raw[j]); cudaFree(dr_rot[j]); }
        cudaFree(dd1_raw); cudaFree(dd2_raw); cudaFree(dd1); cudaFree(dd2);
        cudaFree(dids); cudaFree(dx_d1); cudaFree(dx_d2); cudaFree(dt_d);
    }
    } // end mask & 2

    // =========================================================================
    // BENCHMARK 3: FULL REAL SAGE ALL-7-K MIX (INTERLEAVED vs SHIPPED!)
    // =========================================================================
    if (mask & 4) {
    printf("\n==========================================================================================================\n");
    printf("[BENCH 3] Full SAGE All-7-K Mix (1.5..5.0, exact sgemv_bench.cu mix, GU+down, interleaved min-of-31)\n");
    printf("==========================================================================================================\n");
    {
        struct Row { std::string name; double gu_b, dn_b, gu_c, dn_c, gu_cm, dn_cm, gu_err, dn_err; };
        const KH khs[7] = {{1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {4, 0}, {5, 0}};
        const int gu_w[7] = {989, 1274, 450, 227, 64, 29, 0}, dn_w[7] = {542, 1151, 570, 450, 227, 93, 0};
        std::vector<Row> b3_rows;
        double nw_tot = 0;
        for (int si = 0; si < 2; ++si) {
            const Stage S = st[si];
            const int tk = S.k / 16, tn = S.nout / 16, njobs = nslots * S.nsel, n = S.nout * S.nsel;
            const int tzoff = S.nsel == 2 ? S.nout : 0;
            nw_tot += (double)njobs * S.k * S.nout;
            std::vector<KH> jk(njobs);
            const int* wts = si == 0 ? gu_w : dn_w;
            int wsum = 0; for (int i = 0; i < 7; ++i) wsum += wts[i];
            for (int j = 0; j < njobs; ++j) {
                int r = rnd() % wsum, c = 0; while (r >= wts[c]) r -= wts[c++];
                jk[j] = khs[c];
            }
            if (si == 0) { jk[0] = {1,1}; jk[1] = {5,0}; jk[2] = {3,1}; jk[3] = {4,0}; }
            std::vector<std::vector<uint32_t>> tr_raw(njobs), tr_rot(njobs), tr_hyb(njobs);
            std::vector<uint64_t> d1_raw(nslots*4,0), d2_raw(nslots*4,0), d1_rot(nslots*4,0), d2_rot(nslots*4,0), d1_hyb(nslots*4,0), d2_hyb(nslots*4,0);
            std::vector<uint32_t*> dr_raw(njobs), dr_rot(njobs), dr_hyb(njobs);
            for (int j = 0; j < njobs; ++j) {
                int nw = nwords(jk[j]), R = nw * 32;
                int b0 = jk[j].ka - 16 + (jk[j].ka < 16 ? R : 0);
                size_t sz = (size_t)tk * tn * nw;
                tr_raw[j].resize(sz); tr_rot[j].resize(sz); tr_hyb[j].resize(sz);
                for (auto& v : tr_raw[j]) v = rnd();
                for (size_t ti = 0; ti < (size_t)tk * tn; ++ti) {
                    rotate_ring_left(&tr_raw[j][ti * nw], &tr_rot[j][ti * nw], nw, b0);
                    if (jk[j].ka == 2 && jk[j].hk == 0) encode_strided_k2_m32(&tr_raw[j][ti * 16], &tr_hyb[j][ti * 16]);
                    else memcpy(&tr_hyb[j][ti * nw], &tr_rot[j][ti * nw], nw * 4);
                }
                CK(cudaMalloc(&dr_raw[j], sz*4)); CK(cudaMemcpy(dr_raw[j], tr_raw[j].data(), sz*4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dr_rot[j], sz*4)); CK(cudaMemcpy(dr_rot[j], tr_rot[j].data(), sz*4, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dr_hyb[j], sz*4)); CK(cudaMemcpy(dr_hyb[j], tr_hyb[j].data(), sz*4, cudaMemcpyHostToDevice));
                int s = j / S.nsel, z = j % S.nsel;
                uint64_t meta = (uint64_t)(jk[j].ka | (jk[j].hk << 8));
                ((z ? d2_raw : d1_raw).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_raw[j]; ((z ? d2_raw : d1_raw).data() + s*4)[3] = meta;
                ((z ? d2_rot : d1_rot).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_rot[j]; ((z ? d2_rot : d1_rot).data() + s*4)[3] = meta;
                ((z ? d2_hyb : d1_hyb).data() + s*4)[0] = (uint64_t)(uintptr_t)dr_hyb[j]; ((z ? d2_hyb : d1_hyb).data() + s*4)[3] = meta;
            }
            std::vector<float> x((size_t)njobs * S.k);
            for (auto& v : x) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
            for (int j = 0; j < njobs; ++j) x[(size_t)j * S.k + (rnd() % S.k)] *= 6.f;
            std::vector<float> xg((size_t)nslots * S.k), xu((size_t)nslots * S.k);
            for (int s = 0; s < nslots; ++s)
                for (int e = 0; e < S.k; ++e) {
                    xg[(size_t)s * S.k + e] = x[(size_t)(s * S.nsel) * S.k + e];
                    if (S.nsel == 2) xu[(size_t)s * S.k + e] = x[(size_t)(s * 2 + 1) * S.k + e];
                }
            std::vector<double> ref((size_t)nslots * n);
            for (int j = 0; j < njobs; ++j) {
                int s = j / S.nsel, z = j % S.nsel;
                ref_gemv(tr_raw[j].data(), jk[j], &x[(size_t)j * S.k], tk, tn, &ref[(size_t)s * n + z * S.nout]);
            }
            double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));
            uint64_t *dd1_raw, *dd2_raw=nullptr, *dd1_rot, *dd2_rot=nullptr, *dd1_hyb, *dd2_hyb=nullptr;
            int* dids; float *dxg, *dxu, *dt;
            CK(cudaMalloc(&dd1_raw, d1_raw.size()*8)); CK(cudaMemcpy(dd1_raw, d1_raw.data(), d1_raw.size()*8, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dd1_rot, d1_rot.size()*8)); CK(cudaMemcpy(dd1_rot, d1_rot.data(), d1_rot.size()*8, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dd1_hyb, d1_hyb.size()*8)); CK(cudaMemcpy(dd1_hyb, d1_hyb.data(), d1_hyb.size()*8, cudaMemcpyHostToDevice));
            if (S.nsel == 2) {
                CK(cudaMalloc(&dd2_raw, d2_raw.size()*8)); CK(cudaMemcpy(dd2_raw, d2_raw.data(), d2_raw.size()*8, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dd2_rot, d2_rot.size()*8)); CK(cudaMemcpy(dd2_rot, d2_rot.data(), d2_rot.size()*8, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dd2_hyb, d2_hyb.size()*8)); CK(cudaMemcpy(dd2_hyb, d2_hyb.data(), d2_hyb.size()*8, cudaMemcpyHostToDevice));
            }
            std::vector<int> ids(nslots); for (int i = 0; i < nslots; ++i) ids[i] = i;
            CK(cudaMalloc(&dids, nslots*4)); CK(cudaMemcpy(dids, ids.data(), nslots*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dxg, xg.size()*4)); CK(cudaMemcpy(dxg, xg.data(), xg.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dxu, xu.size()*4)); CK(cudaMemcpy(dxu, xu.data(), xu.size()*4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dt, (size_t)nslots * n * 4));
            std::vector<float> out((size_t)nslots * n);
            auto check_err = [&]() {
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                CK(cudaMemcpy(out.data(), dt, out.size()*4, cudaMemcpyDeviceToHost));
                double e = 0; for (size_t i = 0; i < out.size(); ++i) e = std::max(e, fabs((double)out[i] - ref[i]));
                return e / ymax;
            };
            const dim3 g((tn + 7) / 8, nslots, S.nsel);
            auto fship = [&]() {
                inkwell_exl3_gemv_moe_s<<<g, 256>>>(dd1_raw, dd2_raw, dids, dxg, dxu, S.k, dt, n, tzoff, tk, tn, S.k, n, 1, nullptr, 0, pwmask);
            };
            int idx = 0;
            auto run_one = [&](const char* name, auto fn) {
                CK(cudaMemset(dt, 0, (size_t)nslots * n * 4));
                fn();
                double er = check_err();
                Timing tb, tc; timeit_interleaved(fship, fn, tb, tc);
                if (si == 0) b3_rows.push_back({name, tb.min_us, 0, tc.min_us, 0, tc.med_us, 0, er, 0});
                else { b3_rows[idx].dn_b = tb.min_us; b3_rows[idx].dn_c = tc.min_us; b3_rows[idx].dn_cm = tc.med_us; b3_rows[idx].dn_err = er; }
                idx++;
            };
            run_one("Shipped sgemv_run (all-7-K, exact)", fship);
            run_one("M=16-Paired C0c (all-7-K, exact I2F)", [&]() { r2::gemv_m16<0, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C1c (all-7-K, exact Mag)", [&]() { r2::gemv_m16<1, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C3b (all-7-K, folded)", [&]() { r2::gemv_m16<2, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C3a (all-7-K, fold glob)", [&]() { r2::gemv_m16<3, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C4 (all-7-K, blk=128)", [&]() { r2::gemv_m16<4, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C4 (all-7-K, blk=16)", [&]() { r2::gemv_m16<5, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("M=16-Paired C6 (all-7-K, dp4a-8)", [&]() { r2::gemv_m16<6, true><<<g, 256>>>(dd1_rot, dd2_rot, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("Hybrid C0c (Strid K2+Pair, exact)", [&]() { r2::gemv_hybrid_allk<0><<<g, 256>>>(dd1_hyb, dd2_hyb, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            run_one("Hybrid C4  (Strid K2+Pair, S16)", [&]() { r2::gemv_hybrid_allk<1><<<g, 256>>>(dd1_hyb, dd2_hyb, dxg, dxu, dt, tzoff, tk, tn, S.k, n, CM); });
            for (int j = 0; j < njobs; ++j) { cudaFree(dr_raw[j]); cudaFree(dr_rot[j]); cudaFree(dr_hyb[j]); }
            cudaFree(dd1_raw); cudaFree(dd1_rot); cudaFree(dd1_hyb);
            if (dd2_raw) { cudaFree(dd2_raw); cudaFree(dd2_rot); cudaFree(dd2_hyb); }
            cudaFree(dids); cudaFree(dxg); cudaFree(dxu); cudaFree(dt);
        }
        printf("%-38s %7s %7s %7s %7s %7s %7s %8s %9s\n",
               "Design (SAGE all-7-K, interlv)", "Ship_us", "Cand_us", "CandMed", "ShipGw", "CandGw", "Spdup", "max_rel", "GU/DN");
        for (const auto& r : b3_rows) {
            double bmin = r.gu_b + r.dn_b, cmin = r.gu_c + r.dn_c, cmed = r.gu_cm + r.dn_cm;
            double bgw = nw_tot / (bmin * 1e3), cgw = nw_tot / (cmin * 1e3);
            printf("%-38s %7.1f %7.1f %7.1f %7.1f %7.1f %6.2fx %8.2e %.0f/%.0f\n",
                   r.name.c_str(), bmin, cmin, cmed, bgw, cgw, bmin / cmin, std::max(r.gu_err, r.dn_err), r.gu_c, r.dn_c);
        }
    }
    } // end mask & 4

    cudaFree(d_lut16); cudaFree(d_lut8);
    return 0;
}
