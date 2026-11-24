// Round 2 EXL3 decode-math CUDA kernels for exact sm_61 SASS issued-instruction
// counting and -Xptxas -v register / spill auditing.
//
// EVERY design enforces <= 40 regs/thread (0 spills) and preserves the shipped
// 8-tiles/block (256 threads/block, >= 6 blocks/SM) grid parallelism:
//   - M=8  (32 lanes/tile, 0 in-block tk-split, exact shipped warp shape)
//   - M=16 (16 lanes/tile, 2-way in-block tk-split -> exact 8 tiles/block, 2 accs, 0 stash regs!)
//   - M=32 ( 8 lanes/tile, 4-way in-block tk-split -> exact 8 tiles/block, 2 accs, 0 stash regs!)

#include <cstdint>

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

// Clean 3-XMAD 16x32 multiply reading U.H0 / U.H1 directly (0 BFE, 0 LOP, 0 regs for C_mul1 via cbank!)
__device__ __forceinline__ uint32_t mul16_h0(uint32_t U, uint32_t C_mul1) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, cl;\n\t}"
        : "=r"(t) : "r"(U), "r"(C_mul1));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, lo, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(U), "r"(C_mul1), "r"(t));
    return p;
}
__device__ __forceinline__ uint32_t mul16_h1(uint32_t U, uint32_t C_mul1) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, cl;\n\t}"
        : "=r"(t) : "r"(U), "r"(C_mul1));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, hi, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(U), "r"(C_mul1), "r"(t));
    return p;
}

// XMAD + XMAD + LEA.HI + 2*DP2A.LO (tests XMAD.CHI path in stock ptxas)
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

__device__ __forceinline__ uint32_t sgemv_mul16_ship(uint32_t w) {
    const uint32_t h = w & 0xffffu;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}
template <int D> __device__ __forceinline__ uint32_t sgemv_win_ship(uint32_t hi, uint32_t lo) {
    if constexpr (D <= 16) return hi >> (16 - D);
    else return __funnelshift_r(lo, hi, 48 - D);
}

// ============================================================================
// 0. SHIPPED CORE (exact sgemv_run<2,0>, M=8, unroll-4 = 32 weights/iter)
// ============================================================================
__global__ void __launch_bounds__(256) k_shipped_m8_k2(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) float xs[2304];
    for (int e = threadIdx.x; e < 2304; e += 256) xs[e] = xg[e];
    __syncthreads();
    const int t = threadIdx.x & 31;
    const int NW = 16;
    int B = (8 * t + 1) * 2 - 16; if (B < 0) B += 512;
    const int w0 = B >> 5, o = B & 31;
    const int w1 = (w0 + 1 == NW) ? 0 : w0 + 1;
    const int w2 = (w1 + 1 == NW) ? 0 : w1 + 1;
    const int r0 = 2 * (t & 3);
    const size_t stride = (size_t)tn * NW;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 4
    for (int a = 0; a < ntiles; ++a) {
        const uint32_t* q = tb + (size_t)a * stride;
        const int kb = a * 16 + r0;
        const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
        const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
        uint32_t pp[8];
        pp[0] = sgemv_mul16_ship(sgemv_win_ship<0>(hi, lo));
        pp[1] = sgemv_mul16_ship(sgemv_win_ship<2>(hi, lo));
        pp[2] = sgemv_mul16_ship(sgemv_win_ship<4>(hi, lo));
        pp[3] = sgemv_mul16_ship(sgemv_win_ship<6>(hi, lo));
        pp[4] = sgemv_mul16_ship(sgemv_win_ship<8>(hi, lo));
        pp[5] = sgemv_mul16_ship(sgemv_win_ship<10>(hi, lo));
        pp[6] = sgemv_mul16_ship(sgemv_win_ship<12>(hi, lo));
        pp[7] = sgemv_mul16_ship(sgemv_win_ship<14>(hi, lo));
        const float2 xa = *(const float2*)&xs[kb], xc = *(const float2*)&xs[kb + 8];
        float f[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = (float)(int)dp4a_uu(pp[j], 0x01010101u, 0u);
        a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0); a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
        a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1); a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
    }
    out[threadIdx.x] = a0 + a1;
}

// ============================================================================
// 1. M=8 UPGRADED (32 lanes/tile, exact shipped grid, unroll-4, opaque C_mul1)
//    MODE 0: C0c (Zero-centered I2F, exact)
//    MODE 1: C3b (Folded 0x3F800000+255*bs per-tile)
//    MODE 2: C4  (IDP4A(-510) + XMAD.S16, int16 x)
//    MODE 3: C6  (dp4a-8, int8 x)
// ============================================================================
template <int MODE>
__global__ void __launch_bounds__(256, 6) k_m8_opt_k2(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg, const float* __restrict__ sxtg,
    const uint32_t* __restrict__ xqg, uint32_t CM, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) uint32_t xbuf[2304];
    __shared__ float sxt[144];
    float* xs = (float*)xbuf;
    if constexpr (MODE <= 1) {
        for (int e = threadIdx.x; e < 2304; e += 256) xs[e] = xg[e];
        if constexpr (MODE == 1) { if (threadIdx.x < 144) sxt[threadIdx.x] = sxtg[threadIdx.x]; }
    } else if constexpr (MODE == 2) {
        for (int e = threadIdx.x; e < 1152; e += 256) xbuf[e] = xqg[e];
    } else {
        for (int e = threadIdx.x; e < 2304; e += 256) xbuf[e] = xqg[e];
    }
    __syncthreads();
    const int t = threadIdx.x & 31;
    const int NW = 16;
    const int w0 = t >> 1, o = (t & 1) << 4;
    const int w1 = (w0 + 1 == NW) ? 0 : w0 + 1;
    const int r0 = 2 * (t & 3);
    const size_t stride = (size_t)tn * NW;
    const uint32_t* p0 = tb + w0;
    const uint32_t* p1 = tb + w1;
    float a0 = 0.f, a1 = 0.f;
    int ia0 = 0, ia1 = 0;

#pragma unroll 4
    for (int a = 0; a < ntiles; ++a, p0 += stride, p1 += stride) {
        const uint32_t W0 = __ldg(p0);
        const uint32_t W1 = __ldg(p1);
        const uint32_t hi = __funnelshift_l(W1, W0, o);
        uint32_t pp[8];
        pp[0] = mul16_h1(hi, CM);
        pp[1] = mul16_h1(hi << 2, CM);
        pp[2] = mul16_h1(hi << 4, CM);
        pp[3] = mul16_h1(hi << 6, CM);
        pp[4] = mul16_h1(hi << 8, CM);
        pp[5] = mul16_h1(hi << 10, CM);
        pp[6] = mul16_h1(hi << 12, CM);
        pp[7] = mul16_h1(hi << 14, CM);
        if constexpr (MODE <= 1) {
            const float2 xa = *(const float2*)&xs[a * 16 + r0], xc = *(const float2*)&xs[a * 16 + r0 + 8];
            float f[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (MODE == 0) f[j] = (float)(int)dp4a_uu(pp[j], 0x01010101u, (uint32_t)(-510));
                else f[j] = __uint_as_float(dp4a_uu(pp[j], 0xFFFFFFFFu, 0x3F800000u));
            }
            if constexpr (MODE == 1) {
                float t0 = sxt[a], t1 = sxt[a];
                t0 = fmaf(xa.x, f[0], t0); t0 = fmaf(xa.y, f[1], t0); t0 = fmaf(xc.x, f[2], t0); t0 = fmaf(xc.y, f[3], t0);
                t1 = fmaf(xa.x, f[4], t1); t1 = fmaf(xa.y, f[5], t1); t1 = fmaf(xc.x, f[6], t1); t1 = fmaf(xc.y, f[7], t1);
                a0 += t0; a1 += t1;
            } else {
                a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0); a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
                a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1); a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
            }
        } else if constexpr (MODE == 2) {
            const uint32_t xp0 = xbuf[a * 8 + (r0 >> 1)];
            const uint32_t xp1 = xbuf[a * 8 + (r0 >> 1) + 4];
            int b[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) b[j] = (int)dp4a_uu(pp[j], 0x01010101u, (uint32_t)(-510));
            ia0 = xmad_s16_h0(xp0, b[0], ia0); ia0 = xmad_s16_h1(xp0, b[1], ia0);
            ia0 = xmad_s16_h0(xp1, b[2], ia0); ia0 = xmad_s16_h1(xp1, b[3], ia0);
            ia1 = xmad_s16_h0(xp0, b[4], ia1); ia1 = xmad_s16_h1(xp0, b[5], ia1);
            ia1 = xmad_s16_h0(xp1, b[6], ia1); ia1 = xmad_s16_h1(xp1, b[7], ia1);
        } else {
            const uint2 xa = *(const uint2*)&xbuf[a * 16 + r0], xc = *(const uint2*)&xbuf[a * 16 + r0 + 8];
            ia0 = dp4a_su(xa.x, pp[0], ia0); ia0 = dp4a_su(xa.y, pp[1], ia0);
            ia0 = dp4a_su(xc.x, pp[2], ia0); ia0 = dp4a_su(xc.y, pp[3], ia0);
            ia1 = dp4a_su(xa.x, pp[4], ia1); ia1 = dp4a_su(xa.y, pp[5], ia1);
            ia1 = dp4a_su(xc.x, pp[6], ia1); ia1 = dp4a_su(xc.y, pp[7], ia1);
        }
    }
    if constexpr (MODE <= 1) out[threadIdx.x] = a0 + a1;
    else out[threadIdx.x] = (float)(ia0 + ia1);
}
template __global__ void k_m8_opt_k2<0>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_m8_opt_k2<1>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_m8_opt_k2<2>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_m8_opt_k2<3>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);

// ============================================================================
// 2. PAIRED M=16 K=2.0 (16 lanes/tile, 2-way in-block tk-split -> EXACT 8 TILES/BLOCK!)
//    ZERO stash regs! ONLY 2 accs! o=0 for ALL 16 lanes! unroll-2 -> 32 weights/iter!
// ============================================================================
template <int MODE>
__device__ __forceinline__ void mac_pair_f(uint32_t U, uint32_t CM, float x1, float x0, float& a1, float& a0) {
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
__device__ __forceinline__ void mac_pair_c4(uint32_t U, uint32_t CM, uint32_t xp1, bool hi1, uint32_t xp0, bool hi0, int& a1, int& a0) {
    int b1 = (int)dp4a_uu(mul16_h1(U, CM), 0x01010101u, (uint32_t)(-510));
    int b0 = (int)dp4a_uu(mul16_h0(U, CM), 0x01010101u, (uint32_t)(-510));
    a1 = hi1 ? xmad_s16_h1(xp1, b1, a1) : xmad_s16_h0(xp1, b1, a1);
    a0 = hi0 ? xmad_s16_h1(xp0, b0, a0) : xmad_s16_h0(xp0, b0, a0);
}
__device__ __forceinline__ void mac_pair_c6(uint32_t U, uint32_t CM, uint32_t x1, uint32_t x0, int& a1, int& a0) {
    a1 = dp4a_su(x1, mul16_h1(U, CM), a1);
    a0 = dp4a_su(x0, mul16_h0(U, CM), a0);
}

template <int MODE>
__global__ void __launch_bounds__(256, 6) k_paired_m16_k2(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg, const float* __restrict__ sxtg,
    const uint32_t* __restrict__ xqg, uint32_t CM, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) uint32_t xbuf[2304];
    __shared__ float sxt[144];
    float* xs = (float*)xbuf;
    if constexpr (MODE <= 3) {
        for (int e = threadIdx.x; e < 2304; e += 256) {
            int r = e & 15;
            xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = xg[e];
        }
        if constexpr (MODE == 2) { if (threadIdx.x < 144) sxt[threadIdx.x] = sxtg[threadIdx.x]; }
    } else if constexpr (MODE == 4) {
        for (int e = threadIdx.x; e < 1152; e += 256) {
            int p = e & 7;
            xbuf[(e & ~7) + ((p & 2) ? 4 : 0) + ((p & 4) ? 2 : 0) + (p & 1)] = xqg[e];
        }
    } else {
        for (int e = threadIdx.x; e < 2304; e += 256) {
            int r = e & 15;
            xbuf[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = xqg[e];
        }
    }
    __syncthreads();

    const int sub = threadIdx.x & 15;
    const int half_x = (sub & 1) ? 8 : 0;
    const int half_x16 = (sub & 1) ? 4 : 0;
    const int w0 = sub, w1 = (sub + 1) & 15;
    const size_t stride = (size_t)tn * 16;
    const int a_start = (threadIdx.x >> 7);
    const uint32_t* p0 = tb + (size_t)a_start * stride + w0;
    const uint32_t* p1 = tb + (size_t)a_start * stride + w1;
    const size_t step2 = 2 * stride;
    float a0 = 0.f, a1 = 0.f;
    int ia0 = 0, ia1 = 0;

#pragma unroll 2
    for (int a = a_start; a < ntiles; a += 2, p0 += step2, p1 += step2) {
        const uint32_t W0 = __ldg(p0);
        const uint32_t W1 = __ldg(p1);
        if constexpr (MODE <= 3) {
            const float4 xA = *(const float4*)&xs[a * 16 + half_x + 0];
            const float4 xB = *(const float4*)&xs[a * 16 + half_x + 4];
            float t0 = (MODE == 2) ? sxt[a] : a0;
            float t1 = (MODE == 2) ? sxt[a] : a1;
            mac_pair_f<MODE>(W0,                         CM, xA.x, xA.z, t0, t0);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0, 2), CM, xA.y, xA.w, t0, t0);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0, 4), CM, xB.x, xB.z, t0, t0);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0, 6), CM, xB.y, xB.w, t0, t0);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0, 8), CM, xA.x, xA.z, t1, t1);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0,10), CM, xA.y, xA.w, t1, t1);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0,12), CM, xB.x, xB.z, t1, t1);
            mac_pair_f<MODE>(__funnelshift_l(W1, W0,14), CM, xB.y, xB.w, t1, t1);
            if constexpr (MODE == 2) { a0 += t0; a1 += t1; } else { a0 = t0; a1 = t1; }
        } else if constexpr (MODE == 4) {
            const uint4 xp = *(const uint4*)&xbuf[a * 8 + half_x16];
            mac_pair_c4(W0,                         CM, xp.x, false, xp.y, false, ia0, ia0);
            mac_pair_c4(__funnelshift_l(W1, W0, 2), CM, xp.x, true,  xp.y, true,  ia0, ia0);
            mac_pair_c4(__funnelshift_l(W1, W0, 4), CM, xp.z, false, xp.w, false, ia0, ia0);
            mac_pair_c4(__funnelshift_l(W1, W0, 6), CM, xp.z, true,  xp.w, true,  ia0, ia0);
            mac_pair_c4(__funnelshift_l(W1, W0, 8), CM, xp.x, false, xp.y, false, ia1, ia1);
            mac_pair_c4(__funnelshift_l(W1, W0,10), CM, xp.x, true,  xp.y, true,  ia1, ia1);
            mac_pair_c4(__funnelshift_l(W1, W0,12), CM, xp.z, false, xp.w, false, ia1, ia1);
            mac_pair_c4(__funnelshift_l(W1, W0,14), CM, xp.z, true,  xp.w, true,  ia1, ia1);
        } else if constexpr (MODE == 5) {
            const uint4 xA = *(const uint4*)&xbuf[a * 16 + half_x + 0];
            const uint4 xB = *(const uint4*)&xbuf[a * 16 + half_x + 4];
            mac_pair_c6(W0,                         CM, xA.x, xA.z, ia0, ia0);
            mac_pair_c6(__funnelshift_l(W1, W0, 2), CM, xA.y, xA.w, ia0, ia0);
            mac_pair_c6(__funnelshift_l(W1, W0, 4), CM, xB.x, xB.z, ia0, ia0);
            mac_pair_c6(__funnelshift_l(W1, W0, 6), CM, xB.y, xB.w, ia0, ia0);
            mac_pair_c6(__funnelshift_l(W1, W0, 8), CM, xA.x, xA.z, ia1, ia1);
            mac_pair_c6(__funnelshift_l(W1, W0,10), CM, xA.y, xA.w, ia1, ia1);
            mac_pair_c6(__funnelshift_l(W1, W0,12), CM, xB.x, xB.z, ia1, ia1);
            mac_pair_c6(__funnelshift_l(W1, W0,14), CM, xB.y, xB.w, ia1, ia1);
        } else {
            const uint4 xA = *(const uint4*)&xbuf[a * 16 + half_x + 0];
            const uint4 xB = *(const uint4*)&xbuf[a * 16 + half_x + 4];
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
    }
    if constexpr (MODE <= 3) out[threadIdx.x] = a0 + a1;
    else out[threadIdx.x] = (float)(ia0 + ia1);
}
template __global__ void k_paired_m16_k2<0>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_paired_m16_k2<1>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_paired_m16_k2<2>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_paired_m16_k2<3>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_paired_m16_k2<4>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_paired_m16_k2<5>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_paired_m16_k2<6>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, int, float*);

// ============================================================================
// 3. LOSSLESS STRIDED-CRUMB + SHFL + ISCADD (K=2.0, 0% memory bloat!)
//    M=32 with 4-way in-block tk-split -> EXACT 8 tiles/block, 2 accs, <=40 regs!
// ============================================================================
template <int MODE>
__global__ void __launch_bounds__(256, 6) k_strided_m32_k2(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg, const uint32_t* __restrict__ xqg,
    uint32_t CM, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) uint32_t xbuf[2304];
    float* xs = (float*)xbuf;
    if constexpr (MODE == 0) {
        for (int e = threadIdx.x; e < 2304; e += 256) {
            int r = e & 15;
            xs[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3)] = xg[e];
        }
    } else if constexpr (MODE == 1) {
        for (int e = threadIdx.x; e < 1152; e += 256) xbuf[e] = xqg[e];
    } else {
        for (int e = threadIdx.x; e < 2304; e += 256) {
            int r = e & 15;
            xbuf[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3)] = xqg[e];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int sub = lane & 7;
    const uint32_t v_tbl = (lane >> 2) & 3u, u_tbl = lane & 3u;
    const uint32_t D_reg = v_tbl * 0x83DCD12Du - ((u_tbl * 0xD12Du) << 16);
    const size_t stride = (size_t)tn * 16;
    const int a_start = threadIdx.x >> 6;
    const uint32_t* p0 = tb + (size_t)a_start * stride + sub * 2;
    const uint32_t* pn = tb + (size_t)a_start * stride + ((sub + 1) & 7) * 2;
    const size_t step4 = 4 * stride;
    float a0 = 0.f, a1 = 0.f;
    int ia0 = 0, ia1 = 0;

#pragma unroll 1
    for (int a = a_start; a < ntiles; a += 4, p0 += step4, pn += step4) {
        const uint2 w01 = __ldg((const uint2*)p0);
        const uint2 w23 = __ldg((const uint2*)pn);
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
    }
    if constexpr (MODE == 0) out[threadIdx.x] = a0 + a1;
    else out[threadIdx.x] = (float)(ia0 + ia1);
}
template __global__ void k_strided_m32_k2<0>(const uint32_t*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_strided_m32_k2<1>(const uint32_t*, const float*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_strided_m32_k2<2>(const uint32_t*, const float*, const uint32_t*, uint32_t, int, int, float*);

// ============================================================================
// 4. NEGATIVE LUT PROOFS AT M=16 (same 8 tiles/block grid!)
// ============================================================================
__global__ void __launch_bounds__(256, 6) k_lut128k_m16_k2(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg,
    const int16_t* __restrict__ lut, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) float xs[2304];
    for (int e = threadIdx.x; e < 2304; e += 256) {
        int r = e & 15;
        xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = xg[e];
    }
    __syncthreads();
    const int sub = threadIdx.x & 15, hx = (sub & 1) ? 8 : 0;
    const size_t stride = (size_t)tn * 16;
    const int a_start = threadIdx.x >> 7;
    const uint32_t* p0 = tb + (size_t)a_start * stride + sub;
    const uint32_t* p1 = tb + (size_t)a_start * stride + ((sub + 1) & 15);
    const size_t step2 = 2 * stride;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 2
    for (int a = a_start; a < ntiles; a += 2, p0 += step2, p1 += step2) {
        const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
        const float4 xA = *(const float4*)&xs[a*16 + hx], xB = *(const float4*)&xs[a*16 + hx + 4];
        auto lp = [&](uint32_t U, float x1, float x0, float& acc) {
            acc = fmaf(x1, (float)__ldg(lut + (U >> 16)), acc);
            acc = fmaf(x0, (float)__ldg(lut + (U & 0xFFFFu)), acc);
        };
        lp(W0, xA.x, xA.z, a0); lp(__funnelshift_l(W1,W0,2), xA.y, xA.w, a0);
        lp(__funnelshift_l(W1,W0,4), xB.x, xB.z, a0); lp(__funnelshift_l(W1,W0,6), xB.y, xB.w, a0);
        lp(__funnelshift_l(W1,W0,8), xA.x, xA.z, a1); lp(__funnelshift_l(W1,W0,10), xA.y, xA.w, a1);
        lp(__funnelshift_l(W1,W0,12), xB.x, xB.z, a1); lp(__funnelshift_l(W1,W0,14), xB.y, xB.w, a1);
    }
    out[threadIdx.x] = a0 + a1;
}

__global__ void __launch_bounds__(256, 6) k_lut64k_m16_k2(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg,
    const int8_t* __restrict__ lut, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) float xs[2304];
    for (int e = threadIdx.x; e < 2304; e += 256) {
        int r = e & 15;
        xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = xg[e];
    }
    __syncthreads();
    const int sub = threadIdx.x & 15, hx = (sub & 1) ? 8 : 0;
    const size_t stride = (size_t)tn * 16;
    const int a_start = threadIdx.x >> 7;
    const uint32_t* p0 = tb + (size_t)a_start * stride + sub;
    const uint32_t* p1 = tb + (size_t)a_start * stride + ((sub + 1) & 15);
    const size_t step2 = 2 * stride;
    float a0 = 0.f, a1 = 0.f;
#pragma unroll 2
    for (int a = a_start; a < ntiles; a += 2, p0 += step2, p1 += step2) {
        const uint32_t W0 = __ldg(p0), W1 = __ldg(p1);
        const float4 xA = *(const float4*)&xs[a*16 + hx], xB = *(const float4*)&xs[a*16 + hx + 4];
        auto lp = [&](uint32_t U, float x1, float x0, float& acc) {
            acc = fmaf(x1, (float)__ldg(lut + (U >> 16)), acc);
            acc = fmaf(x0, (float)__ldg(lut + (U & 0xFFFFu)), acc);
        };
        lp(W0, xA.x, xA.z, a0); lp(__funnelshift_l(W1,W0,2), xA.y, xA.w, a0);
        lp(__funnelshift_l(W1,W0,4), xB.x, xB.z, a0); lp(__funnelshift_l(W1,W0,6), xB.y, xB.w, a0);
        lp(__funnelshift_l(W1,W0,8), xA.x, xA.z, a1); lp(__funnelshift_l(W1,W0,10), xA.y, xA.w, a1);
        lp(__funnelshift_l(W1,W0,12), xB.x, xB.z, a1); lp(__funnelshift_l(W1,W0,14), xB.y, xB.w, a1);
    }
    out[threadIdx.x] = a0 + a1;
}

// ============================================================================
// 5. DIRECTION 3: MULTI-TOKEN BATCHING (B = 1, 2, 3, 4) AT M=16 (EXACT 8 TILES/BLOCK!)
// ============================================================================
template <int B>
__global__ void __launch_bounds__(256, 6) k_batch_m16_c0c(
    const uint32_t* __restrict__ tb, const float* __restrict__ xg, uint32_t CM, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) float xs[32 * 16 * B];
    const int sub = threadIdx.x & 15;
    const int r_off = (sub & 1) ? 4 : 0;
    const size_t stride = (size_t)tn * 16;
    const int a_off = threadIdx.x >> 7;
    const uint32_t* p0 = tb + (size_t)a_off * stride + sub;
    const uint32_t* p1 = tb + (size_t)a_off * stride + ((sub + 1) & 15);
    const size_t step2 = 2 * stride;
    float a0[B], a1[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { a0[b] = 0.f; a1[b] = 0.f; }

    for (int tbase = 0; tbase < ntiles; tbase += 32) {
        const int nloc = min(32, ntiles - tbase);
        for (int e = threadIdx.x; e < nloc * 16 * B; e += 256) xs[e] = xg[(tbase * 16) * B + e];
        __syncthreads();
#pragma unroll 1
        for (int t = a_off; t < nloc; t += 2, p0 += step2, p1 += step2) {
            const uint32_t W0 = __ldg(p0);
            const uint32_t W1 = __ldg(p1);
            const float* xt = &xs[t * 16 * B];
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
        __syncthreads();
    }
    float s = 0.f;
#pragma unroll
    for (int b = 0; b < B; ++b) s += a0[b] + a1[b];
    out[threadIdx.x] = s;
}
template __global__ void k_batch_m16_c0c<1>(const uint32_t*, const float*, uint32_t, int, int, float*);
template __global__ void k_batch_m16_c0c<2>(const uint32_t*, const float*, uint32_t, int, int, float*);
template __global__ void k_batch_m16_c0c<3>(const uint32_t*, const float*, uint32_t, int, int, float*);
template __global__ void k_batch_m16_c0c<4>(const uint32_t*, const float*, uint32_t, int, int, float*);

template <int B>
__global__ void __launch_bounds__(256, 6) k_batch_m16_c4(
    const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16g, uint32_t CM, int tn, int ntiles, float* __restrict__ out) {
    __shared__ __align__(16) uint32_t xs[32 * 16 * (B / 2)];
    const int sub = threadIdx.x & 15;
    const int r_off = (sub & 1) ? 4 : 0;
    const size_t stride = (size_t)tn * 16;
    const int a_off = threadIdx.x >> 7;
    const uint32_t* p0 = tb + (size_t)a_off * stride + sub;
    const uint32_t* p1 = tb + (size_t)a_off * stride + ((sub + 1) & 15);
    const size_t step2 = 2 * stride;
    int a0[B], a1[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { a0[b] = 0; a1[b] = 0; }

    for (int tbase = 0; tbase < ntiles; tbase += 32) {
        const int nloc = min(32, ntiles - tbase);
        for (int e = threadIdx.x; e < nloc * 16 * (B / 2); e += 256) xs[e] = x16g[(tbase * 16) * (B / 2) + e];
        __syncthreads();
#pragma unroll 1
        for (int t = a_off; t < nloc; t += 2, p0 += step2, p1 += step2) {
            const uint32_t W0 = __ldg(p0);
            const uint32_t W1 = __ldg(p1);
            const uint32_t* xt = &xs[t * 16 * (B / 2)];
            auto do_cols = [&](uint32_t U0, uint32_t U1, int r_hi, int r_lo) {
                {
                    int b0 = (int)dp4a_uu(mul16_h1(U0, CM), 0x01010101u, (uint32_t)(-510));
                    int b1 = (int)dp4a_uu(mul16_h1(U1, CM), 0x01010101u, (uint32_t)(-510));
                    if constexpr (B == 2) {
                        uint32_t xp = xt[r_hi];
                        a0[0] = xmad_s16_h0(xp, b0, a0[0]); a0[1] = xmad_s16_h1(xp, b0, a0[1]);
                        a1[0] = xmad_s16_h0(xp, b1, a1[0]); a1[1] = xmad_s16_h1(xp, b1, a1[1]);
                    } else {
                        uint2 xp = *(const uint2*)&xt[r_hi * 2];
                        a0[0] = xmad_s16_h0(xp.x, b0, a0[0]); a0[1] = xmad_s16_h1(xp.x, b0, a0[1]);
                        a0[2] = xmad_s16_h0(xp.y, b0, a0[2]); a0[3] = xmad_s16_h1(xp.y, b0, a0[3]);
                        a1[0] = xmad_s16_h0(xp.x, b1, a1[0]); a1[1] = xmad_s16_h1(xp.x, b1, a1[1]);
                        a1[2] = xmad_s16_h0(xp.y, b1, a1[2]); a1[3] = xmad_s16_h1(xp.y, b1, a1[3]);
                    }
                }
                {
                    int b0 = (int)dp4a_uu(mul16_h0(U0, CM), 0x01010101u, (uint32_t)(-510));
                    int b1 = (int)dp4a_uu(mul16_h0(U1, CM), 0x01010101u, (uint32_t)(-510));
                    if constexpr (B == 2) {
                        uint32_t xp = xt[r_lo];
                        a0[0] = xmad_s16_h0(xp, b0, a0[0]); a0[1] = xmad_s16_h1(xp, b0, a0[1]);
                        a1[0] = xmad_s16_h0(xp, b1, a1[0]); a1[1] = xmad_s16_h1(xp, b1, a1[1]);
                    } else {
                        uint2 xp = *(const uint2*)&xt[r_lo * 2];
                        a0[0] = xmad_s16_h0(xp.x, b0, a0[0]); a0[1] = xmad_s16_h1(xp.x, b0, a0[1]);
                        a0[2] = xmad_s16_h0(xp.y, b0, a0[2]); a0[3] = xmad_s16_h1(xp.y, b0, a0[3]);
                        a1[0] = xmad_s16_h0(xp.x, b1, a1[0]); a1[1] = xmad_s16_h1(xp.x, b1, a1[1]);
                        a1[2] = xmad_s16_h0(xp.y, b1, a1[2]); a1[3] = xmad_s16_h1(xp.y, b1, a1[3]);
                    }
                }
            };
            do_cols(W0,                         __funnelshift_l(W1, W0,  8), r_off + 0, r_off + 2);
            do_cols(__funnelshift_l(W1, W0, 2), __funnelshift_l(W1, W0, 10), r_off + 1, r_off + 3);
            do_cols(__funnelshift_l(W1, W0, 4), __funnelshift_l(W1, W0, 12), r_off + 8, r_off + 10);
            do_cols(__funnelshift_l(W1, W0, 6), __funnelshift_l(W1, W0, 14), r_off + 9, r_off + 11);
        }
        __syncthreads();
    }
    int s = 0;
#pragma unroll
    for (int b = 0; b < B; ++b) s += a0[b] + a1[b];
    out[threadIdx.x] = (float)s;
}
template __global__ void k_batch_m16_c4<2>(const uint32_t*, const uint32_t*, uint32_t, int, int, float*);
template __global__ void k_batch_m16_c4<4>(const uint32_t*, const uint32_t*, uint32_t, int, int, float*);
