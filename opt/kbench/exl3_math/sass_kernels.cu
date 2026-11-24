// Minimal CUDA __global__ inner-loop snippets compiled for sm_61 to inspect exact SASS
// via cuobjdump -sass. NEVER executed on GPU (compile + static SASS analysis only).
//
// Compile:
//   ~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 --use_fast_math \
//     -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
//     -gencode arch=compute_61,code=sm_61 -c opt/kbench/exl3_math/sass_kernels.cu -o opt/kbench/exl3_math/sass_kernels.o

#include <cstdint>

// ---- PTX helpers targeting exact sm_61 instructions ----

// Clean 3-XMAD (ZERO BFE, ZERO LOP, ZERO SHL) reading U.H0 directly via PTX mul.wide.u16 + mad.lo.u32(*65536).
// Passing C_mul1 = 0x83DCD12D as a kernel param prevents ptxas from folding (0x83DC << 16) into a 32x32 4-XMAD+BFE sequence.
// Emits on sm_61:
//   XMAD     Rt, U.H0, C.H0, RZ
//   XMAD     Ru, U.H0, C.H1, RZ
//   XMAD.PSL Rp, Ru,   0x1,  Rt
__device__ __forceinline__ uint32_t mul16_h0(uint32_t U, uint32_t C_mul1) {
    uint32_t t, p;
    asm("{\n\t"
        ".reg .u16 lo, hi, cl, ch;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "mov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 %0, lo, cl;\n\t"
        "}" : "=r"(t) : "r"(U), "r"(C_mul1));
    asm("{\n\t"
        ".reg .u16 lo, hi, cl, ch;\n\t"
        ".reg .u32 q;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "mov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, lo, ch;\n\t"
        "mad.lo.u32 %0, q, 65536, %3;\n\t"
        "}" : "=r"(p) : "r"(U), "r"(C_mul1), "r"(t));
    return p;
}

// Clean 3-XMAD (ZERO BFE, ZERO LOP, ZERO SHL) reading U.H1 directly.
__device__ __forceinline__ uint32_t mul16_h1(uint32_t U, uint32_t C_mul1) {
    uint32_t t, p;
    asm("{\n\t"
        ".reg .u16 lo, hi, cl, ch;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "mov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 %0, hi, cl;\n\t"
        "}" : "=r"(t) : "r"(U), "r"(C_mul1));
    asm("{\n\t"
        ".reg .u16 lo, hi, cl, ch;\n\t"
        ".reg .u32 q;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "mov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, hi, ch;\n\t"
        "mad.lo.u32 %0, q, 65536, %3;\n\t"
        "}" : "=r"(p) : "r"(U), "r"(C_mul1), "r"(t));
    return p;
}

// Unsigned dp4a.u32.u32(a, b, c)
__device__ __forceinline__ uint32_t dp4a_uu(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// dp2a.lo / dp2a.hi
__device__ __forceinline__ int dp2a_lo(uint32_t a, uint32_t b, int c) {
    int d;
    asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
__device__ __forceinline__ int dp2a_hi(uint32_t a, uint32_t b, int c) {
    int d;
    asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// XMAD.S16.S16: acc += xpair.H0 * (int16_t)bs_c (1 full-rate XMAD!)
__device__ __forceinline__ int xmad_s16_h0(uint32_t xpair, int bs_c, int acc) {
    int d;
    asm("{\n\t"
        ".reg .s16 x0, x1, b0;\n\t"
        ".reg .s32 prod;\n\t"
        "mov.b32 {x0, x1}, %1;\n\t"
        "cvt.s16.s32 b0, %2;\n\t"
        "mul.wide.s16 prod, x0, b0;\n\t"
        "add.s32 %0, prod, %3;\n\t"
        "}" : "=r"(d) : "r"(xpair), "r"(bs_c), "r"(acc));
    return d;
}

// XMAD.S16.S16: acc += xpair.H1 * (int16_t)bs_c (1 full-rate XMAD!)
__device__ __forceinline__ int xmad_s16_h1(uint32_t xpair, int bs_c, int acc) {
    int d;
    asm("{\n\t"
        ".reg .s16 x0, x1, b0;\n\t"
        ".reg .s32 prod;\n\t"
        "mov.b32 {x0, x1}, %1;\n\t"
        "cvt.s16.s32 b0, %2;\n\t"
        "mul.wide.s16 prod, x1, b0;\n\t"
        "add.s32 %0, prod, %3;\n\t"
        "}" : "=r"(d) : "r"(xpair), "r"(bs_c), "r"(acc));
    return d;
}

// Baseline helpers (exact copy from kernels.cu:2437-2445 / sgemv_bench.cu:30-48)
__device__ __forceinline__ uint32_t sgemv_mul16_orig(uint32_t w) {
    const uint32_t h = w & 0xffffu;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}
template <int D> __device__ __forceinline__ uint32_t sgemv_win_orig(uint32_t hi, uint32_t lo) {
    if constexpr (D <= 16) return hi >> (16 - D);
    else return __funnelshift_r(lo, hi, 48 - D);
}

// ============================================================================
// 1. BASELINE KERNELS (M=8, K=2.0, unrotated ring, exact match to sgemv_bench)
// ============================================================================
template <int MAC>
__global__ void k_base_k2(const uint32_t* __restrict__ tb, const float* __restrict__ xs,
                          const uint32_t* __restrict__ xq, int ntiles, float* __restrict__ out) {
    const int t = threadIdx.x & 31;
    const int NW = 16;
    int B = (8 * t + 1) * 2 - 16;
    if (B < 0) B += 512;
    const int w0 = B >> 5, o = B & 31;
    const int w1 = (w0 + 1 == NW) ? 0 : w0 + 1;
    const int w2 = (w1 + 1 == NW) ? 0 : w1 + 1;
    const int r0 = 2 * (t & 3);
    float a0 = 0.f, a1 = 0.f;
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a) {
        const uint32_t* q = tb + a * NW;
        const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
        const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
        uint32_t pp[8];
        pp[0] = sgemv_mul16_orig(sgemv_win_orig<0>(hi, lo));
        pp[1] = sgemv_mul16_orig(sgemv_win_orig<2>(hi, lo));
        pp[2] = sgemv_mul16_orig(sgemv_win_orig<4>(hi, lo));
        pp[3] = sgemv_mul16_orig(sgemv_win_orig<6>(hi, lo));
        pp[4] = sgemv_mul16_orig(sgemv_win_orig<8>(hi, lo));
        pp[5] = sgemv_mul16_orig(sgemv_win_orig<10>(hi, lo));
        pp[6] = sgemv_mul16_orig(sgemv_win_orig<12>(hi, lo));
        pp[7] = sgemv_mul16_orig(sgemv_win_orig<14>(hi, lo));
        if constexpr (MAC <= 1) {
            const float2 xa = *(const float2*)&xs[a * 16 + r0];
            const float2 xc = *(const float2*)&xs[a * 16 + r0 + 8];
            float f[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (MAC == 0) f[j] = (float)(int)dp4a_uu(pp[j], 0x01010101u, 0u);
                else f[j] = __uint_as_float(dp4a_uu(pp[j], 0x01010101u, 0x4B000000u)) - 8388608.f;
            }
            a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0);
            a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
            a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1);
            a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
        } else {
            const uint2 xa = *(const uint2*)&xq[a * 16 + r0];
            const uint2 xc = *(const uint2*)&xq[a * 16 + r0 + 8];
            i0 = dp2a_lo(xa.x, pp[0], i0); i2 = dp2a_hi(xa.x, pp[0], i2);
            i1 = dp2a_lo(xa.x, pp[4], i1); i3 = dp2a_hi(xa.x, pp[4], i3);
            i0 = dp2a_lo(xa.y, pp[1], i0); i2 = dp2a_hi(xa.y, pp[1], i2);
            i1 = dp2a_lo(xa.y, pp[5], i1); i3 = dp2a_hi(xa.y, pp[5], i3);
            i0 = dp2a_lo(xc.x, pp[2], i0); i2 = dp2a_hi(xc.x, pp[2], i2);
            i1 = dp2a_lo(xc.x, pp[6], i1); i3 = dp2a_hi(xc.x, pp[6], i3);
            i0 = dp2a_lo(xc.y, pp[3], i0); i2 = dp2a_hi(xc.y, pp[3], i2);
            i1 = dp2a_lo(xc.y, pp[7], i1); i3 = dp2a_hi(xc.y, pp[7], i3);
        }
    }
    if constexpr (MAC <= 1) out[t] = a0 + a1;
    else out[t] = (float)(i0 + i1 + i2 + i3);
}
template __global__ void k_base_k2<0>(const uint32_t*, const float*, const uint32_t*, int, float*);
template __global__ void k_base_k2<1>(const uint32_t*, const float*, const uint32_t*, int, float*);
template __global__ void k_base_k2<2>(const uint32_t*, const float*, const uint32_t*, int, float*);

// ============================================================================
// 2. OPTIMIZED K=2.0 INNER LOOPS (M=32 windows/lane, rotated ring, o=0!)
//    MODE 0: C1c Zero-Centered Exact Magic (0x4B000000 - 8389118.f + FFMA)
//    MODE 1: C3a Max-Scale Folded (0x3F800000 + 255*bs, global fold)
//    MODE 2: C3b Max-Scale Folded (per-tile bias cancel)
//    MODE 3: C4  Zero-Centered IDP4A(-510) + XMAD.S16.S16
// ============================================================================
template <int MODE>
__device__ __forceinline__ void mac_pair_f32(uint32_t U, uint32_t C_mul1,
                                             float x_h1, float x_h0, float& acc_h1, float& acc_h0) {
    uint32_t p1 = mul16_h1(U, C_mul1);
    uint32_t p0 = mul16_h0(U, C_mul1);
    if constexpr (MODE == 0) {
        float f1 = __uint_as_float(dp4a_uu(p1, 0x01010101u, 0x4B000000u)) - 8389118.f;
        float f0 = __uint_as_float(dp4a_uu(p0, 0x01010101u, 0x4B000000u)) - 8389118.f;
        acc_h1 = fmaf(x_h1, f1, acc_h1);
        acc_h0 = fmaf(x_h0, f0, acc_h0);
    } else {
        float f1 = __uint_as_float(dp4a_uu(p1, 0xFFFFFFFFu, 0x3F800000u));
        float f0 = __uint_as_float(dp4a_uu(p0, 0xFFFFFFFFu, 0x3F800000u));
        acc_h1 = fmaf(x_h1, f1, acc_h1);
        acc_h0 = fmaf(x_h0, f0, acc_h0);
    }
}

__device__ __forceinline__ void mac_pair_xmad(uint32_t U, uint32_t C_mul1,
                                              uint32_t xp1, bool hi1, uint32_t xp0, bool hi0,
                                              int& acc1, int& acc0) {
    uint32_t p1 = mul16_h1(U, C_mul1);
    uint32_t p0 = mul16_h0(U, C_mul1);
    int b1 = (int)dp4a_uu(p1, 0x01010101u, (uint32_t)(-510));
    int b0 = (int)dp4a_uu(p0, 0x01010101u, (uint32_t)(-510));
    acc1 = hi1 ? xmad_s16_h1(xp1, b1, acc1) : xmad_s16_h0(xp1, b1, acc1);
    acc0 = hi0 ? xmad_s16_h1(xp0, b0, acc0) : xmad_s16_h0(xp0, b0, acc0);
}

template <int MODE>
__global__ void k_opt_k2_m32(const uint32_t* __restrict__ tb, const float* __restrict__ xs,
                             const float* __restrict__ sxt, const uint32_t* __restrict__ x16p,
                             uint32_t C_mul1, int ntiles, float* __restrict__ out) {
    const int lane = threadIdx.x & 7;
    const int NW = 16;
    const int w0 = lane * 2;
    const int w1 = w0 + 1;
    const int w2 = (w0 + 2 == NW) ? 0 : (w0 + 2);
    float a0 = 0.f, a1 = 0.f;
    int ia0 = 0, ia1 = 0;

    const uint32_t* q = tb;
    const float* xp_f = xs;
    const uint32_t* xp_i = x16p;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a, q += NW, xp_f += 16, xp_i += 8) {
        const uint32_t W0 = __ldg(q + w0);
        const uint32_t W1 = __ldg(q + w1);
        const uint32_t W2 = __ldg(q + w2);

        if constexpr (MODE <= 2) {
            const float4 x0  = *(const float4*)(xp_f + 0);
            const float4 x4  = *(const float4*)(xp_f + 4);
            const float4 x8  = *(const float4*)(xp_f + 8);
            const float4 x12 = *(const float4*)(xp_f + 12);
            float xv[16] = {x0.x,x0.y,x0.z,x0.w, x4.x,x4.y,x4.z,x4.w,
                            x8.x,x8.y,x8.z,x8.w, x12.x,x12.y,x12.z,x12.w};
            float t0 = (MODE == 2) ? sxt[a] : a0;
            float t1 = (MODE == 2) ? sxt[a] : a1;

            mac_pair_f32<MODE>(W0, C_mul1, xv[0], xv[2], t0, t0);
            mac_pair_f32<MODE>(W1, C_mul1, xv[4], xv[6], t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0,  2), C_mul1, xv[1],  xv[3],  t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0,  4), C_mul1, xv[8],  xv[10], t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0,  6), C_mul1, xv[9],  xv[11], t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0,  8), C_mul1, xv[0],  xv[2],  t1, t1);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0, 10), C_mul1, xv[1],  xv[3],  t1, t1);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0, 12), C_mul1, xv[8],  xv[10], t1, t1);
            mac_pair_f32<MODE>(__funnelshift_l(W1, W0, 14), C_mul1, xv[9],  xv[11], t1, t1);

            mac_pair_f32<MODE>(__funnelshift_l(W2, W1,  2), C_mul1, xv[5],  xv[7],  t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W2, W1,  4), C_mul1, xv[12], xv[14], t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W2, W1,  6), C_mul1, xv[13], xv[15], t0, t0);
            mac_pair_f32<MODE>(__funnelshift_l(W2, W1,  8), C_mul1, xv[4],  xv[6],  t1, t1);
            mac_pair_f32<MODE>(__funnelshift_l(W2, W1, 10), C_mul1, xv[5],  xv[7],  t1, t1);
            mac_pair_f32<MODE>(__funnelshift_l(W2, W1, 12), C_mul1, xv[12], xv[14], t1, t1);
            mac_pair_f32<MODE>(__funnelshift_l(W2, W1, 14), C_mul1, xv[13], xv[15], t1, t1);

            if constexpr (MODE == 2) { a0 += t0; a1 += t1; }
            else { a0 = t0; a1 = t1; }
        } else {
            const uint4 xpA = *(const uint4*)(xp_i + 0);
            const uint4 xpB = *(const uint4*)(xp_i + 4);

            mac_pair_xmad(W0, C_mul1, xpA.x, false, xpA.y, false, ia0, ia0);
            mac_pair_xmad(W1, C_mul1, xpA.z, false, xpA.w, false, ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W1, W0,  2), C_mul1, xpA.x, true,  xpA.y, true,  ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W1, W0,  4), C_mul1, xpB.x, false, xpB.y, false, ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W1, W0,  6), C_mul1, xpB.x, true,  xpB.y, true,  ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W1, W0,  8), C_mul1, xpA.x, false, xpA.y, false, ia1, ia1);
            mac_pair_xmad(__funnelshift_l(W1, W0, 10), C_mul1, xpA.x, true,  xpA.y, true,  ia1, ia1);
            mac_pair_xmad(__funnelshift_l(W1, W0, 12), C_mul1, xpB.x, false, xpB.y, false, ia1, ia1);
            mac_pair_xmad(__funnelshift_l(W1, W0, 14), C_mul1, xpB.x, true,  xpB.y, true,  ia1, ia1);

            mac_pair_xmad(__funnelshift_l(W2, W1,  2), C_mul1, xpA.z, true,  xpA.w, true,  ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W2, W1,  4), C_mul1, xpB.z, false, xpB.w, false, ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W2, W1,  6), C_mul1, xpB.z, true,  xpB.w, true,  ia0, ia0);
            mac_pair_xmad(__funnelshift_l(W2, W1,  8), C_mul1, xpA.z, false, xpA.w, false, ia1, ia1);
            mac_pair_xmad(__funnelshift_l(W2, W1, 10), C_mul1, xpA.z, true,  xpA.w, true,  ia1, ia1);
            mac_pair_xmad(__funnelshift_l(W2, W1, 12), C_mul1, xpB.z, false, xpB.w, false, ia1, ia1);
            mac_pair_xmad(__funnelshift_l(W2, W1, 14), C_mul1, xpB.z, true,  xpB.w, true,  ia1, ia1);
        }
    }
    if constexpr (MODE <= 2) out[threadIdx.x] = a0 + a1;
    else out[threadIdx.x] = (float)(ia0 + ia1);
}
template __global__ void k_opt_k2_m32<0>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, float*);
template __global__ void k_opt_k2_m32<1>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, float*);
template __global__ void k_opt_k2_m32<2>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, float*);
template __global__ void k_opt_k2_m32<3>(const uint32_t*, const float*, const float*, const uint32_t*, uint32_t, int, float*);

// ============================================================================
// 3. OPTIMIZED K=3.0 INNER LOOP (M=32, rotated ring, o=0) — shows (D-1) SHF + IADD!
// ============================================================================
__global__ void k_opt_k3_m32_c4(const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16p,
                                uint32_t C_mul1, int ntiles, float* __restrict__ out) {
    const int lane = threadIdx.x & 7;
    const int NW = 24;
    const int w0 = lane * 3;
    int ia0 = 0, ia1 = 0;

    auto do_h1 = [&](uint32_t U, uint32_t xp, bool hi, int& acc) {
        uint32_t p = mul16_h1(U, C_mul1);
        int b = (int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510));
        acc = hi ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
    };
    auto do_h0 = [&](uint32_t U, uint32_t xp, bool hi, int& acc) {
        uint32_t p = mul16_h0(U, C_mul1);
        int b = (int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510));
        acc = hi ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
    };

    const uint32_t* q = tb;
    const uint32_t* xp_i = x16p;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a, q += NW, xp_i += 8) {
        const uint32_t W0 = __ldg(q + w0);
        const uint32_t W1 = __ldg(q + w0 + 1);
        const uint32_t W2 = __ldg(q + w0 + 2);
        const uint32_t W3 = __ldg(q + ((w0 + 3 == NW) ? 0 : (w0 + 3)));
        const uint4 xpA = *(const uint4*)(xp_i + 0);
        const uint4 xpB = *(const uint4*)(xp_i + 4);
        const uint32_t xp[8] = {xpA.x, xpA.y, xpA.z, xpA.w, xpB.x, xpB.y, xpB.z, xpB.w};

        auto mac_j = [&](int j, uint32_t U, bool use_h1) {
            int s = j >> 3, k = j & 7;
            int r = 2 * s + (k & 1) + ((k & 2) ? 8 : 0);
            int& acc = (k & 4) ? ia1 : ia0;
            if (use_h1) do_h1(U, xp[r >> 1], (r & 1) != 0, acc);
            else        do_h0(U, xp[r >> 1], (r & 1) != 0, acc);
        };

        mac_j(0,  W0, true);
        mac_j(16, W1, false);
        { uint32_t U = __funnelshift_l(W1, W0,  2); mac_j(6,  U, false); mac_j(1,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0,  5); mac_j(7,  U, false); mac_j(2,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0,  8); mac_j(8,  U, false); mac_j(3,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0, 11); mac_j(9,  U, false); mac_j(4,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0, 14); mac_j(10, U, false); mac_j(5,  U + U, true); }
        mac_j(11, W1 + W1, true);
        { uint32_t U = __funnelshift_l(W2, W1,  3); mac_j(17, U, false); mac_j(12, U + U, true); }
        { uint32_t U = __funnelshift_l(W2, W1,  6); mac_j(18, U, false); mac_j(13, U + U, true); }
        { uint32_t U = __funnelshift_l(W2, W1,  9); mac_j(19, U, false); mac_j(14, U + U, true); }
        { uint32_t U = __funnelshift_l(W2, W1, 12); mac_j(20, U, false); mac_j(15, U + U, true); }
        { uint32_t U = __funnelshift_l(W2, W1, 30); mac_j(26, U, false); mac_j(21, U + U, true); }
        { uint32_t U = __funnelshift_l(W3, W2,  1); mac_j(27, U, false); mac_j(22, U + U, true); }
        { uint32_t U = __funnelshift_l(W3, W2,  4); mac_j(28, U, false); mac_j(23, U + U, true); }
        { uint32_t U = __funnelshift_l(W3, W2,  7); mac_j(29, U, false); mac_j(24, U + U, true); }
        { uint32_t U = __funnelshift_l(W3, W2, 10); mac_j(30, U, false); mac_j(25, U + U, true); }
        { uint32_t U = __funnelshift_l(W3, W2, 29); mac_j(31, U, true); }
    }
    out[threadIdx.x] = (float)(ia0 + ia1);
}

// ============================================================================
// 4. OPTIMIZED K=4.0 INNER LOOP (M=32, rotated ring, o=0) — 0.375 SHF/weight!
// ============================================================================
__global__ void k_opt_k4_m32_c4(const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16p,
                                uint32_t C_mul1, int ntiles, float* __restrict__ out) {
    const int lane = threadIdx.x & 7;
    const int NW = 32;
    const int w0 = lane * 4;
    int ia0 = 0, ia1 = 0;

    const uint32_t* q = tb;
    const uint32_t* xp_i = x16p;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a, q += NW, xp_i += 8) {
        const uint32_t W0 = __ldg(q + w0);
        const uint32_t W1 = __ldg(q + w0 + 1);
        const uint32_t W2 = __ldg(q + w0 + 2);
        const uint32_t W3 = __ldg(q + w0 + 3);
        const uint32_t W4 = __ldg(q + ((w0 + 4 == NW) ? 0 : (w0 + 4)));
        const uint4 xpA = *(const uint4*)(xp_i + 0);
        const uint4 xpB = *(const uint4*)(xp_i + 4);

        // 4 aligned pairs (W0..W3) = 8 windows with 0 SHF!
        mac_pair_xmad(W0, C_mul1, xpA.x, false, xpA.x, false, ia0, ia1);
        mac_pair_xmad(W1, C_mul1, xpA.y, false, xpA.y, false, ia0, ia1);
        mac_pair_xmad(W2, C_mul1, xpA.z, false, xpA.z, false, ia0, ia1);
        mac_pair_xmad(W3, C_mul1, xpA.w, false, xpA.w, false, ia0, ia1);

        // 12 paired SHFs (3 per word pair: shifts 4, 8, 12) = 24 windows!
        mac_pair_xmad(__funnelshift_l(W1, W0,  4), C_mul1, xpA.x, true,  xpA.x, true,  ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W1, W0,  8), C_mul1, xpB.x, false, xpB.x, false, ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W1, W0, 12), C_mul1, xpB.x, true,  xpB.x, true,  ia0, ia1);

        mac_pair_xmad(__funnelshift_l(W2, W1,  4), C_mul1, xpA.y, true,  xpA.y, true,  ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W2, W1,  8), C_mul1, xpB.y, false, xpB.y, false, ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W2, W1, 12), C_mul1, xpB.y, true,  xpB.y, true,  ia0, ia1);

        mac_pair_xmad(__funnelshift_l(W3, W2,  4), C_mul1, xpA.z, true,  xpA.z, true,  ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W3, W2,  8), C_mul1, xpB.z, false, xpB.z, false, ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W3, W2, 12), C_mul1, xpB.z, true,  xpB.z, true,  ia0, ia1);

        mac_pair_xmad(__funnelshift_l(W4, W3,  4), C_mul1, xpA.w, true,  xpA.w, true,  ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W4, W3,  8), C_mul1, xpB.w, false, xpB.w, false, ia0, ia1);
        mac_pair_xmad(__funnelshift_l(W4, W3, 12), C_mul1, xpB.w, true,  xpB.w, true,  ia0, ia1);
    }
    out[threadIdx.x] = (float)(ia0 + ia1);
}

// ============================================================================
// 4b. OPTIMIZED K=1.5 INNER LOOP (M=32, rotated ring, even/odd lane align)
//     Offsets: 0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45,47
//     2 aligned (0,32), 10 SHF pairs (d,d+16), 5 IADD (U+U for d+1), 5 SHF singletons -> 15 SHF + 5 IADD!
// ============================================================================
__global__ void k_opt_k15_m32_c4(const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16p,
                                 uint32_t C_mul1, int ntiles, float* __restrict__ out) {
    const int lane = threadIdx.x & 7;
    const int NW = 12;
    const int b0 = lane * 48;
    const int w0 = b0 >> 5;
    const int o  = b0 & 31; // 0 for even lanes, 16 for odd lanes
    const int w1 = (w0 + 1 == NW) ? 0 : (w0 + 1);
    const int w2 = (w1 + 1 == NW) ? 0 : (w1 + 1);
    int ia0 = 0, ia1 = 0;

    auto do_h1 = [&](uint32_t U, uint32_t xp, bool hi, int& acc) {
        uint32_t p = mul16_h1(U, C_mul1);
        int b = (int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510));
        acc = hi ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
    };
    auto do_h0 = [&](uint32_t U, uint32_t xp, bool hi, int& acc) {
        uint32_t p = mul16_h0(U, C_mul1);
        int b = (int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510));
        acc = hi ? xmad_s16_h1(xp, b, acc) : xmad_s16_h0(xp, b, acc);
    };

    const uint32_t* q = tb;
    const uint32_t* xp_i = x16p;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a, q += NW, xp_i += 8) {
        const uint32_t raw0 = __ldg(q + w0), raw1 = __ldg(q + w1), raw2 = __ldg(q + w2);
        // o is 0 or 16: 2 align SHFs amortized over 32 weights = 0.0625 SHF/w (0 SHF at M=64!)
        const uint32_t W0 = __funnelshift_l(raw1, raw0, o);
        const uint32_t W1 = __funnelshift_l(raw2, raw1, o);
        const uint4 xpA = *(const uint4*)(xp_i + 0);
        const uint4 xpB = *(const uint4*)(xp_i + 4);
        const uint32_t xp[8] = {xpA.x, xpA.y, xpA.z, xpA.w, xpB.x, xpB.y, xpB.z, xpB.w};

        auto mac_j = [&](int j, uint32_t U, bool use_h1) {
            int s = j >> 3, k = j & 7;
            int r = 2 * s + (k & 1) + ((k & 2) ? 8 : 0);
            int& acc = (k & 4) ? ia1 : ia0;
            if (use_h1) do_h1(U, xp[r >> 1], (r & 1) != 0, acc);
            else        do_h0(U, xp[r >> 1], (r & 1) != 0, acc);
        };

        // 2 aligned windows (0 ops): j=0 (D=0 -> W0.H1), j=21 (D=32 -> W1.H1)
        mac_j(0,  W0, true);
        mac_j(21, W1, true);
        // 5 triple-covered groups: 1 SHF(d) -> (d, d+16) + 1 IADD(U+U) -> (d+1) = 3 windows in 1 SHF + 1 IADD!
        { uint32_t U = __funnelshift_l(W1, W0,  2); mac_j(1,  U, true); mac_j(12, U, false); mac_j(2,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0,  5); mac_j(3,  U, true); mac_j(14, U, false); mac_j(4,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0,  8); mac_j(5,  U, true); mac_j(16, U, false); mac_j(6,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0, 11); mac_j(7,  U, true); mac_j(18, U, false); mac_j(8,  U + U, true); }
        { uint32_t U = __funnelshift_l(W1, W0, 14); mac_j(9,  U, true); mac_j(20, U, false); mac_j(10, U + U, true); }
        // 5 paired SHFs (d, d+16) for d=17,20,23,26,29 -> (11,22), (13,24), (15,26), (17,28), (19,30)
        { uint32_t U = __funnelshift_l(W1, W0, 17); mac_j(11, U, true); mac_j(22, U, false); }
        { uint32_t U = __funnelshift_l(W1, W0, 20); mac_j(13, U, true); mac_j(24, U, false); }
        { uint32_t U = __funnelshift_l(W1, W0, 23); mac_j(15, U, true); mac_j(26, U, false); }
        { uint32_t U = __funnelshift_l(W1, W0, 26); mac_j(17, U, true); mac_j(28, U, false); }
        { uint32_t U = __funnelshift_l(W1, W0, 29); mac_j(19, U, true); mac_j(30, U, false); }
        // 5 remaining tail singletons in W1 (D=35,38,41,44,47 -> shifts 3,6,9,12,15 of W1)
        mac_j(23, W1 <<  3, true);
        mac_j(25, W1 <<  6, true);
        mac_j(27, W1 <<  9, true);
        mac_j(29, W1 << 12, true);
        mac_j(31, W1 << 15, true);
    }
    out[threadIdx.x] = (float)(ia0 + ia1);
}

// ============================================================================
// 5. NEGATIVE TESTS (exact SASS proof of why each fails)
// ============================================================================

// B2 Negative Test: Incremental update P_{i+1} = (P_i << 2) + v*C - u*(CL<<16) for K=2.0
__global__ void k_fail_incr_k2(const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16p,
                               uint32_t C_mul1, int ntiles, float* __restrict__ out) {
    const int lane = threadIdx.x & 7;
    int ia0 = 0, ia1 = 0;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a) {
        const uint32_t* q = tb + a * 16 + lane * 2;
        uint32_t W0 = __ldg(q), W1 = __ldg(q + 1);
        const uint4 xpA = *(const uint4*)&x16p[a * 8 + 0];
        const uint4 xpB = *(const uint4*)&x16p[a * 8 + 4];
        uint32_t P = mul16_h1(W0, C_mul1);
        int b = (int)dp4a_uu(P, 0x01010101u, (uint32_t)(-510));
        ia0 = xmad_s16_h0(xpA.x, b, ia0);
        uint32_t stream_u = W0;
        uint32_t stream_v = (W0 << 16) | (W1 >> 16);
#pragma unroll
        for (int j = 1; j < 8; ++j) {
            uint32_t u = stream_u >> 30;
            uint32_t v = stream_v >> 30;
            stream_u <<= 2;
            stream_v <<= 2;
            P = (P << 2) + v * C_mul1 - (u * 0xD12Du << 16);
            int bj = (int)dp4a_uu(P, 0x01010101u, (uint32_t)(-510));
            if (j < 4) ia0 = xmad_s16_h1(xpA.x, bj, ia0);
            else       ia1 = xmad_s16_h0(xpB.x, bj, ia1);
        }
    }
    out[threadIdx.x] = (float)(ia0 + ia1);
}

// A4 Negative Test: XMAD-based bit-shift extraction (8 windows, K=2.0)
__global__ void k_fail_xmad_shift(const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16p,
                                  uint32_t C_mul1, int ntiles, float* __restrict__ out) {
    const int lane = threadIdx.x & 7;
    int ia0 = 0, ia1 = 0;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a) {
        uint32_t hi = __ldg(tb + a * 16 + lane * 2);
        const uint4 xpA = *(const uint4*)&x16p[a * 8 + 0];
        uint32_t p0 = mul16_h1(hi, C_mul1);
        ia0 = xmad_s16_h0(xpA.x, (int)dp4a_uu(p0, 0x01010101u, (uint32_t)(-510)), ia0);
#pragma unroll
        for (int j = 1; j < 8; ++j) {
            const uint32_t mul = 1u << (2 * j);
            uint32_t t = (hi & 0xFFFFu) * mul;
            uint32_t u = (hi >> 16) * mul + (t >> 16);
            uint32_t p = mul16_h0(u, C_mul1);
            int b = (int)dp4a_uu(p, 0x01010101u, (uint32_t)(-510));
            if (j < 4) ia0 = xmad_s16_h1(xpA.x, b, ia0);
            else       ia1 = xmad_s16_h0(xpA.y, b, ia1);
        }
    }
    out[threadIdx.x] = (float)(ia0 + ia1);
}

// B3 Negative Test: Byte-shared decomposition P = (Q[A] << 8) + Q[B] via 1KB smem LUT
__global__ void k_fail_byte_share(const uint32_t* __restrict__ tb, const uint32_t* __restrict__ x16p,
                                  const uint32_t* __restrict__ Q_in, int ntiles, float* __restrict__ out) {
    __shared__ uint32_t Q[256];
    if (threadIdx.x < 256) Q[threadIdx.x] = Q_in[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 7;
    int ia0 = 0, ia1 = 0;
#pragma unroll 1
    for (int a = 0; a < ntiles; ++a) {
        const uint32_t* q = tb + a * 16 + lane * 2;
        uint32_t W0 = __ldg(q), W1 = __ldg(q + 1);
        const uint4 xpA = *(const uint4*)&x16p[a * 8 + 0];
        // 4 consecutive byte-aligned windows (D=0,8,16,24) sharing 5 bytes (b0..b4)
        uint32_t q0 = Q[(W0 >> 24) & 0xFFu];
        uint32_t q1 = Q[(W0 >> 16) & 0xFFu];
        uint32_t q2 = Q[(W0 >>  8) & 0xFFu];
        uint32_t q3 = Q[(W0 >>  0) & 0xFFu];
        uint32_t q4 = Q[(W1 >> 24) & 0xFFu];
        uint32_t p0 = (q0 << 8) + q1;
        uint32_t p1 = (q1 << 8) + q2;
        uint32_t p2 = (q2 << 8) + q3;
        uint32_t p3 = (q3 << 8) + q4;
        ia0 = xmad_s16_h0(xpA.x, (int)dp4a_uu(p0, 0x01010101u, (uint32_t)(-510)), ia0);
        ia0 = xmad_s16_h1(xpA.x, (int)dp4a_uu(p1, 0x01010101u, (uint32_t)(-510)), ia0);
        ia1 = xmad_s16_h0(xpA.y, (int)dp4a_uu(p2, 0x01010101u, (uint32_t)(-510)), ia1);
        ia1 = xmad_s16_h1(xpA.y, (int)dp4a_uu(p3, 0x01010101u, (uint32_t)(-510)), ia1);
    }
    out[threadIdx.x] = (float)(ia0 + ia1);
}
