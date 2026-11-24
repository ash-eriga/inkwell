// kbench: "stream-decode" EXL3 MoE GEMV vs the serial kernel (inkwell_exl3_gemv_moe).
//
// Idea: one warp owns one 16x16 tile column (tn) and walks tk. Lane t owns ring
// positions 8t..8t+7, which are 8 CONSECUTIVE windows: their bit offsets relative
// to window 8t are compile-time (j*KA + H*((j+1)>>1)). Each lane loads 3 trellis
// words straight from global (no smem ring, no __syncthreads), aligns them once
// into a 64-bit register stream with 2 SHF, then every window is 1 SHF with an
// immediate shift, a 16x32 multiply (XMAD), and the MAC. Σx is precomputed.
//
// MAC variants:
//   0 f32-i2f : bs=dp4a(p,1s); acc=fma(x,(float)bs)                 exact
//   1 f32-mag : f=as_float(dp4a(p,1s,0x4B000000))-2^23; acc=fma(x,f) exact
//   2 dp2a    : x as int16 (scale per 128 k); dp2a.lo+dp2a.hi       ~15-bit x
//   3 dp4a    : x as int8  (scale per 128 k); 1 dp4a splat           ~7-bit x
//   4 f32-i2f + float4 x (permuted xs) unroll4                        exact
//   5 f32-mag + float4 x unroll8                                     exact
//   6 f32-mag + float4 x unroll8 + next-tile prefetch                 exact
//
// build+run: bash opt/kbench/build.sh sgemv_bench && opt/kbench/sgemv_bench
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

namespace sg {

__device__ __forceinline__ uint32_t mul16(uint32_t w) {   // (w & 0xffff) * 0x83DCD12D mod 2^32
    const uint32_t h = w & 0xffffu;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}
__device__ __forceinline__ int dp2lo(uint32_t a, uint32_t b, int c) {
    int d; asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ int dp2hi(uint32_t a, uint32_t b, int c) {
    int d; asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}
__device__ __forceinline__ int dp4su(uint32_t a, uint32_t b, int c) {   // a signed int8 x4, b unsigned bytes
    int d; asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}

// window j (16 bits, MSB-first) out of the 64-bit stream hi:lo; offset compile-time
template <int D> __device__ __forceinline__ uint32_t win(uint32_t hi, uint32_t lo) {
    if constexpr (D <= 16) return hi >> (16 - D);
    else return __funnelshift_r(lo, hi, 48 - D);
}

constexpr int MAXK = 2304;

// MAC=0/1: xs = float x; MAC=2: xq = int16 splat (x16|x16<<16); MAC=3: xq = int8 splat x4
template <int KA, int H, int MAC>
__device__ __forceinline__ void run(const uint32_t* __restrict__ tb, int tn, int tn_i, int tk_lo, int tk_hi,
                                    const float* xs, const uint32_t* xq, const float* qs,
                                    float& a0, float& a1, const float* xg = nullptr) {
    constexpr int NW = 8 * KA + 4 * H;
    const int t = threadIdx.x & 31;
    const int R = 32 * NW;
    int B = (H ? (8 * t + 1) * KA + ((8 * t + 1) >> 1) : (8 * t + 1) * KA) - 16;
    if (B < 0) B += R;
    const int w0 = B >> 5, o = B & 31;
    const int w1 = (w0 + 1 == NW) ? 0 : w0 + 1;
    const int w2 = (w1 + 1 == NW) ? 0 : w1 + 1;
    const int r0 = 2 * (t & 3);
    const size_t stride = (size_t)tn * NW;
    const uint32_t* p = tb + (size_t)tn_i * NW;
    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;   // int accumulators (MAC 2/3)
    auto tile = [&](const uint32_t* __restrict__ q, int kb) {
        const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
        const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
        uint32_t pp[8];
#define SG_D(j) ((j) * KA + H * (((j) + 1) >> 1))
        pp[0] = mul16(win<SG_D(0)>(hi, lo)); pp[1] = mul16(win<SG_D(1)>(hi, lo));
        pp[2] = mul16(win<SG_D(2)>(hi, lo)); pp[3] = mul16(win<SG_D(3)>(hi, lo));
        pp[4] = mul16(win<SG_D(4)>(hi, lo)); pp[5] = mul16(win<SG_D(5)>(hi, lo));
        pp[6] = mul16(win<SG_D(6)>(hi, lo)); pp[7] = mul16(win<SG_D(7)>(hi, lo));
#undef SG_D
                if constexpr (MAC <= 1) {
            const float2 xa = *(const float2*)&xs[kb], xc = *(const float2*)&xs[kb + 8];
            float f[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (MAC == 0) f[j] = (float)(int)__dp4a(pp[j], 0x01010101u, 0u);
                else f[j] = __int_as_float((int)__dp4a(pp[j], 0x01010101u, 0x4B000000u)) - 8388608.f;
            }
            a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0); a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
            a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1); a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
        } else {
            const uint2 xa = *(const uint2*)&xq[kb], xc = *(const uint2*)&xq[kb + 8];
            if constexpr (MAC == 2) {   // 4 independent chains (2 per column)
                i0 = dp2lo(xa.x, pp[0], i0); i2 = dp2hi(xa.x, pp[0], i2);
                i1 = dp2lo(xa.x, pp[4], i1); i3 = dp2hi(xa.x, pp[4], i3);
                i0 = dp2lo(xa.y, pp[1], i0); i2 = dp2hi(xa.y, pp[1], i2);
                i1 = dp2lo(xa.y, pp[5], i1); i3 = dp2hi(xa.y, pp[5], i3);
                i0 = dp2lo(xc.x, pp[2], i0); i2 = dp2hi(xc.x, pp[2], i2);
                i1 = dp2lo(xc.x, pp[6], i1); i3 = dp2hi(xc.x, pp[6], i3);
                i0 = dp2lo(xc.y, pp[3], i0); i2 = dp2hi(xc.y, pp[3], i2);
                i1 = dp2lo(xc.y, pp[7], i1); i3 = dp2hi(xc.y, pp[7], i3);
            } else {
                i0 = dp4su(xa.x, pp[0], i0); i2 = dp4su(xa.y, pp[1], i2);
                i1 = dp4su(xa.x, pp[4], i1); i3 = dp4su(xa.y, pp[5], i3);
                i0 = dp4su(xc.x, pp[2], i0); i2 = dp4su(xc.y, pp[3], i2);
                i1 = dp4su(xc.x, pp[6], i1); i3 = dp4su(xc.y, pp[7], i3);
            }
        }
    };
    if constexpr (MAC <= 1) {   // f32: plain per-tile loop, unroll 4 (best measured: 201/103 us)
#pragma unroll 4
        for (int tk = tk_lo; tk < tk_hi; ++tk) tile(p + (size_t)tk * stride, tk * 16 + r0);
        return;
    }
    if constexpr (MAC >= 4) {   // f32 + float4 x (permuted xs)
        const float4* xs4 = (const float4*)xs;
        const int x4 = t & 3;
        if constexpr (MAC == 12) {   // x straight from global, no smem
#pragma unroll 4
            for (int a = tk_lo; a < tk_hi; ++a) {
                const uint32_t* q = p + (size_t)a * stride;
                const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
                const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
                uint32_t pp[8];
#define SG_D3(j) ((j) * KA + H * (((j) + 1) >> 1))
                pp[0] = mul16(win<SG_D3(0)>(hi, lo)); pp[1] = mul16(win<SG_D3(1)>(hi, lo));
                pp[2] = mul16(win<SG_D3(2)>(hi, lo)); pp[3] = mul16(win<SG_D3(3)>(hi, lo));
                pp[4] = mul16(win<SG_D3(4)>(hi, lo)); pp[5] = mul16(win<SG_D3(5)>(hi, lo));
                pp[6] = mul16(win<SG_D3(6)>(hi, lo)); pp[7] = mul16(win<SG_D3(7)>(hi, lo));
#undef SG_D3
                const int kb = a * 16 + r0;
                const float2 xa = __ldg((const float2*)&xg[kb]), xc = __ldg((const float2*)&xg[kb + 8]);
                float f[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = (float)(int)__dp4a(pp[j], 0x01010101u, 0u);
                a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0); a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
                a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1); a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
            }
            return;
        }
        auto tile4 = [&](const uint32_t* __restrict__ q, uint32_t W0, uint32_t W1, uint32_t W2, int a) {
            const float4 xv0 = (MAC == 10) ? xs4[a * 4 + x4] : make_float4(0,0,0,0);
            const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
            uint32_t pp[8];
#define SG_D2(j) ((j) * KA + H * (((j) + 1) >> 1))
            pp[0] = mul16(win<SG_D2(0)>(hi, lo)); pp[1] = mul16(win<SG_D2(1)>(hi, lo));
            pp[2] = mul16(win<SG_D2(2)>(hi, lo)); pp[3] = mul16(win<SG_D2(3)>(hi, lo));
            pp[4] = mul16(win<SG_D2(4)>(hi, lo)); pp[5] = mul16(win<SG_D2(5)>(hi, lo));
            pp[6] = mul16(win<SG_D2(6)>(hi, lo)); pp[7] = mul16(win<SG_D2(7)>(hi, lo));
#undef SG_D2
            const float4 xv = (MAC == 10) ? xv0 : xs4[a * 4 + x4];
            float f[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if constexpr (MAC != 5 && MAC != 6) f[j] = (float)(int)__dp4a(pp[j], 0x01010101u, 0u);
                else f[j] = __int_as_float((int)__dp4a(pp[j], 0x01010101u, 0x4B000000u)) - 8388608.f;
            }
            a0 = fmaf(xv.x, f[0], a0); a0 = fmaf(xv.y, f[1], a0); a0 = fmaf(xv.z, f[2], a0); a0 = fmaf(xv.w, f[3], a0);
            a1 = fmaf(xv.x, f[4], a1); a1 = fmaf(xv.y, f[5], a1); a1 = fmaf(xv.z, f[6], a1); a1 = fmaf(xv.w, f[7], a1);
        };
        const uint32_t* q = p + (size_t)tk_lo * stride;
        if constexpr (MAC == 6) {   // prefetch one tile ahead, unroll 4
            uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
#pragma unroll 4
            for (int a = tk_lo; a < tk_hi; ++a) {
                const uint32_t* qn = q + stride;
                uint32_t N0 = 0, N1 = 0, N2 = 0;
                if (a + 1 < tk_hi) { N0 = __ldg(qn + w0); N1 = __ldg(qn + w1); N2 = __ldg(qn + w2); }
                tile4(q, W0, W1, W2, a);
                W0 = N0; W1 = N1; W2 = N2; q = qn;
            }
            return;
        }
        if constexpr (MAC == 5) {
#pragma unroll 8
            for (int a = tk_lo; a < tk_hi; ++a, q += stride)
                tile4(q, __ldg(q + w0), __ldg(q + w1), __ldg(q + w2), a);
        } else if constexpr (MAC == 11) {
#pragma unroll 3
            for (int a = tk_lo; a < tk_hi; ++a, q += stride)
                tile4(q, __ldg(q + w0), __ldg(q + w1), __ldg(q + w2), a);
        } else if constexpr (MAC == 7) {
#pragma unroll 2
            for (int a = tk_lo; a < tk_hi; ++a, q += stride)
                tile4(q, __ldg(q + w0), __ldg(q + w1), __ldg(q + w2), a);
        } else if constexpr (MAC == 8) {
#pragma unroll 1
            for (int a = tk_lo; a < tk_hi; ++a, q += stride)
                tile4(q, __ldg(q + w0), __ldg(q + w1), __ldg(q + w2), a);
        } else {
#pragma unroll 4
            for (int a = tk_lo; a < tk_hi; ++a, q += stride)
                tile4(q, __ldg(q + w0), __ldg(q + w1), __ldg(q + w2), a);
        }
        return;
    }
    const uint32_t* q = p + (size_t)tk_lo * stride;
    for (int tk = tk_lo; tk < tk_hi;) {   // int: 8-tile blocks, flush per 128-k scale block
        const int nb = min(8 - (tk & 7), tk_hi - tk);   // tiles up to the next 128-k scale block
        if (nb == 8) {
#pragma unroll
            for (int u = 0; u < 8; ++u) { tile(q, (tk + u) * 16 + r0); q += stride; }
        } else {
            for (int u = 0; u < nb; ++u) { tile(q, (tk + u) * 16 + r0); q += stride; }
        }
        if constexpr (MAC >= 2) {
            const float s = qs[tk >> 3];
            a0 = fmaf(s, (float)(i0 + i2), a0); a1 = fmaf(s, (float)(i1 + i3), a1); i0 = i1 = i2 = i3 = 0;
        }
        tk += nb;
    }
}

// grid (ceil(tn/8), nslots, nsel*nky), 256 threads = 8 warps = 8 tn tiles of one slot.
// descs[e*4]: d[0]=trellis ptr, d[3]=ka|halfk<<8. Output t[slot*n + zsel*tzoff + col] (atomic if nky>1).
// sxg[slot*nsel+zsel][tk>>3]: Σx per 128-k block (precomputed; in the real pipeline had_in emits it).
template <int MAC>
__global__ void __launch_bounds__(256) gemv_moe_s(const uint64_t* __restrict__ descs,
                                                  const uint64_t* __restrict__ descs2,
                                                  const int* __restrict__ ids,
                                                  const float* __restrict__ xh, const float* __restrict__ xh2,
                                                  const uint32_t* __restrict__ xqg, const float* __restrict__ qsg,
                                                  float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, int nky) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1;
    const int ky = blockIdx.z / nsel, zsel = blockIdx.z % nsel;
    const uint64_t* d = (zsel ? descs2 : descs) + ids[slot] * 4;
    const int js = slot * nsel + zsel;
    const int tk_lo = (ky * tk) / nky, tk_hi = ((ky + 1) * tk) / nky;
    __shared__ __align__(16) uint32_t xbuf[MAXK];
    float* xs = (float*)xbuf; uint32_t* xq = xbuf;
    __shared__ float qs[MAXK / 128];
    __shared__ float red[8];
    const float* xg = (zsel ? xh2 : xh) + (size_t)js / nsel * k;
    float sxl = 0.f;
    for (int e = tk_lo * 16 + threadIdx.x; e < tk_hi * 16; e += 256) {
        const float v = xg[e];
        sxl += v;
        if (MAC <= 1) xs[e] = v;
        else if (MAC >= 4) { const int r = e & 15; xs[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3)] = v; }
        else if (MAC == 12) { }  // no smem x at all
        else xq[e] = xqg[(size_t)js * k + e];
    }
    if ((MAC == 2 || MAC == 3) && threadIdx.x < k / 128) qs[threadIdx.x] = qsg[(size_t)js * (k / 128) + threadIdx.x];
#pragma unroll
    for (int s = 16; s; s >>= 1) sxl += __shfl_xor_sync(0xffffffffu, sxl, s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sxl;
    __syncthreads();
    float sx = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx += red[i];
    const int tn_i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tn_i >= tn) return;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff), hk = (int)((d[3] >> 8) & 1);
    float a0 = 0.f, a1 = 0.f;
    switch (ka * 2 + hk) {
        case 3:  run<1, 1, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        case 4:  run<2, 0, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        case 5:  run<2, 1, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        case 6:  run<3, 0, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        case 7:  run<3, 1, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        case 8:  run<4, 0, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        case 10: run<5, 0, MAC>(tb, tn, tn_i, tk_lo, tk_hi, xs, xq, qs, a0, a1, xg); break;
        default: __trap();
    }
    a0 += __shfl_xor_sync(0xffffffffu, a0, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    a0 += __shfl_xor_sync(0xffffffffu, a0, 2); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    const int lane = threadIdx.x & 31;
    if ((lane & 3) == 0) {
        const float KINV = __half2float(__ushort_as_half(0x1eee));
        const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
        const float off = KOFF * sx;   // whole-k-range Σx belongs to every column once
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (lane >> 2)];
        const float v0 = fmaf(KINV, a0, off), v1 = fmaf(KINV, a1, off);
        if (nky > 1) { atomicAdd(dst, v0); atomicAdd(dst + 8, v1); } else { dst[0] = v0; dst[8] = v1; }
    }
}

// same body as gemv_moe_s<4> but capped at 6 blocks/SM for occupancy A/B
__global__ void __launch_bounds__(256, 6) gemv_moe_s_b6(const uint64_t* __restrict__ descs,
                                                  const uint64_t* __restrict__ descs2,
                                                  const int* __restrict__ ids,
                                                  const float* __restrict__ xh, const float* __restrict__ xh2,
                                                  const uint32_t* __restrict__ xqg, const float* __restrict__ qsg,
                                                  float* __restrict__ t, int tzoff, int tk, int tn, int k, int n, int nky) {
    const int slot = blockIdx.y;
    const int nsel = descs2 ? 2 : 1;
    const int ky = blockIdx.z / nsel, zsel = blockIdx.z % nsel;
    const uint64_t* d = (zsel ? descs2 : descs) + ids[slot] * 4;
    const int js = slot * nsel + zsel;
    const int tk_lo = (ky * tk) / nky, tk_hi = ((ky + 1) * tk) / nky;
    __shared__ __align__(16) uint32_t xbuf[MAXK];
    float* xs = (float*)xbuf;
    __shared__ float red[8];
    const float* xg = (zsel ? xh2 : xh) + (size_t)js / nsel * k;
    float sxl = 0.f;
    for (int e = tk_lo * 16 + threadIdx.x; e < tk_hi * 16; e += 256) {
        const float v = xg[e];
        sxl += v;
        const int r = e & 15; xs[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3)] = v;
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) sxl += __shfl_xor_sync(0xffffffffu, sxl, s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sxl;
    __syncthreads();
    float sx = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx += red[i];
    const int tn_i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tn_i >= tn) return;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff), hk = (int)((d[3] >> 8) & 1);
    float a0 = 0.f, a1 = 0.f;
    switch (ka * 2 + hk) {
        case 3:  run<1, 1, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        case 4:  run<2, 0, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        case 5:  run<2, 1, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        case 6:  run<3, 0, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        case 7:  run<3, 1, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        case 8:  run<4, 0, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        case 10: run<5, 0, 4>(tb, tn, tn_i, tk_lo, tk_hi, xs, (const uint32_t*)nullptr, (const float*)nullptr, a0, a1); break;
        default: __trap();
    }
    a0 += __shfl_xor_sync(0xffffffffu, a0, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    a0 += __shfl_xor_sync(0xffffffffu, a0, 2); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    const int lane = threadIdx.x & 31;
    if ((lane & 3) == 0) {
        const float KINV = __half2float(__ushort_as_half(0x1eee));
        const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
        const float off = KOFF * sx;
        float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (lane >> 2)];
        const float v0 = fmaf(KINV, a0, off), v1 = fmaf(KINV, a1, off);
        if (nky > 1) { atomicAdd(dst, v0); atomicAdd(dst + 8, v1); } else { dst[0] = v0; dst[8] = v1; }
    }
}

}  // namespace sg

// ---------------- host ----------------
static uint32_t rs = 12345u;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; float v = e ? ldexpf(1.f + m / 1024.f, e - 15) : ldexpf(m / 1024.f, -14); return (h & 0x8000) ? -v : v; }

struct KH { int ka, hk; };
static int nwords(KH k) { return 8 * k.ka + 4 * k.hk; }

// CPU ref straight from the validated extractor formula (kernels.cu exl3_window) + tile_perm
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

template <typename F> static float timeit(F&& f, int reps = 31) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) f();
    CK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// one MoE stage: nslots experts, GU fused pair (k=2304 -> 2x1152) or down (1152 -> 2304)
struct Stage {
    const char* name; int k, nout, nsel;
};

int main(int argc, char** argv) {
    const int nslots = 8;
    const int nky_base = 2;
    const int nky_new = argc > 1 ? atoi(argv[1]) : 1;
    // K mix of the real SAGE experts (gate/up and down histograms from the safetensors header)
    const KH khs[7] = {{1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {4, 0}, {5, 0}};
    const int gu_w[7] = {989, 1274, 450, 227, 64, 29, 0}, dn_w[7] = {542, 1151, 570, 450, 227, 93, 0};
    Stage st[2] = {{"GU  2304->2x1152", 2304, 1152, 2}, {"down 1152->2304 ", 1152, 2304, 1}};
    printf("nslots=%d  base=inkwell_exl3_gemv_moe nky=%d  new nky=%d\n", nslots, nky_base, nky_new);
    printf("%-18s %-8s %9s %9s  %8s  %s\n", "stage", "kernel", "us", "Gw/s", "speedup", "max|err|/max|y|");
    double tot_b = 0, tot_n[13] = {};
    for (int si = 0; si < 2; ++si) {
        const Stage S = st[si];
        const int tk = S.k / 16, tn = S.nout / 16, njobs = nslots * S.nsel;
        // per job: K drawn from the histogram
        std::vector<KH> jk(njobs);
        const int* wts = si == 0 ? gu_w : dn_w;
        int wsum = 0; for (int i = 0; i < 7; ++i) wsum += wts[i];
        for (int j = 0; j < njobs; ++j) {
            int r = rnd() % wsum, c = 0; while (r >= wts[c]) r -= wts[c++];
            jk[j] = khs[c];
        }
        if (si == 0) jk[1] = {5, 0};   // make sure every geometry is exercised once
        std::vector<std::vector<uint32_t>> tr(njobs);
        std::vector<uint64_t> d1(nslots * 4, 0), d2(nslots * 4, 0);
        std::vector<uint32_t*> dtr(njobs);
        for (int j = 0; j < njobs; ++j) {
            tr[j].resize((size_t)tk * tn * nwords(jk[j]));
            for (auto& v : tr[j]) v = rnd();
            CK(cudaMalloc(&dtr[j], tr[j].size() * 4));
            CK(cudaMemcpy(dtr[j], tr[j].data(), tr[j].size() * 4, cudaMemcpyHostToDevice));
            const int s = j / S.nsel, z = j % S.nsel;
            uint64_t* d = (z ? d2 : d1).data() + s * 4;
            d[0] = (uint64_t)(uintptr_t)dtr[j]; d[3] = (uint64_t)(jk[j].ka | (jk[j].hk << 8));
        }
        // x per job (Hadamard-like: gaussian-ish, a few outliers), plus int16/int8 splats with per-128 scale
        std::vector<float> x((size_t)njobs * S.k);
        for (auto& v : x) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
        for (int j = 0; j < njobs; ++j) x[(size_t)j * S.k + (rnd() % S.k)] *= 6.f;
        std::vector<uint32_t> x16(x.size()), x8(x.size());
        std::vector<float> q16((size_t)njobs * S.k / 128), q8(q16.size());
        for (size_t b = 0; b < q16.size(); ++b) {
            float am = 1e-30f; for (int i = 0; i < 128; ++i) am = fmaxf(am, fabsf(x[b * 128 + i]));
            q16[b] = am / 32767.f; q8[b] = am / 127.f;
            for (int i = 0; i < 128; ++i) {
                const int a = (int)lrintf(x[b * 128 + i] / q16[b]), c = (int)lrintf(x[b * 128 + i] / q8[b]);
                x16[b * 128 + i] = (uint32_t)(a & 0xffff) * 0x00010001u;
                x8[b * 128 + i] = (uint32_t)(c & 0xff) * 0x01010101u;
            }
        }
        // GU base kernel reads xh[slot*k] for gate, xh2[slot*xh2s] for up; new kernel uses the same split
        std::vector<float> xg((size_t)nslots * S.k), xu((size_t)nslots * S.k);
        for (int s = 0; s < nslots; ++s)
            for (int e = 0; e < S.k; ++e) {
                xg[(size_t)s * S.k + e] = x[(size_t)(s * S.nsel) * S.k + e];
                if (S.nsel == 2) xu[(size_t)s * S.k + e] = x[(size_t)(s * 2 + 1) * S.k + e];
            }
        // CPU ref
        const int n = S.nout * S.nsel;
        std::vector<double> ref((size_t)nslots * n);
        for (int j = 0; j < njobs; ++j) {
            const int s = j / S.nsel, z = j % S.nsel;
            ref_gemv(tr[j].data(), jk[j], &x[(size_t)j * S.k], tk, tn, &ref[(size_t)s * n + z * S.nout]);
        }
        double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));
        uint64_t *dd1, *dd2 = nullptr; int* dids; float *dxg, *dxu, *dt, *dq16, *dq8; uint32_t *dx16, *dx8;
        CK(cudaMalloc(&dd1, d1.size() * 8)); CK(cudaMemcpy(dd1, d1.data(), d1.size() * 8, cudaMemcpyHostToDevice));
        if (S.nsel == 2) { CK(cudaMalloc(&dd2, d2.size() * 8)); CK(cudaMemcpy(dd2, d2.data(), d2.size() * 8, cudaMemcpyHostToDevice)); }
        std::vector<int> ids(nslots); for (int i = 0; i < nslots; ++i) ids[i] = i;
        CK(cudaMalloc(&dids, nslots * 4)); CK(cudaMemcpy(dids, ids.data(), nslots * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxg, xg.size() * 4)); CK(cudaMemcpy(dxg, xg.data(), xg.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxu, xu.size() * 4)); CK(cudaMemcpy(dxu, xu.data(), xu.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx16, x16.size() * 4)); CK(cudaMemcpy(dx16, x16.data(), x16.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dx8, x8.size() * 4)); CK(cudaMemcpy(dx8, x8.data(), x8.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dq16, q16.size() * 4)); CK(cudaMemcpy(dq16, q16.data(), q16.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dq8, q8.size() * 4)); CK(cudaMemcpy(dq8, q8.data(), q8.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dt, (size_t)nslots * n * 4));
        std::vector<float> out((size_t)nslots * n);
        auto err = [&]() {
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(out.data(), dt, out.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0; for (size_t i = 0; i < out.size(); ++i) e = std::max(e, fabs(out[i] - ref[i]));
            return e / ymax;
        };
        const double nweights = (double)njobs * S.k * S.nout;
        auto base = [&]() {
            CK(cudaMemsetAsync(dt, 0, (size_t)nslots * n * 4));
            inkwell_exl3_gemv_moe<<<dim3(tn / 4, nslots, S.nsel * nky_base), 256>>>(dd1, dd2, dids, dxg, dxu, S.k, dt, n,
                S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_base, nullptr, 0);
        };
        base(); const double eb = err(); const float ub = timeit(base) * 1e3f;
        printf("%-18s %-8s %9.1f %9.1f  %8s  %.2e\n", S.name, "base", ub, nweights / ub / 1e3, "1.00x", eb);
        tot_b += ub;
        const char* mn[13] = {"f32-i2f", "f32-mag", "dp2a-16", "dp4a-8", "i2f-x4", "mag-x4u8", "mag-pf", "x4-u2", "x4-u1", "x4-b6", "x4-ord", "x4-u3", "x-glob"};
        for (int m = 0; m < 13; ++m) {
            auto nw = [&]() {
                if (nky_new > 1) CK(cudaMemsetAsync(dt, 0, (size_t)nslots * n * 4));
                const dim3 g((tn + 7) / 8, nslots, S.nsel * nky_new);
                const uint32_t* xq = m == 2 ? dx16 : dx8; const float* qs = m == 2 ? dq16 : dq8;
                if (m == 0) sg::gemv_moe_s<0><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 1) sg::gemv_moe_s<1><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 2) sg::gemv_moe_s<2><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 3) sg::gemv_moe_s<3><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 4) sg::gemv_moe_s<4><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 5) sg::gemv_moe_s<5><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 6) sg::gemv_moe_s<6><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 7) sg::gemv_moe_s<7><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 8) sg::gemv_moe_s<8><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 9) sg::gemv_moe_s_b6<<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 10) sg::gemv_moe_s<10><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 11) sg::gemv_moe_s<11><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
                if (m == 12) sg::gemv_moe_s<12><<<g, 256>>>(dd1, dd2, dids, dxg, dxu, xq, qs, dt, S.nsel == 2 ? S.nout : 0, tk, tn, S.k, n, nky_new);
            };
            CK(cudaMemset(dt, 0, (size_t)nslots * n * 4));
            nw(); const double en = err(); const float un = timeit(nw) * 1e3f;
            printf("%-18s %-8s %9.1f %9.1f  %7.2fx  %.2e\n", S.name, mn[m], un, nweights / un / 1e3, ub / un, en);
            tot_n[m] += un;
        }
    }
    printf("\nper MoE visit (GU+down gemvs): base %.1f us", tot_b);
    const char* mn[13] = {"f32-i2f", "f32-mag", "dp2a-16", "dp4a-8", "i2f-x4", "mag-x4u8", "mag-pf", "x4-u2", "x4-u1", "x4-b6", "x4-ord", "x4-u3", "x-glob"};
    for (int m = 0; m < 13; ++m) printf(" | %s %.1f us", mn[m], tot_n[m]);
    printf("\nx48 visits/token: base %.1f ms", tot_b * 48e-3);
    for (int m = 0; m < 13; ++m) printf(" | %s %.1f ms", mn[m], tot_n[m] * 48e-3);
    printf("\n");
    return 0;
}
