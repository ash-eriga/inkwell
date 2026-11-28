// Chunked-prefill kernels for sm_61.  These are independent of the decode
// kernels in kernels.cu; nothing here runs unless a prefill driver invokes it.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include "gpu_caps.cuh"

#define PF_INV_SQRT128 0.08838834764831845f

__device__ __forceinline__ int pf_rbits_words(int ka, int hk) {
    return (hk ? 256 * ka + 128 : 256 * ka) / 32;
}

__device__ __forceinline__ uint32_t pf_exl3_win(const uint32_t* ring, int i, int ka, int hk) {
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const int s_i = hk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
    const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
    const int i0 = b0 >> 5;
    const int s0 = 48 - (b0 & 31);
    const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[i1];
    return (uint32_t)(v >> s0) & 0xffffu;
}

__device__ __forceinline__ float pf_bs_f32(uint32_t w16) {
    return (float)(int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
}

// Rounded EXL3 byte-sum in exact uint8 range.  bytesum is at most 1020, so
// (bytesum + 2) >> 2 needs no clamp; the +2 rides free in dp4a's accumulator.
__device__ __forceinline__ uint32_t pf_bs_u8(uint32_t w16) {
    return __dp4a(w16 * 0x83DCD12Du, 0x01010101u, 2u) >> 2;
}

__device__ __forceinline__ int pf_dp4a_s32_s32(int x, uint32_t w, int acc) {
    asm("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(acc) : "r"(x), "r"(w));
    return acc;
}

__device__ __forceinline__ uint32_t pf_pack_q8(int q0, int q1, int q2, int q3) {
    return (uint32_t)(q0 & 0xff) | ((uint32_t)(q1 & 0xff) << 8)
        | ((uint32_t)(q2 & 0xff) << 16) | ((uint32_t)(q3 & 0xff) << 24);
}

__device__ __forceinline__ int pf_quant_i8(float x, float inv_scale) {
    return max(-127, min(127, __float2int_rn(x * inv_scale)));
}

// Exact 128-point Hadamard in the same pairwise order as inkwell_exl3_had_in.
__device__ __forceinline__ void pf_had128_smem(float* sm, int t) {
    for (int s = 1; s < 128; s <<= 1) {
        const float u = (t & s) ? sm[t - s] - sm[t] : sm[t] + sm[t + s];
        __syncthreads();
        sm[t] = u;
        __syncthreads();
    }
}

// 128-dim Hadamard via four in-register elements per lane plus five warp
// shuffle levels.  This is the same butterfly order as pf_had128_smem.
__device__ __forceinline__ void pf_had128_warp(float* __restrict__ v, int ebase) {
#pragma unroll
    for (int s = 1; s < 4; s <<= 1) {
        float u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            u[i] = (i & s) ? __fsub_rn(v[i ^ s], v[i]) : __fadd_rn(v[i], v[i ^ s]);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = u[i];
    }
#pragma unroll
    for (int s = 4; s < 128; s <<= 1) {
        const int mask = s >> 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], mask);
            v[i] = ((ebase + i) & s) ? __fsub_rn(o, v[i]) : __fadd_rn(v[i], o);
        }
    }
}

extern "C" __global__ void inkwell_exl3_had_in_prefill(
        const float* __restrict__ x, const uint16_t* __restrict__ suh,
        float* __restrict__ xh, int m, int k) {
    __shared__ float sm[128];
    const int row = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    sm[t] = (row < m && e < k) ? x[(size_t)row * k + e] * __half2float(__ushort_as_half(suh[e])) : 0.f;
    __syncthreads();
    pf_had128_smem(sm, t);
    if (row < m && e < k) xh[(size_t)row * k + e] = sm[t] * PF_INV_SQRT128;
}

extern "C" __global__ void inkwell_exl3_had_out_prefill(
        const float* __restrict__ tmat, const uint16_t* __restrict__ svh,
        float* __restrict__ y, int m, int n) {
    __shared__ float sm[128];
    const int row = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    sm[t] = (row < m && e < n) ? tmat[(size_t)row * n + e] : 0.f;
    __syncthreads();
    pf_had128_smem(sm, t);
    if (row < m && e < n) y[(size_t)row * n + e] = sm[t] * PF_INV_SQRT128
        * __half2float(__ushort_as_half(svh[e]));
}

// The weight-decode + FP32 FMA sequence inside the GEMM loops below is kept
// contiguous and small so Challenge 4 can replace it with an IDP.4A core.

struct PfDecodePlan {
    int m0w, m1w, shd, s1, s2, s3;
};

__device__ __forceinline__ int pf_exl3_bstart(int i, int ka, int hk, int r_bits) {
    const int s_i = (i + 1) * ka + (hk ? (i + 1) >> 1 : 0);
    return s_i - 16 + (s_i < 16 ? r_bits : 0);
}

__device__ __forceinline__ PfDecodePlan pf_decode_plan(int i0, int ka, int hk, int r_bits, int nw) {
    PfDecodePlan p;
    const int b = pf_exl3_bstart(i0, ka, hk, r_bits);
    p.m0w = b >> 5;
    p.m1w = (p.m0w + 1 == nw) ? 0 : p.m0w + 1;
    p.shd = b & 31;
    p.s1 = (pf_exl3_bstart(i0 + 1, ka, hk, r_bits) - b + r_bits) % r_bits;
    p.s2 = (pf_exl3_bstart(i0 + 2, ka, hk, r_bits) - b + r_bits) % r_bits;
    p.s3 = (pf_exl3_bstart(i0 + 3, ka, hk, r_bits) - b + r_bits) % r_bits;
    return p;
}

__device__ __forceinline__ void pf_decode4(const uint32_t* __restrict__ rg, PfDecodePlan p,
        float& f0, float& f1, float& f2, float& f3) {
    const uint32_t x = __funnelshift_l(rg[p.m1w], rg[p.m0w], p.shd);
    f0 = pf_bs_f32(x >> 16);
    f1 = pf_bs_f32((x >> (16 - p.s1)) & 0xffffu);
    f2 = pf_bs_f32((x >> (16 - p.s2)) & 0xffffu);
    f3 = pf_bs_f32((x >> (16 - p.s3)) & 0xffffu);
}

// Two-word form used by tiled GEMM kernels.  Each decode lane needs only the
// ring words at m0w/m1w; loading those directly avoids staging the whole ring
// in shared memory and removes one barrier per k tile.
__device__ __forceinline__ void pf_decode4_words(uint32_t r0, uint32_t r1, PfDecodePlan p,
        float& f0, float& f1, float& f2, float& f3) {
    const uint32_t x = __funnelshift_l(r1, r0, p.shd);
    f0 = pf_bs_f32(x >> 16);
    f1 = pf_bs_f32((x >> (16 - p.s1)) & 0xffffu);
    f2 = pf_bs_f32((x >> (16 - p.s2)) & 0xffffu);
    f3 = pf_bs_f32((x >> (16 - p.s3)) & 0xffffu);
}


// Decode the same four trellis outputs as pf_decode4_words, but round each
// byte-sum once into uint8 and pack them in ascending output order.
__device__ __forceinline__ uint32_t pf_decode4_words_q8(uint32_t r0, uint32_t r1, PfDecodePlan p) {
    const uint32_t x = __funnelshift_l(r1, r0, p.shd);
    const uint32_t b0 = pf_bs_u8(x >> 16);
    const uint32_t b1 = pf_bs_u8((x >> (16 - p.s1)) & 0xffffu);
    const uint32_t b2 = pf_bs_u8((x >> (16 - p.s2)) & 0xffffu);
    const uint32_t b3 = pf_bs_u8((x >> (16 - p.s3)) & 0xffffu);
    return (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) ^ 0x80808080u;
}

// Register-tiled EXL3 GEMM core, adapted from the Phase-0 microbenchmark.
// x is [M,K] row-major after the input Hadamard; t is [M,N] row-major before
// the output Hadamard.  Every output accumulates tk-tile outer, r inner, the
// same sequence as the scalar exl3_ref_m1 oracle.
template<int BM, int BN>
__global__ void __launch_bounds__(256) inkwell_exl3_gemm_smem_core(
        const uint32_t* __restrict__ trel,
        const float* __restrict__ x, float* __restrict__ y,
        int M, int tk, int tn, int ka, int hk) {
    static_assert(BN == 64, "prefill tiles are BN=64");
    constexpr int NTILE = BN / 16;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int tn0 = blockIdx.x * NTILE;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));

    __shared__ float Ws[16][BN + 1];
    __shared__ float Xs[16][BM + 1];
    __shared__ uint32_t rings[NTILE][80];

    const int dg = tid >> 6;
    const int dc = (tid & 63) >> 2;
    const int dm = tid & 3;
    const int i0 = 32 * (dc & 7) + 8 * dm + 4 * (dc >> 3);
    auto bstart = [&](int i) {
        int s_i = (i + 1) * ka + (hk ? (i + 1) >> 1 : 0);
        return s_i - 16 + (s_i < 16 ? r_bits : 0);
    };
    const int b = bstart(i0);
    const int m0w = b >> 5;
    const int m1w = (m0w + 1 == nw) ? 0 : m0w + 1;
    const int shd = b & 31;
    const int s1 = (bstart(i0 + 1) - b + r_bits) % r_bits;
    const int s2 = (bstart(i0 + 2) - b + r_bits) % r_bits;
    const int s3 = (bstart(i0 + 3) - b + r_bits) % r_bits;
    const int dr0 = 2 * dm, dr1 = 2 * dm + 1, dr2 = 2 * dm + 8, dr3 = 2 * dm + 9;

    constexpr int TN = (BM == 4) ? 1 : (BM == 8 ? 2 : 4);
    constexpr int TM2 = (BM * BN) / (256 * TN);
    constexpr int NCOLS_THR = BN / TN;
    const int ty = tid / NCOLS_THR;
    const int tx = tid % NCOLS_THR;
    float acc[TM2][TN];
    float sx[TM2];
#pragma unroll
    for (int im = 0; im < TM2; ++im) {
        sx[im] = 0.f;
#pragma unroll
        for (int in = 0; in < TN; ++in) acc[im][in] = 0.f;
    }

    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        if (dg < NTILE) {
            const int u = tid & 63;
            const int tn_i = tn0 + dg;
            if (tn_i < tn && u < w) rings[dg][u] = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + u);
        }
        for (int idx = tid; idx < 16 * BM; idx += 256) {
            const int mm = idx >> 4;
            const int rr = idx & 15;
            const int gm = m0 + mm;
            Xs[rr][mm] = (gm < M) ? x[(size_t)gm * tk * 16 + tk_i * 16 + rr] : 0.f;
        }
        __syncthreads();
        if (dg < NTILE && (tn0 + dg) < tn) {
            const uint32_t* rg = rings[dg];
            const uint32_t X = __funnelshift_l(rg[m1w], rg[m0w], shd);
            const int col = dg * 16 + dc;
            Ws[dr0][col] = pf_bs_f32(X >> 16);
            Ws[dr1][col] = pf_bs_f32((X >> (16 - s1)) & 0xffffu);
            Ws[dr2][col] = pf_bs_f32((X >> (16 - s2)) & 0xffffu);
            Ws[dr3][col] = pf_bs_f32((X >> (16 - s3)) & 0xffffu);
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float xv[TM2];
#pragma unroll
            for (int im = 0; im < TM2; ++im) {
                xv[im] = Xs[r][ty * TM2 + im];
                sx[im] += xv[im];
            }
            float wv[TN];
#pragma unroll
            for (int in = 0; in < TN; ++in) wv[in] = Ws[r][tx + in * NCOLS_THR];
#pragma unroll
            for (int im = 0; im < TM2; ++im)
#pragma unroll
                for (int in = 0; in < TN; ++in)
                    acc[im][in] = fmaf(xv[im], wv[in], acc[im][in]);
        }
        __syncthreads();
    }

    const int N = tn * 16;
#pragma unroll
    for (int im = 0; im < TM2; ++im) {
        const int gm = m0 + ty * TM2 + im;
        if (gm < M) {
            const float off = KOFF * sx[im];
#pragma unroll
            for (int in = 0; in < TN; ++in) {
                const int gn = tn0 * 16 + tx + in * NCOLS_THR;
                if (gn < N) y[(size_t)gm * N + gn] = fmaf(KINV, acc[im][in], off);
            }
        }
    }
}

// BM=64/BN=64 dense-tile core with 128 threads and a 4x8 register tile.
// The strided column map makes each shared-memory scalar load contiguous across
// adjacent lanes.  It preserves the reference reduction order for every output.
__global__ void __launch_bounds__(128) inkwell_exl3_gemm_wide_core(
        const uint32_t* __restrict__ trel,
        const float* __restrict__ x, float* __restrict__ y,
        int M, int tk, int tn, int ka, int hk) {
    constexpr int BM = 64, BN = 64, RM = 4, RN = 8;
    constexpr int NTILE = BN / 16;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int tn0 = blockIdx.x * NTILE;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));

    __shared__ __align__(16) float Ws[2][16][BN + 4];
    __shared__ __align__(16) float Xs[2][16][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 3;
    const int tx = tid & 7;
    float acc[RM][RN];
    float sx[RM];
#pragma unroll
    for (int im = 0; im < RM; ++im) {
        sx[im] = 0.f;
#pragma unroll
        for (int in = 0; in < RN; ++in) acc[im][in] = 0.f;
    }

    float4 xreg[2];
    uint32_t rw[2][2];
    auto load_tile = [&](int tk_i) {
        const int tn_i = tn0 + dtile;
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int idx = tid + ii * 128;
            const int mm = idx >> 2;
            const int rq = idx & 3;
            const int gm = m0 + mm;
            xreg[ii] = (gm < M)
                ? *(const float4*)(x + (size_t)gm * tk * 16 + tk_i * 16 + rq * 4)
                : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (tn_i < tn) {
            const uint32_t* tile_ring = trel + ((size_t)tk_i * tn + tn_i) * (size_t)w;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                rw[z][0] = __ldg(tile_ring + plan[z].m0w);
                rw[z][1] = __ldg(tile_ring + plan[z].m1w);
            }
        }
    };
    auto store_tile = [&](int tk_i, int buf) {
        const int tn_i = tn0 + dtile;
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const int idx = tid + ii * 128;
            const int mm = idx >> 2;
            const int rq = idx & 3;
            Xs[buf][rq * 4][mm] = xreg[ii].x;
            Xs[buf][rq * 4 + 1][mm] = xreg[ii].y;
            Xs[buf][rq * 4 + 2][mm] = xreg[ii].z;
            Xs[buf][rq * 4 + 3][mm] = xreg[ii].w;
        }
        if (tn_i < tn) {
            const int col = dtile * 16 + dc;
            const int wcol = (col & 7) + (col >> 3) * 4 + ((col & 4) ? 28 : 0);
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int dmz = dm + 2 * z;
                float f0, f1, f2, f3;
                pf_decode4_words(rw[z][0], rw[z][1], plan[z], f0, f1, f2, f3);
                Ws[buf][2 * dmz][wcol] = f0;
                Ws[buf][2 * dmz + 1][wcol] = f1;
                Ws[buf][2 * dmz + 8][wcol] = f2;
                Ws[buf][2 * dmz + 9][wcol] = f3;
            }
        }
    };

    load_tile(0);
    store_tile(0, 0);
    __syncthreads();
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        const int cur = tk_i & 1;
        const bool have_next = tk_i + 1 < tk;
        if (have_next) load_tile(tk_i + 1);
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const float4 xv4 = *(const float4*)&Xs[cur][r][ty * RM];
            const float xv[RM] = {xv4.x, xv4.y, xv4.z, xv4.w};
#pragma unroll
            for (int im = 0; im < RM; ++im) sx[im] += xv[im];
            const float4 wa4 = *(const float4*)&Ws[cur][r][tx * 4];
            const float4 wb4 = *(const float4*)&Ws[cur][r][32 + tx * 4];
            const float wv[RN] = {wa4.x, wa4.y, wa4.z, wa4.w, wb4.x, wb4.y, wb4.z, wb4.w};
#pragma unroll
            for (int im = 0; im < RM; ++im)
#pragma unroll
                for (int in = 0; in < RN; ++in)
                    acc[im][in] = fmaf(xv[im], wv[in], acc[im][in]);
        }
        if (have_next) store_tile(tk_i + 1, cur ^ 1);
        __syncthreads();
    }

    const int N = tn * 16;
#pragma unroll
    for (int im = 0; im < RM; ++im) {
        const int gm = m0 + ty * RM + im;
        if (gm < M) {
            const float off = KOFF * sx[im];
#pragma unroll
            for (int in = 0; in < RN; ++in) {
                const int gn = tn0 * 16 + tx * RN + in;
                if (gn < N) y[(size_t)gm * N + gn] = fmaf(KINV, acc[im][in], off);
            }
        }
    }
}


// BM=32/BN=64 dense-tile core with 128 threads and a 4x4 register tile.
// The strided column map makes each shared-memory scalar load contiguous across
// adjacent lanes.  It preserves the reference reduction order for every output.
__global__ void __launch_bounds__(128) inkwell_exl3_gemm_narrow_core(
        const uint32_t* __restrict__ trel,
        const float* __restrict__ x, float* __restrict__ y,
        int M, int tk, int tn, int ka, int hk) {
    constexpr int BM = 32, BN = 64, RM = 4, RN = 4;
    constexpr int NTILE = BN / 16;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int tn0 = blockIdx.x * NTILE;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));

    __shared__ float Ws[16][BN + 1];
    __shared__ float Xs[16][BM + 1];
    __shared__ uint32_t rings[NTILE][80];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 4;
    const int tx = tid & 15;
    float acc[RM][RN];
    float sx[RM];
#pragma unroll
    for (int im = 0; im < RM; ++im) {
        sx[im] = 0.f;
#pragma unroll
        for (int in = 0; in < RN; ++in) acc[im][in] = 0.f;
    }

    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        const int tn_i = tn0 + dtile;
        const uint32_t* tile_ring = trel + ((size_t)tk_i * tn + tn_i) * (size_t)w;
        for (int idx = tid; idx < 16 * BM; idx += 128) {
            const int mm = idx >> 4;
            const int rr = idx & 15;
            const int gm = m0 + mm;
            Xs[rr][mm] = (gm < M) ? x[(size_t)gm * tk * 16 + tk_i * 16 + rr] : 0.f;
        }
        if (tn_i < tn) {
            const int col = dtile * 16 + dc;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int dmz = dm + 2 * z;
                float f0, f1, f2, f3;
                pf_decode4_words(__ldg(tile_ring + plan[z].m0w), __ldg(tile_ring + plan[z].m1w), plan[z], f0, f1, f2, f3);
                Ws[2 * dmz][col] = f0;
                Ws[2 * dmz + 1][col] = f1;
                Ws[2 * dmz + 8][col] = f2;
                Ws[2 * dmz + 9][col] = f3;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float xv[RM];
#pragma unroll
            for (int im = 0; im < RM; ++im) {
                xv[im] = Xs[r][ty * RM + im];
                sx[im] += xv[im];
            }
            float wv[RN];
            for (int in = 0; in < RN; ++in) wv[in] = Ws[r][tx + in * 16];
#pragma unroll
            for (int im = 0; im < RM; ++im)
#pragma unroll
                for (int in = 0; in < RN; ++in)
                    acc[im][in] = fmaf(xv[im], wv[in], acc[im][in]);
        }
        __syncthreads();
    }

    const int N = tn * 16;
#pragma unroll
    for (int im = 0; im < RM; ++im) {
        const int gm = m0 + ty * RM + im;
        if (gm < M) {
            const float off = KOFF * sx[im];
#pragma unroll
            for (int in = 0; in < RN; ++in) {
                const int gn = tn0 * 16 + tx + in * 16;
                if (gn < N) y[(size_t)gm * N + gn] = fmaf(KINV, acc[im][in], off);
            }
        }
    }
}



// Fused dense Hadamard-input quantization.  The output activation layout is
// one int32 word per four K positions, ordered in the same {r, r+1, r+8, r+9}
// groups that the EXL3 tile decoder emits.  Each 128-element Hadamard block
// has one scale in xscale and contributes one rowsum atomically.
extern "C" __global__ void inkwell_exl3_had_in_q8(
        const float* __restrict__ x, const uint16_t* __restrict__ suh,
        int32_t* __restrict__ xq, float* __restrict__ xscale,
        float* __restrict__ rowsum, int m, int k) {
    __shared__ float sm[128];
    __shared__ float rmax[128];
    __shared__ float rsum[128];
    const int row = blockIdx.y;
    const int seg = blockIdx.x;
    const int base = seg * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    if (row >= m) return;
    sm[t] = (e < k) ? x[(size_t)row * k + e] * __half2float(__ushort_as_half(suh[e])) : 0.f;
    __syncthreads();
    pf_had128_smem(sm, t);
    const float v = sm[t] * PF_INV_SQRT128;
    sm[t] = v;
    rmax[t] = fabsf(v);
    rsum[t] = v;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (t < st) {
            rmax[t] = fmaxf(rmax[t], rmax[t + st]);
            rsum[t] = rsum[t] + rsum[t + st];
        }
        __syncthreads();
    }
    const float amax = rmax[0];
    const float scale = amax * (1.0f / 127.0f);
    const float inv = amax > 0.f ? 127.0f / amax : 0.f;
    if (t == 0) {
        xscale[(size_t)row * (k / 128) + seg] = scale;
        atomicAdd(rowsum + row, rsum[0]);
    }
    if (t < 32) {
        const int sub = t >> 2;
        const int dmz = t & 3;
        const int p0 = 16 * sub + 2 * dmz;
        const int p1 = p0 + 1;
        const int p2 = p0 + 8;
        const int p3 = p0 + 9;
        const uint32_t pack = pf_pack_q8(
            pf_quant_i8(sm[p0], inv), pf_quant_i8(sm[p1], inv),
            pf_quant_i8(sm[p2], inv), pf_quant_i8(sm[p3], inv));
        xq[(size_t)row * (k >> 2) + seg * 32 + t] = (int32_t)pack;
    }
}

// Wide dense-tile INT8 EXL3 core.  It keeps the FP32 core's BM=64/BN=64,
// 4x8 register geometry and double-buffered ring/activation prefetch, but the
// 16-position K tile is four packed dp4a steps instead of sixteen scalar FMA steps.
__global__ void __launch_bounds__(128) inkwell_exl3_gemm_dp4a_wide_core(
        const uint32_t* __restrict__ trel,
        const int32_t* __restrict__ xq, const float* __restrict__ xscale,
        const float* __restrict__ rowsum, float* __restrict__ y,
        int M, int tk, int tn, int ka, int hk) {
    constexpr int BM = 64, BN = 64, RM = 4, RN = 8;
    constexpr int NTILE = BN / 16;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int tn0 = blockIdx.x * NTILE;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF_C = 1024.f * KINV + __half2float(__ushort_as_half(0xc931)) + 512.f * KINV;
    const int kb = tk >> 3;

    __shared__ __align__(16) uint32_t Ws[2][4][BN + 4];
    __shared__ __align__(16) int32_t Xs[2][4][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 3;
    const int tx = tid & 7;
    float acc[RM][RN];
    int iacc[RM][RN];
#pragma unroll
    for (int im = 0; im < RM; ++im)
#pragma unroll
        for (int in = 0; in < RN; ++in) {
            acc[im][in] = 0.f;
            iacc[im][in] = 0;
        }

    uint4 xreg = make_uint4(0u, 0u, 0u, 0u);
    uint32_t rw[2][2];
    auto load_tile = [&](int tk_i) {
        const int tn_i = tn0 + dtile;
        if (tid < BM) {
            const int gm = m0 + tid;
            xreg = (gm < M)
                ? *(const uint4*)(xq + (size_t)gm * (tk * 4) + tk_i * 4)
                : make_uint4(0u, 0u, 0u, 0u);
        }
        if (tn_i < tn) {
            const uint32_t* tile_ring = trel + ((size_t)tk_i * tn + tn_i) * (size_t)w;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                rw[z][0] = __ldg(tile_ring + plan[z].m0w);
                rw[z][1] = __ldg(tile_ring + plan[z].m1w);
            }
        }
    };
    auto store_tile = [&](int buf) {
        const int tn_i = tn0 + dtile;
        if (tid < BM) {
            const int mm = tid;
            Xs[buf][0][mm] = (int32_t)xreg.x;
            Xs[buf][1][mm] = (int32_t)xreg.y;
            Xs[buf][2][mm] = (int32_t)xreg.z;
            Xs[buf][3][mm] = (int32_t)xreg.w;
        }
        if (tn_i < tn) {
            const int col = dtile * 16 + dc;
            const int wcol = (col & 7) + (col >> 3) * 4 + ((col & 4) ? 28 : 0);
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int dmz = dm + 2 * z;
                Ws[buf][dmz][wcol] = pf_decode4_words_q8(rw[z][0], rw[z][1], plan[z]);
            }
        }
    };

    load_tile(0);
    store_tile(0);
    __syncthreads();
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        const int cur = tk_i & 1;
        const bool have_next = tk_i + 1 < tk;
        if (have_next) load_tile(tk_i + 1);
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
            const int4 xv4 = *(const int4*)&Xs[cur][k4][ty * RM];
            const int xv[RM] = {xv4.x, xv4.y, xv4.z, xv4.w};
            const int4 wa4 = *(const int4*)&Ws[cur][k4][tx * 4];
            const int4 wb4 = *(const int4*)&Ws[cur][k4][32 + tx * 4];
            const uint32_t wv[RN] = {
                (uint32_t)wa4.x, (uint32_t)wa4.y, (uint32_t)wa4.z, (uint32_t)wa4.w,
                (uint32_t)wb4.x, (uint32_t)wb4.y, (uint32_t)wb4.z, (uint32_t)wb4.w
            };
#pragma unroll
            for (int im = 0; im < RM; ++im)
#pragma unroll
                for (int in = 0; in < RN; ++in)
                    iacc[im][in] = pf_dp4a_s32_s32(xv[im], wv[in], iacc[im][in]);
        }
        if ((tk_i & 7) == 7) {
            const int b128 = tk_i >> 3;
#pragma unroll
            for (int im = 0; im < RM; ++im) {
                const int gm = m0 + ty * RM + im;
                const float sc = (gm < M) ? xscale[(size_t)gm * kb + b128] : 0.f;
#pragma unroll
                for (int in = 0; in < RN; ++in) {
                    acc[im][in] = fmaf((float)iacc[im][in], sc, acc[im][in]);
                    iacc[im][in] = 0;
                }
            }
        }
        if (have_next) store_tile(cur ^ 1);
        __syncthreads();
    }

    const int N = tn * 16;
#pragma unroll
    for (int im = 0; im < RM; ++im) {
        const int gm = m0 + ty * RM + im;
        if (gm < M) {
            const float off = KOFF_C * rowsum[gm];
#pragma unroll
            for (int in = 0; in < RN; ++in) {
                const int gn = tn0 * 16 + tx * RN + in;
                if (gn < N) y[(size_t)gm * N + gn] = fmaf(4.0f * KINV, acc[im][in], off);
            }
        }
    }
}

// Narrow dense-tile INT8 core for small-N shapes; same arithmetic as the wide
// core, with BM=32/BN=64 and a 4x4 register tile.
__global__ void __launch_bounds__(128) inkwell_exl3_gemm_dp4a_narrow_core(
        const uint32_t* __restrict__ trel,
        const int32_t* __restrict__ xq, const float* __restrict__ xscale,
        const float* __restrict__ rowsum, float* __restrict__ y,
        int M, int tk, int tn, int ka, int hk) {
    constexpr int BM = 32, BN = 64, RM = 4, RN = 4;
    constexpr int NTILE = BN / 16;
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM;
    const int tn0 = blockIdx.x * NTILE;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF_C = 1024.f * KINV + __half2float(__ushort_as_half(0xc931)) + 512.f * KINV;
    const int kb = tk >> 3;

    __shared__ __align__(16) uint32_t Ws[2][4][BN + 4];
    __shared__ __align__(16) int32_t Xs[2][4][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 4;
    const int tx = tid & 15;
    float acc[RM][RN];
    int iacc[RM][RN];
#pragma unroll
    for (int im = 0; im < RM; ++im)
#pragma unroll
        for (int in = 0; in < RN; ++in) {
            acc[im][in] = 0.f;
            iacc[im][in] = 0;
        }

    uint4 xreg = make_uint4(0u, 0u, 0u, 0u);
    uint32_t rw[2][2];
    auto load_tile = [&](int tk_i) {
        const int tn_i = tn0 + dtile;
        if (tid < BM) {
            const int gm = m0 + tid;
            xreg = (gm < M)
                ? *(const uint4*)(xq + (size_t)gm * (tk * 4) + tk_i * 4)
                : make_uint4(0u, 0u, 0u, 0u);
        }
        if (tn_i < tn) {
            const uint32_t* tile_ring = trel + ((size_t)tk_i * tn + tn_i) * (size_t)w;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                rw[z][0] = __ldg(tile_ring + plan[z].m0w);
                rw[z][1] = __ldg(tile_ring + plan[z].m1w);
            }
        }
    };
    auto store_tile = [&](int buf) {
        const int tn_i = tn0 + dtile;
        if (tid < BM) {
            const int mm = tid;
            Xs[buf][0][mm] = (int32_t)xreg.x;
            Xs[buf][1][mm] = (int32_t)xreg.y;
            Xs[buf][2][mm] = (int32_t)xreg.z;
            Xs[buf][3][mm] = (int32_t)xreg.w;
        }
        if (tn_i < tn) {
            const int col = dtile * 16 + dc;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int dmz = dm + 2 * z;
                Ws[buf][dmz][col] = pf_decode4_words_q8(rw[z][0], rw[z][1], plan[z]);
            }
        }
    };

    load_tile(0);
    store_tile(0);
    __syncthreads();
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        const int cur = tk_i & 1;
        const bool have_next = tk_i + 1 < tk;
        if (have_next) load_tile(tk_i + 1);
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
            const int4 xv4 = *(const int4*)&Xs[cur][k4][ty * RM];
            const int xv[RM] = {xv4.x, xv4.y, xv4.z, xv4.w};
            const int4 wq4 = *(const int4*)&Ws[cur][k4][tx * 4];
            const uint32_t wv[RN] = {(uint32_t)wq4.x, (uint32_t)wq4.y, (uint32_t)wq4.z, (uint32_t)wq4.w};
#pragma unroll
            for (int im = 0; im < RM; ++im)
#pragma unroll
                for (int in = 0; in < RN; ++in)
                    iacc[im][in] = pf_dp4a_s32_s32(xv[im], wv[in], iacc[im][in]);
        }
        if ((tk_i & 7) == 7) {
            const int b128 = tk_i >> 3;
#pragma unroll
            for (int im = 0; im < RM; ++im) {
                const int gm = m0 + ty * RM + im;
                const float sc = (gm < M) ? xscale[(size_t)gm * kb + b128] : 0.f;
#pragma unroll
                for (int in = 0; in < RN; ++in) {
                    acc[im][in] = fmaf((float)iacc[im][in], sc, acc[im][in]);
                    iacc[im][in] = 0;
                }
            }
        }
        if (have_next) store_tile(cur ^ 1);
        __syncthreads();
    }

    const int N = tn * 16;
#pragma unroll
    for (int im = 0; im < RM; ++im) {
        const int gm = m0 + ty * RM + im;
        if (gm < M) {
            const float off = KOFF_C * rowsum[gm];
#pragma unroll
            for (int in = 0; in < RN; ++in) {
                const int gn = tn0 * 16 + tx * RN + in;
                if (gn < N) y[(size_t)gm * N + gn] = fmaf(4.0f * KINV, acc[im][in], off);
            }
        }
    }
}

template<int BM>
static inline void pf_launch_gemm_core(const uint32_t* trel, const float* xh, float* t,
        int M, int tk, int tn, int ka, int hk, cudaStream_t st) {
    dim3 g((tn + 3) / 4, (M + BM - 1) / BM);
    inkwell_exl3_gemm_smem_core<BM, 64><<<g, 256, 0, st>>>(trel, xh, t, M, tk, tn, ka, hk);
}

extern "C" void ink_launch_exl3_gemm_smem(const void* trel, const void* suh, const void* svh,
        const float* x, float* y, int M, long long k, long long n, long long ka, long long halfk,
        float* scratch, void* stream) {
    // scratch: xh [M*k] followed by t [M*n].  The caller owns it so Phase 2 can
    // allocate one maximum-size arena instead of per-weight decode scratch.
    cudaStream_t st = (cudaStream_t)stream;
    float* xh = scratch;
    float* t = scratch + (size_t)M * k;
    dim3 gi((unsigned)(k / 128), (unsigned)M);
    const int tk = (int)(k / 16), tn = (int)(n / 16);
    const bool q8 = ink_pf_int8_enabled() && M > 32 && (k & 127) == 0;
    if (q8) {
        int32_t* xq = (int32_t*)scratch;
        float* xscale = scratch + (size_t)M * k / 4;
        float* rowsum = xscale + (size_t)M * (k / 128);
        cudaMemsetAsync(rowsum, 0, (size_t)M * 4, st);
        inkwell_exl3_had_in_q8<<<gi, 128, 0, st>>>(x, (const uint16_t*)suh, xq, xscale, rowsum, M, (int)k);
        if (tn <= 72) {
            dim3 g((tn + 3) / 4, (M + 31) / 32);
            inkwell_exl3_gemm_dp4a_narrow_core<<<g, 128, 0, st>>>((const uint32_t*)trel, xq, xscale, rowsum, t, M, tk, tn, (int)ka, (int)halfk);
        } else {
            dim3 g((tn + 3) / 4, (M + 63) / 64);
            inkwell_exl3_gemm_dp4a_wide_core<<<g, 128, 0, st>>>((const uint32_t*)trel, xq, xscale, rowsum, t, M, tk, tn, (int)ka, (int)halfk);
        }
    } else {
    inkwell_exl3_had_in_prefill<<<gi, 128, 0, st>>>(x, (const uint16_t*)suh, xh, M, (int)k);
    if (M <= 4) pf_launch_gemm_core<4>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else if (M <= 8) pf_launch_gemm_core<8>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else if (M <= 16) pf_launch_gemm_core<16>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else if (M <= 32) pf_launch_gemm_core<32>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else {
        if (tn <= 72) {
            dim3 g((tn + 3) / 4, (M + 31) / 32);
            inkwell_exl3_gemm_narrow_core<<<g, 128, 0, st>>>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk);
        } else {
            dim3 g((tn + 3) / 4, (M + 63) / 64);
            inkwell_exl3_gemm_wide_core<<<g, 128, 0, st>>>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk);
        }
    }
    }
    dim3 go((unsigned)(n / 128), (unsigned)M);
    inkwell_exl3_had_out_prefill<<<go, 128, 0, st>>>(t, (const uint16_t*)svh, y, M, (int)n);
}

// ---------------- MoE grouped routing and GEMM ----------------
// ids/w are token-major [C,8].  The sorter emits expanded rows sorted by
// expert, inverse_row[token*8+j], expert_offsets[129], and conservative
// 64-row tile metadata.  Rows within a tile belong to one expert.

extern "C" __global__ void inkwell_exl3_moe_group_sort(
        const int* __restrict__ ids, const float* __restrict__ weights, int C,
        int* __restrict__ expert_offsets, int* __restrict__ sorted_token,
        int* __restrict__ sorted_slot, float* __restrict__ sorted_weight,
        int* __restrict__ inverse_row, int* __restrict__ tile_expert,
        int* __restrict__ tile_start, int* __restrict__ tile_count,
        int* __restrict__ tile_n) {
    __shared__ int count[128], start[128], cursor[128];
    const int P = C * 8;
    const int e = threadIdx.x;
    if (e < 128) count[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < P; i += blockDim.x) atomicAdd(&count[ids[i] & 127], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < 128; ++i) {
            expert_offsets[i] = sum;
            start[i] = sum;
            sum += count[i];
        }
        expert_offsets[128] = sum;
    }
    __syncthreads();
    if (e < 128) cursor[e] = start[e];
    __syncthreads();
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const int ex = ids[i] & 127;
        const int row = atomicAdd(&cursor[ex], 1);
        sorted_token[row] = i >> 3;
        sorted_slot[row] = i & 7;
        sorted_weight[row] = weights[i];
        inverse_row[i] = row;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nt = 0;
        for (int ex = 0; ex < 128; ++ex) {
            int p = expert_offsets[ex];
            const int end = expert_offsets[ex + 1];
            while (p < end) {
                const int cnt = min(64, end - p);
                tile_expert[nt] = ex;
                tile_start[nt] = p;
                tile_count[nt] = cnt;
                ++nt;
                p += cnt;
            }
        }
        *tile_n = nt;
    }
}

// Expanded-row Hadamard input for one expert descriptor table.  One warp
// owns each 128-column segment; each lane owns four contiguous elements.
extern "C" __global__ void inkwell_moe_had_in_rows(
        const uint64_t* __restrict__ descs, const int* __restrict__ tile_expert_id,
        const int* __restrict__ sorted_token, const float* __restrict__ x,
        float* __restrict__ xh, int row_base, int P, int k) {
    const int row = row_base + blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int seg = blockIdx.x * (blockDim.x >> 5) + wid;
    const int base = seg * 128;
    if (blockIdx.y >= P || base >= k) return;
    const int ex = tile_expert_id[row];
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 1];
    const int e0 = base + lane * 4;
    const float4 xv = *(const float4*)(x + (size_t)sorted_token[row] * k + e0);
    const uint2 sv = *(const uint2*)(suh + e0);
    float v[4];
    v[0] = __fmul_rn(xv.x, __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu))));
    v[1] = __fmul_rn(xv.y, __half2float(__ushort_as_half((uint16_t)(sv.x >> 16))));
    v[2] = __fmul_rn(xv.z, __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu))));
    v[3] = __fmul_rn(xv.w, __half2float(__ushort_as_half((uint16_t)(sv.y >> 16))));
    pf_had128_warp(v, lane * 4);
    *(float4*)(xh + (size_t)row * k + e0) = make_float4(
        __fmul_rn(v[0], PF_INV_SQRT128), __fmul_rn(v[1], PF_INV_SQRT128),
        __fmul_rn(v[2], PF_INV_SQRT128), __fmul_rn(v[3], PF_INV_SQRT128));
}


// Warp-local grouped-MoE Hadamard input with fused per-128-block INT8
// quantization and rowsum.  xq uses the packed four-position EXL3 order.
extern "C" __global__ void inkwell_moe_had_in_rows_q8(
        const uint64_t* __restrict__ descs, const int* __restrict__ row_expert,
        const int* __restrict__ sorted_token, const float* __restrict__ x,
        int32_t* __restrict__ xq, float* __restrict__ xscale,
        float* __restrict__ rowsum, int row_base, int P, int k) {
    const int row = row_base + blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int seg = blockIdx.x * (blockDim.x >> 5) + wid;
    const int base = seg * 128;
    if (blockIdx.y >= P || base >= k) return;
    const int ex = row_expert[row];
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 1];
    const int e0 = base + lane * 4;
    const float4 xv = *(const float4*)(x + (size_t)sorted_token[row] * k + e0);
    const uint2 sv = *(const uint2*)(suh + e0);
    float v[4];
    v[0] = __fmul_rn(xv.x, __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu))));
    v[1] = __fmul_rn(xv.y, __half2float(__ushort_as_half((uint16_t)(sv.x >> 16))));
    v[2] = __fmul_rn(xv.z, __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu))));
    v[3] = __fmul_rn(xv.w, __half2float(__ushort_as_half((uint16_t)(sv.y >> 16))));
    pf_had128_warp(v, lane * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], PF_INV_SQRT128);
    float amax = 0.f, sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
        sum = __fadd_rn(sum, v[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    const float scale = amax * (1.0f / 127.0f);
    const float inv = amax > 0.f ? 127.0f / amax : 0.f;
    if (lane == 0) {
        xscale[(size_t)row * (k / 128) + seg] = scale;
        atomicAdd(rowsum + row, sum);
    }
    __shared__ float qs[4][128];
#pragma unroll
    for (int i = 0; i < 4; ++i) qs[wid][lane * 4 + i] = v[i];
    __syncwarp();
    const int sub = lane >> 2;
    const int dmz = lane & 3;
    const int p0 = 16 * sub + 2 * dmz;
    const int p1 = p0 + 1;
    const int p2 = p0 + 8;
    const int p3 = p0 + 9;
    const uint32_t pack = pf_pack_q8(
        pf_quant_i8(qs[wid][p0], inv), pf_quant_i8(qs[wid][p1], inv),
        pf_quant_i8(qs[wid][p2], inv), pf_quant_i8(qs[wid][p3], inv));
    xq[(size_t)row * (k >> 2) + seg * 32 + lane] = (int32_t)pack;
}

// SiTU plus down-projection Hadamard, fused with the same per-block INT8
// quantization/layout as inkwell_moe_had_in_rows_q8.
extern "C" __global__ void inkwell_moe_situ_had_rows_q8(
        const uint64_t* __restrict__ descs_d, const int* __restrict__ row_expert,
        const float* __restrict__ gu, int32_t* __restrict__ xq,
        float* __restrict__ xscale, float* __restrict__ rowsum,
        float beta, float lin_beta, int row_base, int P, int ncols) {
    const int row = row_base + blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int seg = blockIdx.x * (blockDim.x >> 5) + wid;
    const int base = seg * 128;
    if (blockIdx.y >= P || base >= ncols) return;
    const int ex = row_expert[row];
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs_d[ex * 4 + 1];
    const int e0 = base + lane * 4;
    const float4 gv = *(const float4*)(gu + (size_t)row * 2 * ncols + e0);
    const float4 uv = *(const float4*)(gu + (size_t)row * 2 * ncols + ncols + e0);
    const uint2 sv = *(const uint2*)(suh + e0);
    const float gf[4] = {gv.x, gv.y, gv.z, gv.w};
    const float uf[4] = {uv.x, uv.y, uv.z, uv.w};
    const uint16_t sh[4] = {
        (uint16_t)(sv.x & 0xffffu), (uint16_t)(sv.x >> 16),
        (uint16_t)(sv.y & 0xffffu), (uint16_t)(sv.y >> 16)
    };
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float g = gf[i];
        const float u = uf[i];
        const float a = (beta * tanhf(g / beta)) / (1.0f + expf(-g));
        const float b = lin_beta * tanhf(u / lin_beta);
        v[i] = __fmul_rn(__fmul_rn(a, b), __half2float(__ushort_as_half(sh[i])));
    }
    pf_had128_warp(v, lane * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], PF_INV_SQRT128);
    float amax = 0.f, sum = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
        sum = __fadd_rn(sum, v[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    const float scale = amax * (1.0f / 127.0f);
    const float inv = amax > 0.f ? 127.0f / amax : 0.f;
    if (lane == 0) {
        xscale[(size_t)row * (ncols / 128) + seg] = scale;
        atomicAdd(rowsum + row, sum);
    }
    __shared__ float qs[4][128];
#pragma unroll
    for (int i = 0; i < 4; ++i) qs[wid][lane * 4 + i] = v[i];
    __syncwarp();
    const int sub = lane >> 2;
    const int dmz = lane & 3;
    const int p0 = 16 * sub + 2 * dmz;
    const int p1 = p0 + 1;
    const int p2 = p0 + 8;
    const int p3 = p0 + 9;
    const uint32_t pack = pf_pack_q8(
        pf_quant_i8(qs[wid][p0], inv), pf_quant_i8(qs[wid][p1], inv),
        pf_quant_i8(qs[wid][p2], inv), pf_quant_i8(qs[wid][p3], inv));
    xq[(size_t)row * (ncols >> 2) + seg * 32 + lane] = (int32_t)pack;
}

extern "C" __global__ void inkwell_moe_had_out_rows(
        const uint64_t* __restrict__ descs, const int* __restrict__ row_expert,
        const float* __restrict__ tmat, float* __restrict__ y, int row_base, int P, int n) {
    const int row = row_base + blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int seg = blockIdx.x * (blockDim.x >> 5) + wid;
    const int base = seg * 128;
    if (blockIdx.y >= P || base >= n) return;
    const int ex = row_expert[row];
    const uint16_t* svh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 2];
    const int e0 = base + lane * 4;
    const float4 tv = *(const float4*)(tmat + (size_t)row * n + e0);
    float v[4] = {tv.x, tv.y, tv.z, tv.w};
    pf_had128_warp(v, lane * 4);
    const uint2 sv = *(const uint2*)(svh + e0);
    *(float4*)(y + (size_t)row * n + e0) = make_float4(
        __fmul_rn(__fmul_rn(v[0], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu)))),
        __fmul_rn(__fmul_rn(v[1], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.x >> 16)))),
        __fmul_rn(__fmul_rn(v[2], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu)))),
        __fmul_rn(__fmul_rn(v[3], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.y >> 16)))));
}

// Row sums for the EXL3 affine offset.  A 32-column tile is staged through
// shared memory so global loads stay coalesced, but each row is still accumulated
// in strictly ascending k order with explicit rounding.  That matches the order
// formerly produced inside the grouped GEMM while computing the sum only once
// per row instead of once per output-column CTA.
__global__ void inkwell_moe_rowsum_rows(const float* __restrict__ x,
        float* __restrict__ rowsum, int row_base, int P, int K) {
    constexpr int ROWS = 128, KT = 32;
    __shared__ float xs[KT][ROWS + 1];
    const int tid = threadIdx.x;
    const int row0 = row_base + blockIdx.x * ROWS;
    const int row = row0 + tid;
    const int row_end = row_base + P;
    float acc = 0.f;
    for (int kb = 0; kb < K; kb += KT) {
        for (int idx = tid; idx < ROWS * KT; idx += 256) {
            const int rr = idx >> 5;
            const int kk = idx & 31;
            const int gr = row0 + rr;
            const int gk = kb + kk;
            xs[kk][rr] = (gr < row_end && gk < K) ? x[(size_t)gr * K + gk] : 0.f;
        }
        __syncthreads();
        if (tid < ROWS && row < row_end) {
#pragma unroll
            for (int kk = 0; kk < KT; ++kk)
                acc = __fadd_rn(acc, xs[kk][tid]);
        }
        __syncthreads();
    }
    if (tid < ROWS && row < row_end) rowsum[row] = acc;
}

static void pf_launch_moe_rowsum(const float* x, float* rowsum, int row_base, int P, int K, cudaStream_t st) {
    inkwell_moe_rowsum_rows<<<dim3((unsigned)((P + 127) / 128)), 256, 0, st>>>(x, rowsum, row_base, P, K);
}

// Grouped EXL3 core.  The externally visible launch contract still supplies
// one CTA per 64 output columns and 256 threads; even column CTAs cover 128
// columns and odd CTAs exit.  Rows are processed in unpadded 32-row pieces,
// which matches the realistic mean of 32 routed rows per expert.
__global__ void __launch_bounds__(256) inkwell_exl3_moe_grouped_gemm_kernel(
        const uint64_t* __restrict__ descs,
        const int* __restrict__ tile_expert, const int* __restrict__ tile_start,
        const int* __restrict__ tile_count, const int* __restrict__ tile_n,
        const float* __restrict__ xh, const float* __restrict__ rowsum, float* __restrict__ y,
        int tk, int tn, int tile_base_index, int tile_mode) {
    if (blockIdx.x & 1) return;
    constexpr int BM = 32, BN = 128, RM = 4, RN = 4;
    const int mt = tile_base_index + blockIdx.y;
    if (mt >= *tile_n) return;
    const int ex = tile_expert[mt];
    const int row0 = tile_start[mt];
    const int rows = tile_count[mt];
    if (tile_mode == 1 && rows > 32) return;
    if (tile_mode == 2 && rows <= 32) return;
    const uint64_t* d = descs + ex * 4;
    const uint32_t* trel = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int hk = (int)((d[3] >> 8) & 1);
    const int tid = threadIdx.x;
    const int tn0 = blockIdx.x * 4;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    __shared__ __align__(16) float Ws[2][16][BN + 4];
    __shared__ __align__(16) float Xs[2][16][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 5;
    const int tx = tid & 31;
    const int K = tk * 16;
    const int N = tn * 16;
    for (int rbase = 0; rbase < rows; rbase += BM) {
        const int rcnt = min(BM, rows - rbase);
        float4 xreg = make_float4(0.f, 0.f, 0.f, 0.f);
        uint32_t ringv = 0;
        auto load_tile = [&](int tk_i) {
            const int tn_i = tn0 + dtile;
            ringv = 0;
            if (tn_i < tn && du < w) {
                ringv = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
            }
            if (tid < 4 * BM) {
                const int mm = tid >> 2;
                const int rq = tid & 3;
                xreg = (mm < rcnt)
                    ? *(const float4*)(xh + (size_t)(row0 + rbase + mm) * K + tk_i * 16 + rq * 4)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        };
        auto store_tile = [&](int buf) {
            const int tn_i = tn0 + dtile;
            if (tid < 4 * BM) {
                const int mm = tid >> 2;
                const int rq = tid & 3;
                Xs[buf][rq * 4][mm] = xreg.x;
                Xs[buf][rq * 4 + 1][mm] = xreg.y;
                Xs[buf][rq * 4 + 2][mm] = xreg.z;
                Xs[buf][rq * 4 + 3][mm] = xreg.w;
            }
            if (tn_i < tn) {
                const int col = dtile * 16 + dc;
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int dmz = dm + 2 * z;
                    float f0, f1, f2, f3;
                    const uint32_t r0 = __shfl_sync(0xffffffffu, ringv, plan[z].m0w);
                    const uint32_t r1 = __shfl_sync(0xffffffffu, ringv, plan[z].m1w);
                    pf_decode4_words(r0, r1, plan[z], f0, f1, f2, f3);
                    Ws[buf][2 * dmz][col] = f0;
                    Ws[buf][2 * dmz + 1][col] = f1;
                    Ws[buf][2 * dmz + 8][col] = f2;
                    Ws[buf][2 * dmz + 9][col] = f3;
                }
            }
        };
        float acc[RM][RN];
#pragma unroll
        for (int im = 0; im < RM; ++im) {
#pragma unroll
            for (int in = 0; in < RN; ++in) acc[im][in] = 0.f;
        }
        load_tile(0);
        store_tile(0);
        __syncthreads();
        for (int tk_i = 0; tk_i < tk; ++tk_i) {
            const int cur = tk_i & 1;
            const bool have_next = tk_i + 1 < tk;
            if (have_next) load_tile(tk_i + 1);

            if (ty * RM < rcnt) {
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    const float4 xv = *(const float4*)&Xs[cur][r][ty * RM];
                    const float4 wq = *(const float4*)&Ws[cur][r][tx * 4];
                    const float wv[RN] = {wq.x, wq.y, wq.z, wq.w};
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        acc[0][in] = fmaf(xv.x, wv[in], acc[0][in]);
                        acc[1][in] = fmaf(xv.y, wv[in], acc[1][in]);
                        acc[2][in] = fmaf(xv.z, wv[in], acc[2][in]);
                        acc[3][in] = fmaf(xv.w, wv[in], acc[3][in]);
                    }
                }
            }
            if (have_next) store_tile(cur ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int im = 0; im < RM; ++im) {
            const int lr = rbase + ty * RM + im;
            if (lr < rows) {
                const float off = KOFF * rowsum[row0 + lr];
#pragma unroll
                for (int in = 0; in < RN; ++in) {
                    const int gn = tn0 * 16 + tx * RN + in;
                    if (gn < N) y[(size_t)(row0 + lr) * N + gn] = fmaf(KINV, acc[im][in], off);
                }
            }
        }
    }
}



// Grouped INT8 EXL3 core.  This is the dp4a counterpart of the 32-row FP32
// grouped kernel: one uint32 activation word and one uint32 weight word now
// encode the same four K positions that fp32 processed as four scalar FMAs.
__global__ void __launch_bounds__(256) inkwell_exl3_moe_grouped_gemm_dp4a_kernel(
        const uint64_t* __restrict__ descs,
        const int* __restrict__ tile_expert, const int* __restrict__ tile_start,
        const int* __restrict__ tile_count, const int* __restrict__ tile_n,
        const int32_t* __restrict__ xq, const float* __restrict__ xscale,
        const float* __restrict__ rowsum, float* __restrict__ y,
        int tk, int tn, int tile_base_index, int tile_mode) {
    if (blockIdx.x & 1) return;
    constexpr int BM = 32, BN = 128, RM = 4, RN = 4;
    const int mt = tile_base_index + blockIdx.y;
    if (mt >= *tile_n) return;
    const int ex = tile_expert[mt];
    const int row0 = tile_start[mt];
    const int rows = tile_count[mt];
    if (tile_mode == 1 && rows > 32) return;
    if (tile_mode == 2 && rows <= 32) return;
    const uint64_t* d = descs + ex * 4;
    const uint32_t* trel = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int hk = (int)((d[3] >> 8) & 1);
    const int tid = threadIdx.x;
    const int tn0 = blockIdx.x * 4;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF_C = 1024.f * KINV + __half2float(__ushort_as_half(0xc931)) + 512.f * KINV;
    const int K = tk * 16;
    const int N = tn * 16;
    const int kb = tk >> 3;
    __shared__ __align__(16) uint32_t Ws[2][4][BN + 4];
    __shared__ __align__(16) int32_t Xs[2][4][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 5;
    const int tx = tid & 31;
    for (int rbase = 0; rbase < rows; rbase += BM) {
        const int rcnt = min(BM, rows - rbase);
        uint4 xreg = make_uint4(0u, 0u, 0u, 0u);
        uint32_t ringv = 0;
        auto load_tile = [&](int tk_i) {
            const int tn_i = tn0 + dtile;
            ringv = 0;
            if (tn_i < tn && du < w)
                ringv = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
            if (tid < BM) {
                const int mm = tid;
                xreg = (mm < rcnt)
                    ? *(const uint4*)(xq + (size_t)(row0 + rbase + mm) * (tk * 4) + tk_i * 4)
                    : make_uint4(0u, 0u, 0u, 0u);
            }
        };
        auto store_tile = [&](int buf) {
            const int tn_i = tn0 + dtile;
            if (tid < BM) {
                const int mm = tid;
                Xs[buf][0][mm] = (int32_t)xreg.x;
                Xs[buf][1][mm] = (int32_t)xreg.y;
                Xs[buf][2][mm] = (int32_t)xreg.z;
                Xs[buf][3][mm] = (int32_t)xreg.w;
            }
            if (tn_i < tn) {
                const int col = dtile * 16 + dc;
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int dmz = dm + 2 * z;
                    const uint32_t r0 = __shfl_sync(0xffffffffu, ringv, plan[z].m0w);
                    const uint32_t r1 = __shfl_sync(0xffffffffu, ringv, plan[z].m1w);
                    Ws[buf][dmz][col] = pf_decode4_words_q8(r0, r1, plan[z]);
                }
            }
        };
        float acc[RM][RN];
        int iacc[RM][RN];
#pragma unroll
        for (int im = 0; im < RM; ++im)
#pragma unroll
            for (int in = 0; in < RN; ++in) {
                acc[im][in] = 0.f;
                iacc[im][in] = 0;
            }
        load_tile(0);
        store_tile(0);
        __syncthreads();
        for (int tk_i = 0; tk_i < tk; ++tk_i) {
            const int cur = tk_i & 1;
            const bool have_next = tk_i + 1 < tk;
            if (have_next) load_tile(tk_i + 1);

            if (ty * RM < rcnt) {
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4) {
                    const int4 xv = *(const int4*)&Xs[cur][k4][ty * RM];
                    const int4 wq = *(const int4*)&Ws[cur][k4][tx * 4];
                    const uint32_t wv[RN] = {(uint32_t)wq.x, (uint32_t)wq.y, (uint32_t)wq.z, (uint32_t)wq.w};
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        iacc[0][in] = pf_dp4a_s32_s32(xv.x, wv[in], iacc[0][in]);
                        iacc[1][in] = pf_dp4a_s32_s32(xv.y, wv[in], iacc[1][in]);
                        iacc[2][in] = pf_dp4a_s32_s32(xv.z, wv[in], iacc[2][in]);
                        iacc[3][in] = pf_dp4a_s32_s32(xv.w, wv[in], iacc[3][in]);
                    }
                }
            }
            if ((tk_i & 7) == 7) {
                const int b128 = tk_i >> 3;
#pragma unroll
                for (int im = 0; im < RM; ++im) {
                    const int lr = rbase + ty * RM + im;
                    const float sc = (lr < rows)
                        ? xscale[(size_t)(row0 + lr) * kb + b128] : 0.f;
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        acc[im][in] = fmaf((float)iacc[im][in], sc, acc[im][in]);
                        iacc[im][in] = 0;
                    }
                }
            }
            if (have_next) store_tile(cur ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int im = 0; im < RM; ++im) {
            const int lr = rbase + ty * RM + im;
            if (lr < rows) {
                const float off = KOFF_C * rowsum[row0 + lr];
#pragma unroll
                for (int in = 0; in < RN; ++in) {
                    const int gn = tn0 * 16 + tx * RN + in;
                    if (gn < N) y[(size_t)(row0 + lr) * N + gn] = fmaf(4.0f * KINV, acc[im][in], off);
                }
            }
        }
    }
}


// 64-row grouped INT8 core.  It is the dp4a analogue of the FP32 weight-reuse
// kernel: both 32-row halves share each decoded packed weight tile.
__global__ void __launch_bounds__(256) inkwell_exl3_moe_grouped_gemm64_dp4a_kernel(
        const uint64_t* __restrict__ descs,
        const int* __restrict__ tile_expert, const int* __restrict__ tile_start,
        const int* __restrict__ tile_count, const int* __restrict__ tile_n,
        const int32_t* __restrict__ xq, const float* __restrict__ xscale,
        const float* __restrict__ rowsum, float* __restrict__ y,
        int tk, int tn, int tile_base_index, int tile_mode) {
    if (blockIdx.x & 1) return;
    constexpr int BM = 64, BN = 128, RM = 4, RN = 4;
    const int mt = tile_base_index + blockIdx.y;
    if (mt >= *tile_n) return;
    const int ex = tile_expert[mt];
    const int row0 = tile_start[mt];
    const int rows = tile_count[mt];
    if (tile_mode == 1 && rows > 32) return;
    if (tile_mode == 2 && rows <= 32) return;
    const uint64_t* d = descs + ex * 4;
    const uint32_t* trel = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int hk = (int)((d[3] >> 8) & 1);
    const int tid = threadIdx.x;
    const int tn0 = blockIdx.x * 4;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF_C = 1024.f * KINV + __half2float(__ushort_as_half(0xc931)) + 512.f * KINV;
    const int N = tn * 16;
    const int kb = tk >> 3;
    __shared__ __align__(16) uint32_t Ws[2][4][BN + 4];
    __shared__ __align__(16) int32_t Xs[2][4][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }
    const int ty = tid >> 5;
    const int tx = tid & 31;
    for (int rbase = 0; rbase < rows; rbase += BM) {
        const int rcnt = min(BM, rows - rbase);
        uint4 xreg = make_uint4(0u, 0u, 0u, 0u);
        uint32_t ringv = 0;
        auto load_tile = [&](int tk_i) {
            const int tn_i = tn0 + dtile;
            ringv = 0;
            if (tn_i < tn && du < w)
                ringv = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
            if (tid < BM) {
                const int mm = tid;
                xreg = (mm < rcnt)
                    ? *(const uint4*)(xq + (size_t)(row0 + rbase + mm) * (tk * 4) + tk_i * 4)
                    : make_uint4(0u, 0u, 0u, 0u);
            }
        };
        auto store_tile = [&](int buf) {
            const int tn_i = tn0 + dtile;
            if (tid < BM) {
                const int mm = tid;
                Xs[buf][0][mm] = (int32_t)xreg.x;
                Xs[buf][1][mm] = (int32_t)xreg.y;
                Xs[buf][2][mm] = (int32_t)xreg.z;
                Xs[buf][3][mm] = (int32_t)xreg.w;
            }
            if (tn_i < tn) {
                const int col = dtile * 16 + dc;
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int dmz = dm + 2 * z;
                    const uint32_t r0 = __shfl_sync(0xffffffffu, ringv, plan[z].m0w);
                    const uint32_t r1 = __shfl_sync(0xffffffffu, ringv, plan[z].m1w);
                    Ws[buf][dmz][col] = pf_decode4_words_q8(r0, r1, plan[z]);
                }
            }
        };
        float acc[2][RM][RN];
        int iacc[2][RM][RN];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int im = 0; im < RM; ++im)
#pragma unroll
                for (int in = 0; in < RN; ++in) {
                    acc[h][im][in] = 0.f;
                    iacc[h][im][in] = 0;
                }
        load_tile(0);
        store_tile(0);
        __syncthreads();
        for (int tk_i = 0; tk_i < tk; ++tk_i) {
            const int cur = tk_i & 1;
            const bool have_next = tk_i + 1 < tk;
            if (have_next) load_tile(tk_i + 1);
            if (rcnt > 32) {
                const bool valid1 = 32 + ty * RM < rcnt;
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4) {
                    const int4 x0v = *(const int4*)&Xs[cur][k4][ty * RM];
                    const int4 x1v = *(const int4*)&Xs[cur][k4][32 + ty * RM];
                    const int4 wq = *(const int4*)&Ws[cur][k4][tx * 4];
                    const uint32_t wv[RN] = {(uint32_t)wq.x, (uint32_t)wq.y, (uint32_t)wq.z, (uint32_t)wq.w};
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        iacc[0][0][in] = pf_dp4a_s32_s32(x0v.x, wv[in], iacc[0][0][in]);
                        iacc[0][1][in] = pf_dp4a_s32_s32(x0v.y, wv[in], iacc[0][1][in]);
                        iacc[0][2][in] = pf_dp4a_s32_s32(x0v.z, wv[in], iacc[0][2][in]);
                        iacc[0][3][in] = pf_dp4a_s32_s32(x0v.w, wv[in], iacc[0][3][in]);
                        if (valid1) {
                            iacc[1][0][in] = pf_dp4a_s32_s32(x1v.x, wv[in], iacc[1][0][in]);
                            iacc[1][1][in] = pf_dp4a_s32_s32(x1v.y, wv[in], iacc[1][1][in]);
                            iacc[1][2][in] = pf_dp4a_s32_s32(x1v.z, wv[in], iacc[1][2][in]);
                            iacc[1][3][in] = pf_dp4a_s32_s32(x1v.w, wv[in], iacc[1][3][in]);
                        }
                    }
                }
            } else if (ty * RM < rcnt) {
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4) {
                    const int4 x0v = *(const int4*)&Xs[cur][k4][ty * RM];
                    const int4 wq = *(const int4*)&Ws[cur][k4][tx * 4];
                    const uint32_t wv[RN] = {(uint32_t)wq.x, (uint32_t)wq.y, (uint32_t)wq.z, (uint32_t)wq.w};
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        iacc[0][0][in] = pf_dp4a_s32_s32(x0v.x, wv[in], iacc[0][0][in]);
                        iacc[0][1][in] = pf_dp4a_s32_s32(x0v.y, wv[in], iacc[0][1][in]);
                        iacc[0][2][in] = pf_dp4a_s32_s32(x0v.z, wv[in], iacc[0][2][in]);
                        iacc[0][3][in] = pf_dp4a_s32_s32(x0v.w, wv[in], iacc[0][3][in]);
                    }
                }
            }
            if ((tk_i & 7) == 7) {
                const int b128 = tk_i >> 3;
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int im = 0; im < RM; ++im) {
                        const int lr = rbase + h * 32 + ty * RM + im;
                        const float sc = (lr < rows)
                            ? xscale[(size_t)(row0 + lr) * kb + b128] : 0.f;
#pragma unroll
                        for (int in = 0; in < RN; ++in) {
                            acc[h][im][in] = fmaf((float)iacc[h][im][in], sc, acc[h][im][in]);
                            iacc[h][im][in] = 0;
                        }
                    }
            }
            if (have_next) store_tile(cur ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int im = 0; im < RM; ++im) {
                const int lr = rbase + h * 32 + ty * RM + im;
                if (lr < rows) {
                    const float off = KOFF_C * rowsum[row0 + lr];
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        const int gn = tn0 * 16 + tx * RN + in;
                        if (gn < N) y[(size_t)(row0 + lr) * N + gn] = fmaf(4.0f * KINV, acc[h][im][in], off);
                    }
                }
            }
    }
}

// Experimental 64-row grouped core.  It keeps one decoded weight tile in
// shared memory live for both 32-row halves of a tile instead of rereading the
// compressed weights for the second half.  The launch contract and output
// order are the same as inkwell_exl3_moe_grouped_gemm_kernel.
// Experimental 64-row grouped core.  It keeps one decoded weight tile in
// shared memory live for both 32-row halves of a tile instead of rereading the
// compressed weights for the second half.  The launch contract and output
// order are the same as inkwell_exl3_moe_grouped_gemm_kernel.
__global__ void __launch_bounds__(256) inkwell_exl3_moe_grouped_gemm64_kernel(
        const uint64_t* __restrict__ descs,
        const int* __restrict__ tile_expert, const int* __restrict__ tile_start,
        const int* __restrict__ tile_count, const int* __restrict__ tile_n,
        const float* __restrict__ xh, const float* __restrict__ rowsum, float* __restrict__ y,
        int tk, int tn, int tile_base_index, int tile_mode) {
    if (blockIdx.x & 1) return;
    constexpr int BM = 64, BN = 128, RM = 4, RN = 4;
    const int mt = tile_base_index + blockIdx.y;
    if (mt >= *tile_n) return;
    const int ex = tile_expert[mt];
    const int row0 = tile_start[mt];
    const int rows = tile_count[mt];
    if (tile_mode == 1 && rows > 32) return;
    if (tile_mode == 2 && rows <= 32) return;
    const uint64_t* d = descs + ex * 4;
    const uint32_t* trel = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int hk = (int)((d[3] >> 8) & 1);
    const int tid = threadIdx.x;
    const int tn0 = blockIdx.x * 4;
    const int w = pf_rbits_words(ka, hk);
    const int r_bits = hk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    __shared__ __align__(16) float Ws[2][16][BN + 4];
    __shared__ __align__(16) float Xs[2][16][BM + 4];

    const int dtile = tid >> 5;
    const int du = tid & 31;
    const int dc = du >> 1;
    const int dm = du & 1;
    PfDecodePlan plan[2];
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const int dmz = dm + 2 * z;
        const int i0 = 32 * (dc & 7) + 8 * dmz + 4 * (dc >> 3);
        plan[z] = pf_decode_plan(i0, ka, hk, r_bits, nw);
    }

    const int ty = tid >> 5;
    const int tx = tid & 31;
    const int K = tk * 16;
    const int N = tn * 16;
    for (int rbase = 0; rbase < rows; rbase += BM) {
        const int rcnt = min(BM, rows - rbase);
        float4 xreg = make_float4(0.f, 0.f, 0.f, 0.f);
        uint32_t ringv = 0;
        auto load_tile = [&](int tk_i) {
            const int tn_i = tn0 + dtile;
            ringv = 0;
            if (tn_i < tn && du < w) {
                ringv = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
            }
            if (tid < 4 * BM) {
                const int mm = tid >> 2;
                const int rq = tid & 3;
                xreg = (mm < rcnt)
                    ? *(const float4*)(xh + (size_t)(row0 + rbase + mm) * K + tk_i * 16 + rq * 4)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        };
        auto store_tile = [&](int buf) {
            const int tn_i = tn0 + dtile;
            if (tid < 4 * BM) {
                const int mm = tid >> 2;
                const int rq = tid & 3;
                Xs[buf][rq * 4][mm] = xreg.x;
                Xs[buf][rq * 4 + 1][mm] = xreg.y;
                Xs[buf][rq * 4 + 2][mm] = xreg.z;
                Xs[buf][rq * 4 + 3][mm] = xreg.w;
            }
            if (tn_i < tn) {
                const int col = dtile * 16 + dc;
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int dmz = dm + 2 * z;
                    float f0, f1, f2, f3;
                    const uint32_t r0 = __shfl_sync(0xffffffffu, ringv, plan[z].m0w);
                    const uint32_t r1 = __shfl_sync(0xffffffffu, ringv, plan[z].m1w);
                    pf_decode4_words(r0, r1, plan[z], f0, f1, f2, f3);
                    Ws[buf][2 * dmz][col] = f0;
                    Ws[buf][2 * dmz + 1][col] = f1;
                    Ws[buf][2 * dmz + 8][col] = f2;
                    Ws[buf][2 * dmz + 9][col] = f3;
                }
            }
        };
        float acc[2][RM][RN];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int im = 0; im < RM; ++im) {
#pragma unroll
                for (int in = 0; in < RN; ++in) acc[h][im][in] = 0.f;
            }
        load_tile(0);
        store_tile(0);
        __syncthreads();
        for (int tk_i = 0; tk_i < tk; ++tk_i) {
            const int cur = tk_i & 1;
            const bool have_next = tk_i + 1 < tk;
            if (have_next) load_tile(tk_i + 1);

            if (rcnt > 32) {
                const bool valid1 = 32 + ty * RM < rcnt;
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    const float4 wq = *(const float4*)&Ws[cur][r][tx * 4];
                    const float wv[RN] = {wq.x, wq.y, wq.z, wq.w};
                    const float4 x0v = *(const float4*)&Xs[cur][r][ty * RM];
                    const float4 x1v = *(const float4*)&Xs[cur][r][32 + ty * RM];
#pragma unroll
                    for (int im = 0; im < RM; ++im) {
                        const float x0 = (im == 0) ? x0v.x : (im == 1) ? x0v.y : (im == 2) ? x0v.z : x0v.w;
#pragma unroll
                        for (int in = 0; in < RN; ++in)
                            acc[0][im][in] = fmaf(x0, wv[in], acc[0][im][in]);
                        if (valid1) {
                            const float x1 = (im == 0) ? x1v.x : (im == 1) ? x1v.y : (im == 2) ? x1v.z : x1v.w;
#pragma unroll
                            for (int in = 0; in < RN; ++in)
                                acc[1][im][in] = fmaf(x1, wv[in], acc[1][im][in]);
                        }
                    }
                }
            } else {
                if (ty * RM < rcnt) {
#pragma unroll
                    for (int r = 0; r < 16; ++r) {
                        const float4 wq = *(const float4*)&Ws[cur][r][tx * 4];
                        const float wv[RN] = {wq.x, wq.y, wq.z, wq.w};
                        const float4 x0v = *(const float4*)&Xs[cur][r][ty * RM];
#pragma unroll
                        for (int im = 0; im < RM; ++im) {
                            const float x0 = (im == 0) ? x0v.x : (im == 1) ? x0v.y : (im == 2) ? x0v.z : x0v.w;
#pragma unroll
                            for (int in = 0; in < RN; ++in)
                                acc[0][im][in] = fmaf(x0, wv[in], acc[0][im][in]);
                        }
                    }
                }
            }
            if (have_next) store_tile(cur ^ 1);
            __syncthreads();
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int im = 0; im < RM; ++im) {
                const int lr = rbase + h * 32 + ty * RM + im;
                if (lr < rows) {
                    const float off = KOFF * rowsum[row0 + lr];
#pragma unroll
                    for (int in = 0; in < RN; ++in) {
                        const int gn = tn0 * 16 + tx * RN + in;
                        if (gn < N) y[(size_t)(row0 + lr) * N + gn] = fmaf(KINV, acc[h][im][in], off);
                    }
                }
            }
        }
    }
}

// SiTU followed by the down-projection input Hadamard.  gu is [P,2,ncols]
// with gate in the first half and up in the second.
extern "C" __global__ void inkwell_moe_situ_had_rows(
        const uint64_t* __restrict__ descs_d, const int* __restrict__ row_expert,
        const float* __restrict__ gu, float* __restrict__ xh,
        float beta, float lin_beta, int row_base, int P, int ncols) {
    const int row = row_base + blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int seg = blockIdx.x * (blockDim.x >> 5) + wid;
    const int base = seg * 128;
    if (blockIdx.y >= P || base >= ncols) return;
    const int ex = row_expert[row];
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs_d[ex * 4 + 1];
    const int e0 = base + lane * 4;
    const float4 gv = *(const float4*)(gu + (size_t)row * 2 * ncols + e0);
    const float4 uv = *(const float4*)(gu + (size_t)row * 2 * ncols + ncols + e0);
    const uint2 sv = *(const uint2*)(suh + e0);
    const float gf[4] = {gv.x, gv.y, gv.z, gv.w};
    const float uf[4] = {uv.x, uv.y, uv.z, uv.w};
    const uint16_t sh[4] = {
        (uint16_t)(sv.x & 0xffffu), (uint16_t)(sv.x >> 16),
        (uint16_t)(sv.y & 0xffffu), (uint16_t)(sv.y >> 16)
    };
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float g = gf[i];
        const float u = uf[i];
        const float a = (beta * tanhf(g / beta)) / (1.0f + expf(-g));
        const float b = lin_beta * tanhf(u / lin_beta);
        v[i] = __fmul_rn(__fmul_rn(a, b), __half2float(__ushort_as_half(sh[i])));
    }
    pf_had128_warp(v, lane * 4);
    *(float4*)(xh + (size_t)row * ncols + e0) = make_float4(
        __fmul_rn(v[0], PF_INV_SQRT128), __fmul_rn(v[1], PF_INV_SQRT128),
        __fmul_rn(v[2], PF_INV_SQRT128), __fmul_rn(v[3], PF_INV_SQRT128));
}

// Deterministic weighted scatter: each token reduces its eight slots in the
// original slot order.  There is no atomicAdd anywhere in the value path.
extern "C" __global__ void inkwell_moe_weighted_scatter(
        const float* __restrict__ dn, const float* __restrict__ sorted_weight,
        const int* __restrict__ inverse_row, float* __restrict__ out,
        int C, int n) {
    const int tok = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (tok >= C || e >= n) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int pair = tok * 8 + j;
        const int row = inverse_row[pair];
        acc += sorted_weight[row] * dn[(size_t)row * n + e];
    }
    out[(size_t)tok * n + e] = acc;
}

extern "C" void ink_launch_exl3_moe_group_sort(const int* ids, const float* weights, int C,
        int* expert_offsets, int* sorted_token, int* sorted_slot, float* sorted_weight,
        int* inverse_row, int* tile_expert, int* tile_start, int* tile_count, int* tile_n,
        void* stream) {
    inkwell_exl3_moe_group_sort<<<1, 128, 0, (cudaStream_t)stream>>>(ids, weights, C,
        expert_offsets, sorted_token, sorted_slot, sorted_weight, inverse_row,
        tile_expert, tile_start, tile_count, tile_n);
}

__global__ void inkwell_moe_fill_row_expert(const int*, const int*, const int*, const int*, int*);
__global__ void inkwell_moe_had_out_pair_rows(const uint64_t*, const uint64_t*, const int*, const float*, const float*, float*, int, int, int);

static void pf_launch_grouped_gemm_dp4a(bool bm64, dim3 g, const void* descs,
        const int* tile_expert, const int* tile_start, const int* tile_count, const int* tile_n,
        const int32_t* xq, const float* xscale, const float* rowsum, float* y,
        int tk, int tn, int tile_base, cudaStream_t st) {
    if (bm64) {
        inkwell_exl3_moe_grouped_gemm_dp4a_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
            tile_expert, tile_start, tile_count, tile_n, xq, xscale, rowsum, y, tk, tn, tile_base, 1);
        inkwell_exl3_moe_grouped_gemm64_dp4a_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
            tile_expert, tile_start, tile_count, tile_n, xq, xscale, rowsum, y, tk, tn, tile_base, 2);
    } else {
        inkwell_exl3_moe_grouped_gemm_dp4a_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
            tile_expert, tile_start, tile_count, tile_n, xq, xscale, rowsum, y, tk, tn, tile_base, 0);
    }
}

static void pf_launch_grouped_gemm_kernel(bool bm64, dim3 g, const void* descs,
        const int* tile_expert, const int* tile_start, const int* tile_count, const int* tile_n,
        const float* xh, const float* rowsum, float* y, int tk, int tn, int tile_base, cudaStream_t st) {
    if (bm64) {
        const char* ae = getenv("INKWELL_PF_MOE_ADAPT");
        if (ae == nullptr || ae[0] != '0') {
            inkwell_exl3_moe_grouped_gemm_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
                tile_expert, tile_start, tile_count, tile_n, xh, rowsum, y, tk, tn, tile_base, 1);
            inkwell_exl3_moe_grouped_gemm64_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
                tile_expert, tile_start, tile_count, tile_n, xh, rowsum, y, tk, tn, tile_base, 2);
        } else {
            inkwell_exl3_moe_grouped_gemm64_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
                tile_expert, tile_start, tile_count, tile_n, xh, rowsum, y, tk, tn, tile_base, 0);
        }
    } else {
        inkwell_exl3_moe_grouped_gemm_kernel<<<g, 256, 0, st>>>((const uint64_t*)descs,
            tile_expert, tile_start, tile_count, tile_n, xh, rowsum, y, tk, tn, tile_base, 0);
    }
}

// Full grouped MoE pipeline over rows already sorted by inkwell_exl3_moe_group_sort.
// Workspace layout (floats unless noted): A[P*2304], B[P*2304], T[P*2304],
// then row_expert[P] as int.  Stream ordering lets the short-lived arrays
// share those three regions: xhg=A, xhu=B, tg/tu=T, gu=A, xhd=B, td=A,
// dn=T.  Peak MoE scratch is therefore 3*P*2304 floats rather than the
// naive Phase-1 layout; Phase 2 still owns the arena and expert streaming.
extern "C" void ink_launch_exl3_moe_grouped_gemm_batch(
        const void* descs_g, const void* descs_u, const void* descs_d,
        const float* x, int C, int row_base, int batch_rows,
        int tile_base, int tile_batches, int do_scatter, int separate_dn,
        const int* sorted_token, const int* tile_expert, const int* tile_start,
        const int* tile_count, const int* tile_n,
        const float* sorted_weight, const int* inverse_row,
        float* workspace, float* out, void* stream) {
    const int P = C * 8;
    const int KG = 2304, NG = 1152, ND = 2304;
    cudaStream_t st = (cudaStream_t)stream;
    float* xhg = workspace;
    float* xhu = xhg + (size_t)P * KG;
    float* tg = xhu + (size_t)P * KG;
    float* tu = tg + (size_t)P * NG;
    float* gu = xhg;
    float* xhd = xhu;
    float* td = xhg;
    float* rowsum = out;
    int* row_expert = (int*)(tg + (size_t)P * (2 * NG));
    float* dn_sep = (float*)(((uintptr_t)(row_expert + P) + 15) & ~(uintptr_t)15);
    float* dn = separate_dn ? dn_sep : tg;
    // row_expert is derived on-device: one row per expanded selection.
    // tile_start/count are inclusive contiguous ranges; expand them with a tiny kernel below.
    inkwell_moe_fill_row_expert<<<256, 64, 0, st>>>(tile_expert, tile_start, tile_count, tile_n, row_expert);
    if (batch_rows > 0 && tile_batches > 0) {
    dim3 hig((KG / 128 + 3) / 4, batch_rows), hug((KG / 128 + 3) / 4, batch_rows);
    const char* bm64e = getenv("INKWELL_PF_MOE_BM64");
    const bool bm64 = bm64e == nullptr || bm64e[0] != '0';
    const bool q8 = ink_pf_int8_enabled();
    dim3 gg((NG / 16 + 3) / 4, tile_batches);
    dim3 gd((ND / 16 + 3) / 4, tile_batches);
    if (q8) {
        int32_t* xqg = (int32_t*)xhg;
        int32_t* xqu = (int32_t*)xhu;
        int32_t* xqd = (int32_t*)xhd;
        float* xsg = xhg + (size_t)P * KG / 4;
        float* xsu = xhu + (size_t)P * KG / 4;
        float* xsd = xhu + (size_t)P * NG / 4;
        cudaMemsetAsync(rowsum + row_base, 0, (size_t)batch_rows * 4, st);
        inkwell_moe_had_in_rows_q8<<<hig, 128, 0, st>>>((const uint64_t*)descs_g, row_expert, sorted_token, x,
            xqg, xsg, rowsum, row_base, batch_rows, KG);
        pf_launch_grouped_gemm_dp4a(bm64, gg, descs_g, tile_expert, tile_start, tile_count, tile_n,
            xqg, xsg, rowsum, tg, KG / 16, NG / 16, tile_base, st);
        cudaMemsetAsync(rowsum + row_base, 0, (size_t)batch_rows * 4, st);
        inkwell_moe_had_in_rows_q8<<<hug, 128, 0, st>>>((const uint64_t*)descs_u, row_expert, sorted_token, x,
            xqu, xsu, rowsum, row_base, batch_rows, KG);
        pf_launch_grouped_gemm_dp4a(bm64, gg, descs_u, tile_expert, tile_start, tile_count, tile_n,
            xqu, xsu, rowsum, tu, KG / 16, NG / 16, tile_base, st);
        dim3 hog((NG / 128 + 3) / 4, batch_rows, 2);
        inkwell_moe_had_out_pair_rows<<<hog, 128, 0, st>>>((const uint64_t*)descs_g, (const uint64_t*)descs_u,
            row_expert, tg, tu, gu, row_base, batch_rows, NG);
        dim3 hd((NG / 128 + 3) / 4, batch_rows);
        cudaMemsetAsync(rowsum + row_base, 0, (size_t)batch_rows * 4, st);
        inkwell_moe_situ_had_rows_q8<<<hd, 128, 0, st>>>((const uint64_t*)descs_d, row_expert, gu,
            xqd, xsd, rowsum, 4.f, 25.f, row_base, batch_rows, NG);
        pf_launch_grouped_gemm_dp4a(bm64, gd, descs_d, tile_expert, tile_start, tile_count, tile_n,
            xqd, xsd, rowsum, td, NG / 16, ND / 16, tile_base, st);
        dim3 hod((ND / 128 + 3) / 4, batch_rows);
        inkwell_moe_had_out_rows<<<hod, 128, 0, st>>>((const uint64_t*)descs_d, row_expert, td, dn, row_base, batch_rows, ND);
    } else {
    inkwell_moe_had_in_rows<<<hig, 128, 0, st>>>((const uint64_t*)descs_g, row_expert, sorted_token, x, xhg, row_base, batch_rows, KG);
    pf_launch_moe_rowsum(xhg, rowsum, row_base, batch_rows, KG, st);
    pf_launch_grouped_gemm_kernel(bm64, gg, descs_g,
        tile_expert, tile_start, tile_count, tile_n, xhg, rowsum, tg, KG / 16, NG / 16, tile_base, st);
    inkwell_moe_had_in_rows<<<hug, 128, 0, st>>>((const uint64_t*)descs_u, row_expert, sorted_token, x, xhu, row_base, batch_rows, KG);
    pf_launch_moe_rowsum(xhu, rowsum, row_base, batch_rows, KG, st);
    pf_launch_grouped_gemm_kernel(bm64, gg, descs_u,
        tile_expert, tile_start, tile_count, tile_n, xhu, rowsum, tu, KG / 16, NG / 16, tile_base, st);
    dim3 hog((NG / 128 + 3) / 4, batch_rows, 2);
    // Write gate into gu[:,0,:] and up into gu[:,1,:] using a strided temporary layout.
    // tg/tu are contiguous rows; the epilogue kernels below interleave them into gu halves.
    inkwell_moe_had_out_pair_rows<<<hog, 128, 0, st>>>((const uint64_t*)descs_g, (const uint64_t*)descs_u,
        row_expert, tg, tu, gu, row_base, batch_rows, NG);
    dim3 hd((NG / 128 + 3) / 4, batch_rows);
    inkwell_moe_situ_had_rows<<<hd, 128, 0, st>>>((const uint64_t*)descs_d, row_expert, gu, xhd, 4.f, 25.f, row_base, batch_rows, NG);
    pf_launch_moe_rowsum(xhd, rowsum, row_base, batch_rows, NG, st);
    pf_launch_grouped_gemm_kernel(bm64, gd, descs_d,
        tile_expert, tile_start, tile_count, tile_n, xhd, rowsum, td, NG / 16, ND / 16, tile_base, st);
    dim3 hod((ND / 128 + 3) / 4, batch_rows);
    inkwell_moe_had_out_rows<<<hod, 128, 0, st>>>((const uint64_t*)descs_d, row_expert, td, dn, row_base, batch_rows, ND);
    }
    }
    if (do_scatter) {
        dim3 gs((ND + 255) / 256, C);
        inkwell_moe_weighted_scatter<<<gs, 256, 0, st>>>(dn, sorted_weight, inverse_row, out, C, ND);
    }
}

extern "C" void ink_launch_exl3_moe_grouped_gemm(
        const void* descs_g, const void* descs_u, const void* descs_d,
        const float* x, int C,
        const int* sorted_token, const int* tile_expert, const int* tile_start,
        const int* tile_count, const int* tile_n,
        const float* sorted_weight, const int* inverse_row,
        float* workspace, float* out, void* stream) {
    const int P = C * 8;
    const int max_tiles = P / 64 + 128;
    ink_launch_exl3_moe_grouped_gemm_batch(descs_g, descs_u, descs_d, x, C, 0, P,
        0, max_tiles, 1, 0, sorted_token, tile_expert, tile_start, tile_count, tile_n,
        sorted_weight, inverse_row, workspace, out, stream);
}

__global__ void inkwell_moe_fill_row_expert(const int* tile_expert, const int* tile_start,
        const int* tile_count, const int* tile_n, int* row_expert) {
    const int tn = *tile_n;
    for (int t = blockIdx.x; t < tn; t += gridDim.x) {
        const int ex = tile_expert[t];
        const int st = tile_start[t];
        const int cnt = tile_count[t];
        for (int i = threadIdx.x; i < cnt; i += blockDim.x)
            row_expert[st + i] = ex;
    }
}

__global__ void inkwell_moe_had_out_pair_rows(const uint64_t* descs_g, const uint64_t* descs_u,
        const int* row_expert, const float* tg, const float* tu, float* gu, int row_base, int P, int n) {
    const int row = row_base + blockIdx.y;
    const int half = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int seg = blockIdx.x * (blockDim.x >> 5) + wid;
    const int base = seg * 128;
    if (blockIdx.y >= P || base >= n) return;
    const int ex = row_expert[row];
    const uint64_t* descs = half ? descs_u : descs_g;
    const uint16_t* svh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 2];
    const float* tm = half ? tu : tg;
    const int e0 = base + lane * 4;
    const float4 tv = *(const float4*)(tm + (size_t)row * n + e0);
    float v[4] = {tv.x, tv.y, tv.z, tv.w};
    pf_had128_warp(v, lane * 4);
    const uint2 sv = *(const uint2*)(svh + e0);
    *(float4*)(gu + (size_t)row * 2 * n + half * n + e0) = make_float4(
        __fmul_rn(__fmul_rn(v[0], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu)))),
        __fmul_rn(__fmul_rn(v[1], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.x >> 16)))),
        __fmul_rn(__fmul_rn(v[2], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu)))),
        __fmul_rn(__fmul_rn(v[3], PF_INV_SQRT128), __half2float(__ushort_as_half((uint16_t)(sv.y >> 16)))));
}

// ---------------- parallel causal FIR convolution ----------------
extern "C" __global__ void inkwell_conv_prefill(
        const float* __restrict__ x, float* __restrict__ state,
        int region_off, const float* __restrict__ w, const float* __restrict__ dyn,
        int C, int ch, int n_heads, float scale, float* __restrict__ out) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C || c >= ch) return;
    const float* st = state + 3 * (size_t)region_off;
    float xs[4];
    // x logical rows: oldest three precede the chunk, then x[0..C).
    for (int r = 0; r < 4; ++r) {
        const int q = t + r - 3;
        xs[r] = (q >= 0) ? x[(size_t)q * ch + c] : st[3 * c + q + 3];
    }
    float acc = w[4 * c] * xs[0] + w[4 * c + 1] * xs[1] + w[4 * c + 2] * xs[2] + w[4 * c + 3] * xs[3];
    if (dyn) {
        const int h = c / (ch / n_heads);
        const float* dr = dyn + (size_t)t * 4 * n_heads + 4 * h;
        acc += dr[0] * xs[3] + dr[1] * xs[2] + dr[2] * xs[1] + dr[3] * xs[0];
    }
    out[(size_t)t * ch + c] = (xs[3] + acc) * scale;
}

// State update is a separate kernel launch: rows 0..2 of the parallel FIR read
// the old state, so updating it in the same grid would be a cross-block race.
extern "C" __global__ void inkwell_conv_prefill_state_update(
        const float* __restrict__ x, float* __restrict__ state,
        int region_off, int C, int ch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    const float* st = state + 3 * (size_t)region_off;
    float* sw = state + 3 * (size_t)region_off;
    const float a = st[3 * c], b = st[3 * c + 1], d = st[3 * c + 2];
    sw[3 * c] = (C >= 3) ? x[(size_t)(C - 3) * ch + c] : ((C == 1) ? b : d);
    sw[3 * c + 1] = (C >= 2) ? x[(size_t)(C - 2) * ch + c] : d;
    sw[3 * c + 2] = x[(size_t)(C - 1) * ch + c];
    (void)a;
}

extern "C" void ink_launch_conv_prefill(const float* x, float* state, int region_off,
        const float* w, const float* dyn, int C, int ch, int n_heads, float scale,
        float* out, void* stream) {
    dim3 g((ch + 255) / 256, C);
    inkwell_conv_prefill<<<g, 256, 0, (cudaStream_t)stream>>>(x, state, region_off, w, dyn,
        C, ch, n_heads, scale, out);
    inkwell_conv_prefill_state_update<<<(ch + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        x, state, region_off, C, ch);
}


// ---------------- quantized-KV read support ----------------
// Layout mirror of InkKvQSlot in kernels.cu.  Prefill uses it read-only; the
// batched writer itself lives in kernels.cu.
#define INK_KVQ_INT8 1
#define INK_KVQ_FP8_E4M3 2
#define INK_KVQ_K8V4 3
#define INK_KVQ_K4V2 4
#define INK_KVQ_FP8_E5M2 5
#define INK_KVQ_K8V4G32 6
#define INK_KVQ_K8V6 7
#define INK_KVQ_K10V6 8
struct PfKvQSlot {
    unsigned char* kq;
    unsigned char* vq;
    unsigned char* meta;
    __half* khp;
    __half* vhp;
    int cap;
    int hp_cap;
    int swa;
    int sinks;
    int local;
    int is_swa;
    int fmt;
    int krow;
    int vrow;
    int mrow;
};

__device__ __forceinline__ float pf_fp8_e4m3_f(unsigned char b) {
    const float sign = (b & 0x80u) ? -1.f : 1.f;
    const int e = (b >> 3) & 15;
    const int m = b & 7;
    if (e == 0) return sign * (float)m * 0.001953125f;
    const int hb = ((b & 0x80u) << 8) | ((e + 8) << 10) | (m << 7);
    return __half2float(__ushort_as_half((unsigned short)hb));
}
__device__ __forceinline__ float pf_fp8_e5m2_f(unsigned char b) {
    const int hb = ((b & 0x80u) << 8) | ((b & 0x7fu) << 8);
    return __half2float(__ushort_as_half((unsigned short)hb));
}
__device__ __forceinline__ float pf_fp8_dequant(unsigned char x, int fmt) {
    return fmt == INK_KVQ_FP8_E5M2 ? pf_fp8_e5m2_f(x) : pf_fp8_e4m3_f(x);
}

__device__ __forceinline__ int pf_k8v4_tile_rem(int cap, int tile) {
    return max(0, min(16, cap - tile * 16));
}
__device__ __forceinline__ int pf_k8v4_row_units(int rem, int s) {
    return s < rem ? (s + 8 < rem ? 2 : 1) : 0;
}
__device__ __forceinline__ int pf_k8v4_units_before(int rem, int s) {
    int u = 0;
    #pragma unroll
    for (int z = 0; z < 8; ++z) if (z < s) u += pf_k8v4_row_units(rem, z);
    return u;
}
__device__ __forceinline__ int pf_k8v4_dim_unit(int cap, int qrow, int sl) {
    const int tile = qrow >> 4;
    const int t = qrow & 15;
    const int rem = pf_k8v4_tile_rem(cap, tile);
    const int s = t & 7;
    const int cnt = pf_k8v4_row_units(rem, s);
    return pf_k8v4_units_before(rem, s) * 16 + sl * cnt + (t >> 3);
}
__device__ __forceinline__ size_t pf_k8v4_koff(const PfKvQSlot& q, int qrow, int hk, int d) {
    const int tile = qrow >> 4;
    return (size_t)hk * q.cap * 128 + (size_t)tile * 2048 + pf_k8v4_dim_unit(q.cap, qrow, d >> 3) * 8 + (d & 7);
}
__device__ __forceinline__ size_t pf_k8v4_voff(const PfKvQSlot& q, int qrow, int hk, int vb) {
    const int tile = qrow >> 4;
    return (size_t)hk * q.cap * 64 + (size_t)tile * 1024 + pf_k8v4_dim_unit(q.cap, qrow, vb >> 2) * 4 + (vb & 3);
}
__device__ __forceinline__ size_t pf_k8v4_moff(const PfKvQSlot& q, int qrow, int hk) {
    const int tile = qrow >> 4;
    const int t = qrow & 15;
    const int rem = pf_k8v4_tile_rem(q.cap, tile);
    const int ru = pf_k8v4_units_before(rem, t & 7) + (t >> 3);
    return (size_t)hk * q.cap * 12 + (size_t)tile * 192 + ru * 12;
}
__device__ __forceinline__ bool pf_kvq_is_hp(const PfKvQSlot& q, int logical, int posref) {
    return q.hp_cap > 0 && (logical < q.sinks || logical >= posref - q.local + 1);
}
__device__ __forceinline__ int pf_kvq_hp_idx(const PfKvQSlot& q, int logical) {
    return logical < q.sinks ? logical : q.sinks + (logical - q.sinks) % q.local;
}
__device__ __forceinline__ float pf_kvq_meta_h(const PfKvQSlot& q, int qrow, int hk, int idx) {
    const unsigned char* mp = q.fmt == INK_KVQ_K8V4 ? q.meta + pf_k8v4_moff(q, qrow, hk) :
        (q.meta + (size_t)qrow * q.mrow + hk * ((q.fmt == INK_KVQ_K8V4G32) ? 24 : 12));
    const __half* mh = (const __half*)mp;
    return __half2float(mh[idx]);
}
__device__ __forceinline__ float pf_kvq_read_k(const PfKvQSlot& q, int logical, int qrow, int hk, int d, int posref) {
    if (pf_kvq_is_hp(q, logical, posref)) {
        const int hi = pf_kvq_hp_idx(q, logical);
        return __half2float(q.khp[((size_t)hk * q.hp_cap + hi) * 128 + d]);
    }
    const int grp = d >> 6;
    if (q.fmt == INK_KVQ_INT8 || q.fmt == INK_KVQ_K8V4 || q.fmt == INK_KVQ_K8V4G32 || q.fmt == INK_KVQ_K8V6) {
        const size_t off = q.fmt == INK_KVQ_K8V4 ? pf_k8v4_koff(q, qrow, hk, d) : ((size_t)qrow * q.krow + hk * 128 + d);
        int x = (int)(signed char)q.kq[off];
        return (float)x * pf_kvq_meta_h(q, qrow, hk, grp);
    }
    if (q.fmt == INK_KVQ_K10V6) {
        const unsigned char* kp = q.kq + (size_t)qrow * q.krow + hk * 160;
        const int hi = (int)(signed char)kp[d];
        const int lo = (kp[128 + (d >> 2)] >> ((d & 3) * 2)) & 3;
        const int x = hi * 4 + lo;
        return (float)x * pf_kvq_meta_h(q, qrow, hk, grp);
    }
    if (q.fmt == INK_KVQ_K4V2) {
        const unsigned char b = q.kq[(size_t)qrow * q.krow + hk * 64 + (d >> 1)];
        int x = (b >> ((d & 1) * 4)) & 15;
        if (x >= 8) x -= 16;
        return (float)x * pf_kvq_meta_h(q, qrow, hk, grp);
    }
    return pf_fp8_dequant(q.kq[(size_t)qrow * q.krow + hk * 128 + d], q.fmt);
}
__device__ __forceinline__ float pf_kvq_read_v(const PfKvQSlot& q, int logical, int qrow, int hk, int d, int posref) {
    if (pf_kvq_is_hp(q, logical, posref)) {
        const int hi = pf_kvq_hp_idx(q, logical);
        return __half2float(q.vhp[((size_t)hk * q.hp_cap + hi) * 128 + d]);
    }
    const int grp = d >> 6;
    if (q.fmt == INK_KVQ_INT8) {
        int x = (int)(signed char)q.vq[(size_t)qrow * q.vrow + hk * 128 + d];
        return (float)x * pf_kvq_meta_h(q, qrow, hk, 2 + grp * 2);
    }
    if (q.fmt == INK_KVQ_K8V4 || q.fmt == INK_KVQ_K8V4G32) {
        const size_t off = q.fmt == INK_KVQ_K8V4 ? pf_k8v4_voff(q, qrow, hk, d >> 1) : ((size_t)qrow * q.vrow + hk * 64 + (d >> 1));
        const unsigned char b = q.vq[off];
        const int x = (b >> ((d & 1) * 4)) & 15;
        const int vgrp = q.fmt == INK_KVQ_K8V4G32 ? d >> 5 : grp;
        return (float)x * pf_kvq_meta_h(q, qrow, hk, 2 + vgrp * 2) + pf_kvq_meta_h(q, qrow, hk, 3 + vgrp * 2);
    }
    if (q.fmt == INK_KVQ_K8V6 || q.fmt == INK_KVQ_K10V6) {
        const unsigned char* vp = q.vq + (size_t)qrow * q.vrow + hk * 96;
        const int lo = (vp[d >> 1] >> ((d & 1) * 4)) & 15;
        const int hi = (vp[64 + (d >> 2)] >> ((d & 3) * 2)) & 3;
        const int x = lo | (hi << 4);
        return (float)x * pf_kvq_meta_h(q, qrow, hk, 2 + grp * 2) + pf_kvq_meta_h(q, qrow, hk, 3 + grp * 2);
    }
    if (q.fmt == INK_KVQ_K4V2) {
        const unsigned char b = q.vq[(size_t)qrow * q.vrow + hk * 32 + (d >> 2)];
        const int x = (b >> ((d & 3) * 2)) & 3;
        return (float)x * pf_kvq_meta_h(q, qrow, hk, 2 + grp * 2) + pf_kvq_meta_h(q, qrow, hk, 3 + grp * 2);
    }
    return pf_fp8_dequant(q.vq[(size_t)qrow * q.vrow + hk * 128 + d], q.fmt);
}

// Snapshot rows that a SWA write is about to overwrite, dequantizing them to
// the same fp16 layout as the ordinary snapshot path.
extern "C" __global__ void inkwell_pf_snapshot_kvq(PfKvQSlot qslot,
        void* __restrict__ ks, void* __restrict__ vs,
        int snap_start, int snap_len, int cap, int swa, int posref) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= snap_len || i >= 1024) return;
    const int h = i >> 7, d = i & 127;
    const int logical = snap_start + row;
    const int slot = logical % swa;
    const size_t dst = ((size_t)h * snap_len + row) * 128 + d;
    ((__half*)ks)[dst] = __float2half_rn(pf_kvq_read_k(qslot, logical, slot, h, d, posref));
    ((__half*)vs)[dst] = __float2half_rn(pf_kvq_read_v(qslot, logical, slot, h, d, posref));
}
extern "C" void ink_launch_pf_snapshot_kvq(const PfKvQSlot* qslot, void* ks, void* vs,
        int snap_start, int snap_len, int cap, int swa, int posref, void* stream) {
    dim3 g(4, snap_len);
    inkwell_pf_snapshot_kvq<<<g, 256, 0, (cudaStream_t)stream>>>(*qslot, ks, vs,
        snap_start, snap_len, cap, swa, posref);
}

// ---------------- GQA-3 flash prefill attention ----------------
// Correctness-first GQA-3 kernel.  One CTA owns a 32-query tile and one KV
// head; K/V tiles are loaded once and reused for all three query heads.
// rel_w layout is [nh][1025][16]; r layout is [C,nh,16].  sinks is [C,nh].
// The effective key bias is dot(r[q,h], rel_w[h][dist]) - sinks[q,h]; the
// sink column is the fixed zero reference.  SWA physical slot = logical_pos % swa_win.
// USE_Q is compile-time so the F16 instantiation carries no quantized-read branch.
template <int USE_Q>
__global__ void __launch_bounds__(128) inkwell_flash_prefill_gqa3(
        const __half* __restrict__ K, const __half* __restrict__ V,
        const float* __restrict__ Q, const float* __restrict__ r,
        const float* __restrict__ rel_w, const float* __restrict__ sinks,
        float* __restrict__ out, int C, int pos0, int kv_cap, int swa_win, float scale,
        const __half* __restrict__ KS, const __half* __restrict__ VS, int snap_start, int snap_len,
        PfKvQSlot qslot) {
    // Register-tiled flash attention.  One CTA owns BQ=32 queries for one KV
    // head and reuses every BK=32 K/V tile for the three GQA query heads.
    // QK uses a 4x4 register tile on 64 threads (two shared-memory loads per
    // four FMAs); PV uses a 4x8 register tile on all 128 threads with bank-
    // conflict-free strided columns.  Kf/Vf share one smem allocation.
    constexpr int BQ = 32, BK = 32;
    const int posref = pos0 + C - 1;
    const int q0 = blockIdx.x * BQ;
    const int hk = blockIdx.y;
    const int tid = threadIdx.x;
    __shared__ float Qs[128][36];   // transposed: [dim][query], vector aligned
    __shared__ __half KVh[128][36];  // transposed: [dim][key], vector aligned
    __shared__ float S[BQ][BK + 1];
    __shared__ float Mstate[BQ], Lstate[BQ], Alpha[BQ];

    const __half* Kh = K + (size_t)hk * kv_cap * 128;
    const __half* Vh = V + (size_t)hk * kv_cap * 128;
    const int tx = tid & 15;          // PV: 16 strided column groups
    const int ty = tid >> 4;          // PV: 8 groups of 4 query rows

    const int max_qpos = pos0 + min(q0 + BQ - 1, C - 1);
    const int min_kpos = (swa_win > 0) ? max(0, pos0 + q0 - swa_win + 1) : 0;
    const int k_start = (min_kpos / BK) * BK;
    const int k_end = max_qpos + 1;

    for (int hh = 0; hh < 3; ++hh) {
        const int h = hk * 3 + hh;
        for (int idx = tid; idx < BQ * 128; idx += 128) {
            const int qr = idx >> 7, d = idx & 127;
            const int gq = q0 + qr;
            Qs[d][qr] = (gq < C) ? Q[((size_t)gq * 24 + h) * 128 + d] : 0.f;
        }
        if (tid < BQ) {
            Mstate[tid] = 0.f;           // sink column: fixed zero reference, weight one
            Lstate[tid] = 1.f;
        }
        __syncthreads();

        float acc[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;

        for (int k0 = k_start; k0 < k_end; k0 += BK) {
            const int kcnt = min(BK, k_end - k0);
            for (int idx = tid; idx < kcnt * 32; idx += 128) {
                const int kr = idx >> 5, d = (idx & 31) * 4;
                const int logical = k0 + kr;
                const int slot = (swa_win > 0) ? (logical % swa_win) : logical;
                const bool osnap = KS && logical >= snap_start && logical < snap_start + snap_len;
                if (USE_Q && !osnap) {
                    KVh[d][kr] = __float2half_rn(pf_kvq_read_k(qslot, logical, slot, hk, d, posref));
                    KVh[d + 1][kr] = __float2half_rn(pf_kvq_read_k(qslot, logical, slot, hk, d + 1, posref));
                    KVh[d + 2][kr] = __float2half_rn(pf_kvq_read_k(qslot, logical, slot, hk, d + 2, posref));
                    KVh[d + 3][kr] = __float2half_rn(pf_kvq_read_k(qslot, logical, slot, hk, d + 3, posref));
                } else {
                    const __half* kbase = osnap ? KS + (size_t)hk * snap_len * 128 : Kh;
                    const size_t krow = osnap ? (size_t)(logical - snap_start) : (size_t)slot;
                    const uint2 packed = *(const uint2*)(kbase + krow * 128 + d);
                    KVh[d][kr] = __ushort_as_half((unsigned short)(packed.x & 0xffffu));
                    KVh[d + 1][kr] = __ushort_as_half((unsigned short)(packed.x >> 16));
                    KVh[d + 2][kr] = __ushort_as_half((unsigned short)(packed.y & 0xffffu));
                    KVh[d + 3][kr] = __ushort_as_half((unsigned short)(packed.y >> 16));
                }
            }
            __syncthreads();

            // QK tile: all four warps compute 4x2 register tiles (8 outputs
            // per thread).  The bias tile is then computed by all warps before
            // one barrier; this keeps every scheduler busy in the dominant dot
            // loop instead of reserving half the CTA for the much smaller bias.
            float aqk[4][2];
            {
                const int qtx = tid & 15;  // groups of two keys
                const int qty = tid >> 4;  // groups of four query rows
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) aqk[i][j] = 0.f;
#pragma unroll 4
                for (int d = 0; d < 128; ++d) {
                    const float4 q4 = *(const float4*)&Qs[d][qty * 4];
                    const uint32_t kh = *(const uint32_t*)&KVh[d][qtx * 2];
                    const __half2 k01 = *(const __half2*)&kh;
                    const float qv[4] = {q4.x, q4.y, q4.z, q4.w};
                    const float kv[2] = {__low2float(k01), __high2float(k01)};
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 2; ++j)
                            aqk[i][j] = fmaf(qv[i], kv[j], aqk[i][j]);
                }

                // Bias tile: four threads per query row, each covering eight
                // contiguous keys.  r[16] and the sink are hoisted out of the
                // key loop; r/rel_w rows are aligned float4 vectors.
                const int qr = tid >> 2;
                const int kb = (tid & 3) * 8;
                const int gq = q0 + qr;
                const int qp = pos0 + gq;
                float rv[16];
                float snk = 0.f;
                if (gq < C) {
                    const float* rp = r + ((size_t)gq * 24 + h) * 16;
#pragma unroll
                    for (int z4 = 0; z4 < 4; ++z4) {
                        const float4 v = *(const float4*)(rp + z4 * 4);
                        rv[z4 * 4] = v.x; rv[z4 * 4 + 1] = v.y;
                        rv[z4 * 4 + 2] = v.z; rv[z4 * 4 + 3] = v.w;
                    }
                    snk = sinks[(size_t)gq * 24 + h];
                } else {
#pragma unroll
                    for (int z = 0; z < 16; ++z) rv[z] = 0.f;
                }
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const int kc = kb + jj;
                    const int gk = k0 + kc;
                    float b = -1e30f;
                    // Row 1024 is the zero pad, so the dot can be unconditional:
                    // out-of-range distances clamp to it and contribute zero.
                    const bool live = gq < C && kc < kcnt && gk <= qp &&
                        (swa_win <= 0 || qp - gk < swa_win);
                    const int dist = min(max(qp - gk, 0), 1024);
                    const float* rw = rel_w + ((size_t)h * 1025 + dist) * 16;
                    float rd = 0.f;
#pragma unroll
                    for (int z4 = 0; z4 < 4; ++z4) {
                        const float4 v = *(const float4*)(rw + z4 * 4);
                        rd = fmaf(rv[z4 * 4], v.x, rd);
                        rd = fmaf(rv[z4 * 4 + 1], v.y, rd);
                        rd = fmaf(rv[z4 * 4 + 2], v.z, rd);
                        rd = fmaf(rv[z4 * 4 + 3], v.w, rd);
                    }
                    if (live) b = rd - snk;
                    S[qr][kc] = b;
                }
            }
            __syncthreads();

            // Merge QK and bias using the same 2x4 ownership.
            {
                const int qtx = tid & 15;
                const int qty = tid >> 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int qr = qty * 4 + i;
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int kc = qtx * 2 + j;
                        const float b = S[qr][kc];
                        S[qr][kc] = (b > -1e29f) ? fmaf(aqk[i][j], scale, b) : b;
                    }
                }
            }
            __syncthreads();

            // K is dead after QK; overwrite the same smem tile with V.
            for (int idx = tid; idx < kcnt * 32; idx += 128) {
                const int kr = idx >> 5, d = (idx & 31) * 4;
                const int logical = k0 + kr;
                const int slot = (swa_win > 0) ? (logical % swa_win) : logical;
                const bool vsnap = VS && logical >= snap_start && logical < snap_start + snap_len;
                if (USE_Q && !vsnap) {
                    KVh[d][kr] = __float2half_rn(pf_kvq_read_v(qslot, logical, slot, hk, d, posref));
                    KVh[d + 1][kr] = __float2half_rn(pf_kvq_read_v(qslot, logical, slot, hk, d + 1, posref));
                    KVh[d + 2][kr] = __float2half_rn(pf_kvq_read_v(qslot, logical, slot, hk, d + 2, posref));
                    KVh[d + 3][kr] = __float2half_rn(pf_kvq_read_v(qslot, logical, slot, hk, d + 3, posref));
                } else {
                    const __half* vbase = vsnap ? VS + (size_t)hk * snap_len * 128 : Vh;
                    const size_t vrow = vsnap ? (size_t)(logical - snap_start) : (size_t)slot;
                    const uint2 packed = *(const uint2*)(vbase + vrow * 128 + d);
                    KVh[d][kr] = __ushort_as_half((unsigned short)(packed.x & 0xffffu));
                    KVh[d + 1][kr] = __ushort_as_half((unsigned short)(packed.x >> 16));
                    KVh[d + 2][kr] = __ushort_as_half((unsigned short)(packed.y & 0xffffu));
                    KVh[d + 3][kr] = __ushort_as_half((unsigned short)(packed.y >> 16));
                }
            }
            __syncthreads();

            // One thread per query row updates the flash-softmax state and
            // rewrites S in place as probabilities.  Row order is fixed.
            if (tid < BQ) {
                const int qr = tid;
                float mx = -1e30f;
                for (int kk = 0; kk < kcnt; ++kk) mx = fmaxf(mx, S[qr][kk]);
                const float mn = fmaxf(Mstate[qr], mx);
                const float alpha = __expf(Mstate[qr] - mn);
                float sum = 0.f;
                for (int kk = 0; kk < kcnt; ++kk) {
                    const float sc = S[qr][kk];
                    const float p = (sc > -1e29f) ? __expf(sc - mn) : 0.f;
                    S[qr][kk] = p;
                    sum += p;
                }
                Lstate[qr] = Lstate[qr] * alpha + sum;
                Mstate[qr] = mn;
                Alpha[qr] = alpha;
            }
            __syncthreads();

            // PV tile: this thread owns query rows ty*4..+3 and output columns
            // tx+16*j.  The strided column map makes every smem vector conflict-free.
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float al = Alpha[ty * 4 + i];
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] *= al;
            }
            if (kcnt == BK) {
#pragma unroll
                for (int kr = 0; kr < BK; kr += 4) {
                    float pv[4][4];
#pragma unroll
                    for (int z = 0; z < 4; ++z)
#pragma unroll
                        for (int i = 0; i < 4; ++i) pv[z][i] = S[ty * 4 + i][kr + z];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int dcol = tx + 16 * j;
                        const uint2 packed = *(const uint2*)&KVh[dcol][kr];
                        const __half2 v01 = *(const __half2*)&packed.x;
                        const __half2 v23 = *(const __half2*)&packed.y;
                        const float vv[4] = {__low2float(v01), __high2float(v01),
                                             __low2float(v23), __high2float(v23)};
#pragma unroll
                        for (int z = 0; z < 4; ++z)
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                acc[i][j] = fmaf(pv[z][i], vv[z], acc[i][j]);
                    }
                }
            } else {
                for (int kr = 0; kr < kcnt; ++kr) {
                    float pv[4], vv[8];
#pragma unroll
                    for (int i = 0; i < 4; ++i) pv[i] = S[ty * 4 + i][kr];
#pragma unroll
                    for (int j = 0; j < 8; ++j) vv[j] = __half2float(KVh[tx + 16 * j][kr]);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            acc[i][j] = fmaf(pv[i], vv[j], acc[i][j]);
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qr = ty * 4 + i;
            const int gq = q0 + qr;
            if (gq < C) {
                const float inv = 1.f / Lstate[qr];
                float* dst = out + ((size_t)gq * 24 + h) * 128;
#pragma unroll
                for (int j = 0; j < 8; ++j) dst[tx + 16 * j] = acc[i][j] * inv;
            }
        }
        __syncthreads();
    }
}

extern "C" void ink_launch_flash_prefill_gqa3(const void* K, const void* V, const float* Q,
        const float* r, const float* rel_w, const float* sinks, float* out,
        int C, int pos0, int kv_cap, int swa_win, float scale, void* stream) {
    dim3 g((C + 31) / 32, 8);
    inkwell_flash_prefill_gqa3<0><<<g, 128, 0, (cudaStream_t)stream>>>((const __half*)K,
        (const __half*)V, Q, r, rel_w, sinks, out, C, pos0, kv_cap, swa_win, scale,
        nullptr, nullptr, 0, 0, PfKvQSlot{});
}

extern "C" void ink_launch_flash_prefill_gqa3_snap(const void* K, const void* V, const float* Q,
        const float* r, const float* rel_w, const float* sinks, float* out,
        int C, int pos0, int kv_cap, int swa_win, float scale,
        const void* KS, const void* VS, int snap_start, int snap_len, void* stream) {
    dim3 g((C + 31) / 32, 8);
    inkwell_flash_prefill_gqa3<0><<<g, 128, 0, (cudaStream_t)stream>>>((const __half*)K,
        (const __half*)V, Q, r, rel_w, sinks, out, C, pos0, kv_cap, swa_win, scale,
        (const __half*)KS, (const __half*)VS, snap_start, snap_len, PfKvQSlot{});
}

extern "C" void ink_launch_flash_prefill_gqa3_snap_kvq(const PfKvQSlot* qslot, const float* Q,
        const float* r, const float* rel_w, const float* sinks, float* out,
        int C, int pos0, int kv_cap, int swa_win, float scale,
        const void* KS, const void* VS, int snap_start, int snap_len, void* stream) {
    dim3 g((C + 31) / 32, 8);
    inkwell_flash_prefill_gqa3<1><<<g, 128, 0, (cudaStream_t)stream>>>(nullptr,
        nullptr, Q, r, rel_w, sinks, out, C, pos0, kv_cap, swa_win, scale,
        (const __half*)KS, (const __half*)VS, snap_start, snap_len, *qslot);
}

// Exact-reference attention for one query token.  This follows the non-split
// GQA-3 decode kernel in kernels.cu: the same lane-strided QK reduction, a
// materialized relative-bias table, full-score max/denominator, and serial-key
// V accumulation.  work must have 3*work_stride floats per KV head.
__device__ __forceinline__ void pf_exact_keymap(int j, int qp, int swa,
        int& logical, int& slot, int& dist) {
    if (swa > 0) {
        dist = (qp - j) % swa;
        logical = qp - dist;
        slot = j;
    } else {
        dist = qp - j;
        logical = j;
        slot = j;
    }
}

__device__ __forceinline__ const __half* pf_exact_kv_row(
        const __half* __restrict__ base, const __half* __restrict__ snap,
        int hk, int kv_cap, int logical, int slot, int snap_start, int snap_len) {
    const bool osnap = snap && logical >= snap_start && logical < snap_start + snap_len;
    if (osnap) return snap + ((size_t)hk * snap_len + (logical - snap_start)) * 128;
    return base + ((size_t)hk * kv_cap + slot) * 128;
}

extern "C" __global__ void __launch_bounds__(256) inkwell_pf_exact_attn_gqa3(
        const __half* __restrict__ K, const __half* __restrict__ V,
        const float* __restrict__ Q, const float* __restrict__ r,
        const float* __restrict__ rel_w, const float* __restrict__ sinks,
        float* __restrict__ out, float* __restrict__ work, int work_stride,
        int qrow, int pos0, int kv_cap, int swa_win, float scale,
        const __half* __restrict__ KS, const __half* __restrict__ VS,
        int snap_start, int snap_len) {
    constexpr int VTILE = 128;
    __shared__ float qs[384];
    __shared__ float btab[3][1025];
    __shared__ float red[3][32];
    __shared__ uint2 kvbuf[VTILE][32];

    const int hk = blockIdx.x;
    const int h0 = hk * 3;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;
    const int qp = pos0 + qrow;
    const int nkv = (swa_win > 0) ? min(qp + 1, swa_win) : qp + 1;
    float* score0 = work + (size_t)hk * 3 * work_stride;
    float* score1 = score0 + work_stride;
    float* score2 = score1 + work_stride;

    for (int i = tid; i < 384; i += 256) {
        const int g = i >> 7;
        const int d = i & 127;
        qs[i] = Q[((size_t)qrow * 24 + h0 + g) * 128 + d];
    }

    // Materialize table[head][dist] once for this token before gathering it per
    // key.  K=16 is small enough that each output uses one ascending FMA chain.
    for (int i = tid; i < 3 * 1025; i += 256) {
        const int g = i / 1025;
        const int dist = i - g * 1025;
        const int h = h0 + g;
        const float* rp = r + ((size_t)qrow * 24 + h) * 16;
        const float* rw = rel_w + ((size_t)h * 1025 + dist) * 16;
        float a = 0.f;
#pragma unroll
        for (int z = 0; z < 16; ++z) a += rp[z] * rw[z];
        btab[g][dist] = a;
    }
    __syncthreads();

    for (int base = 0; base < nkv; base += VTILE) {
        const int cnt = min(VTILE, nkv - base);
        for (int rr = w; rr < cnt; rr += 8) {
            int logical, slot, dist;
            pf_exact_keymap(base + rr, qp, swa_win, logical, slot, dist);
            const __half* kj = pf_exact_kv_row(K, KS, hk, kv_cap, logical, slot,
                snap_start, snap_len);
            kvbuf[rr][lane] = __ldcs((const uint2*)(kj + lane * 4));
        }
        __syncthreads();
        for (int rr = w; rr < cnt; rr += 8) {
            const int j = base + rr;
            const __half* kh = (const __half*)kvbuf[rr];
            float a0 = 0.f, a1 = 0.f, a2 = 0.f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int d = lane + 32 * t;
                const float kv = __half2float(kh[d]);
                a0 += qs[d] * kv;
                a1 += qs[128 + d] * kv;
                a2 += qs[256 + d] * kv;
            }
            for (int o = 16; o > 0; o >>= 1) {
                a0 += __shfl_down_sync(0xffffffffu, a0, o);
                a1 += __shfl_down_sync(0xffffffffu, a1, o);
                a2 += __shfl_down_sync(0xffffffffu, a2, o);
            }
            if (lane == 0) {
                score0[j] = a0;
                score1[j] = a1;
                score2[j] = a2;
            }
        }
        __syncthreads();
    }

    const float snk0 = sinks[(size_t)qrow * 24 + h0];
    const float snk1 = sinks[(size_t)qrow * 24 + h0 + 1];
    const float snk2 = sinks[(size_t)qrow * 24 + h0 + 2];
    float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f;
    for (int j = tid; j < nkv; j += 256) {
        int logical, slot, dist;
        pf_exact_keymap(j, qp, swa_win, logical, slot, dist);
        dist = min(dist, 1024);
        const float b0 = __fsub_rn(btab[0][dist], snk0);
        const float b1 = __fsub_rn(btab[1][dist], snk1);
        const float b2 = __fsub_rn(btab[2][dist], snk2);
        const float s0 = score0[j] * scale + b0;
        const float s1 = score1[j] * scale + b1;
        const float s2 = score2[j] * scale + b2;
        score0[j] = s0;
        score1[j] = s1;
        score2[j] = s2;
        m0 = fmaxf(m0, s0);
        m1 = fmaxf(m1, s1);
        m2 = fmaxf(m2, s2);
    }
    const float msink = 0.f;
    m0 = fmaxf(m0, msink);
    m1 = fmaxf(m1, msink);
    m2 = fmaxf(m2, msink);
    for (int o = 16; o > 0; o >>= 1) {
        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
        m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
        m2 = fmaxf(m2, __shfl_xor_sync(0xffffffffu, m2, o));
    }
    if (lane == 0) {
        red[0][w] = m0;
        red[1][w] = m1;
        red[2][w] = m2;
    }
    __syncthreads();
    if (w == 0) {
        float mm0 = -1e30f, mm1 = -1e30f, mm2 = -1e30f;
        for (int i = lane; i < 8; i += 32) {
            mm0 = fmaxf(mm0, red[0][i]);
            mm1 = fmaxf(mm1, red[1][i]);
            mm2 = fmaxf(mm2, red[2][i]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            mm0 = fmaxf(mm0, __shfl_xor_sync(0xffffffffu, mm0, o));
            mm1 = fmaxf(mm1, __shfl_xor_sync(0xffffffffu, mm1, o));
            mm2 = fmaxf(mm2, __shfl_xor_sync(0xffffffffu, mm2, o));
        }
        if (lane == 0) {
            red[0][0] = mm0;
            red[1][0] = mm1;
            red[2][0] = mm2;
        }
    }
    __syncthreads();
    const float mx0 = red[0][0], mx1 = red[1][0], mx2 = red[2][0];

    float den0 = 0.f, den1 = 0.f, den2 = 0.f;
    for (int j = tid; j < nkv; j += 256) {
        const float e0 = __expf(score0[j] - mx0);
        const float e1 = __expf(score1[j] - mx1);
        const float e2 = __expf(score2[j] - mx2);
        score0[j] = e0;
        score1[j] = e1;
        score2[j] = e2;
        den0 += e0;
        den1 += e1;
        den2 += e2;
    }
    if (tid == 0) {
        den0 += __expf(msink - mx0);
        den1 += __expf(msink - mx1);
        den2 += __expf(msink - mx2);
    }
    for (int o = 16; o > 0; o >>= 1) {
        den0 += __shfl_xor_sync(0xffffffffu, den0, o);
        den1 += __shfl_xor_sync(0xffffffffu, den1, o);
        den2 += __shfl_xor_sync(0xffffffffu, den2, o);
    }
    if (lane == 0) {
        red[0][w] = den0;
        red[1][w] = den1;
        red[2][w] = den2;
    }
    __syncthreads();
    if (w == 0) {
        float dd0 = 0.f, dd1 = 0.f, dd2 = 0.f;
        for (int i = lane; i < 8; i += 32) {
            dd0 += red[0][i];
            dd1 += red[1][i];
            dd2 += red[2][i];
        }
        for (int o = 16; o > 0; o >>= 1) {
            dd0 += __shfl_xor_sync(0xffffffffu, dd0, o);
            dd1 += __shfl_xor_sync(0xffffffffu, dd1, o);
            dd2 += __shfl_xor_sync(0xffffffffu, dd2, o);
        }
        if (lane == 0) {
            red[0][0] = dd0;
            red[1][0] = dd1;
            red[2][0] = dd2;
        }
    }
    __syncthreads();
    const float inv0 = 1.f / red[0][0];
    const float inv1 = 1.f / red[1][0];
    const float inv2 = 1.f / red[2][0];

    const int d = tid;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f;
    for (int base = 0; base < nkv; base += VTILE) {
        const int cnt = min(VTILE, nkv - base);
        for (int rr = w; rr < cnt; rr += 8) {
            int logical, slot, dist;
            pf_exact_keymap(base + rr, qp, swa_win, logical, slot, dist);
            const __half* vj = pf_exact_kv_row(V, VS, hk, kv_cap, logical, slot,
                snap_start, snap_len);
            kvbuf[rr][lane] = __ldcs((const uint2*)(vj + lane * 4));
        }
        __syncthreads();
        if (tid < 128) {
            for (int rr = 0; rr < cnt; ++rr) {
                const int j = base + rr;
                const float vv = __half2float(((const __half*)kvbuf[rr])[d]);
                acc0 += score0[j] * vv;
                acc1 += score1[j] * vv;
                acc2 += score2[j] * vv;
            }
        }
        __syncthreads();
    }
    if (tid < 128) {
        float* dst = out + ((size_t)qrow * 24 + h0) * 128 + d;
        dst[0] = acc0 * inv0;
        dst[128] = acc1 * inv1;
        dst[256] = acc2 * inv2;
    }
}

extern "C" void ink_launch_pf_exact_attn_gqa3_snap(const void* K, const void* V, const float* Q,
        const float* r, const float* rel_w, const float* sinks, float* out,
        float* work, int work_stride, int C, int pos0, int kv_cap, int swa_win, float scale,
        const void* KS, const void* VS, int snap_start, int snap_len, void* stream) {
    for (int qrow = 0; qrow < C; ++qrow) {
        inkwell_pf_exact_attn_gqa3<<<dim3(8), 256, 0, (cudaStream_t)stream>>>((const __half*)K,
            (const __half*)V, Q, r, rel_w, sinks, out, work, work_stride, qrow, pos0,
            kv_cap, swa_win, scale, (const __half*)KS, (const __half*)VS, snap_start, snap_len);
    }
}

extern "C" __global__ void inkwell_pf_snapshot_kv(const void* __restrict__ kd,
        const void* __restrict__ vd, void* __restrict__ ks, void* __restrict__ vs,
        int snap_start, int snap_len, int cap, int swa) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= snap_len || i >= 1024) return;
    const int h = i >> 7, d = i & 127;
    const int logical = snap_start + row;
    const int slot = logical % swa;
    const size_t src = ((size_t)h * cap + slot) * 128 + d;
    const size_t dst = ((size_t)h * snap_len + row) * 128 + d;
    ((__half*)ks)[dst] = ((const __half*)kd)[src];
    ((__half*)vs)[dst] = ((const __half*)vd)[src];
}

extern "C" void ink_launch_pf_snapshot_kv(const void* kd, const void* vd, void* ks, void* vs,
        int snap_start, int snap_len, int cap, int swa, void* stream) {
    dim3 g(4, snap_len);
    inkwell_pf_snapshot_kv<<<g, 256, 0, (cudaStream_t)stream>>>(kd, vd, ks, vs,
        snap_start, snap_len, cap, swa);
}


// ---------------- Phase-2 driver glue kernels ----------------
// All row-major activations are [C,n] with token rows contiguous.

extern "C" __global__ void inkwell_pf_embed_bf16(const uint16_t* __restrict__ embd,
        const int* __restrict__ ids, float* __restrict__ out, int C, int n, long long stride_bytes) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C || i >= n) return;
    const uint16_t* row = (const uint16_t*)((const char*)embd + (size_t)ids[t] * (size_t)stride_bytes);
    out[(size_t)t * n + i] = __uint_as_float((uint32_t)row[i] << 16);
}

extern "C" void ink_launch_pf_embed_bf16(const void* embd, const int* ids, float* out,
        int C, int n, long long stride_bytes, void* stream) {
    dim3 g((n + 255) / 256, C);
    inkwell_pf_embed_bf16<<<g, 256, 0, (cudaStream_t)stream>>>((const uint16_t*)embd, ids, out, C, n, stride_bytes);
}

extern "C" __global__ void inkwell_pf_rmsnorm(float* __restrict__ x,
        const float* __restrict__ w, int rows, int n, float eps) {
    __shared__ float sm[256];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    float ss = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = x[(size_t)row * n + i];
        ss = fmaf(v, v, ss);
    }
    sm[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const float inv = 1.f / sqrtf(sm[0] / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        float v = x[(size_t)row * n + i] * inv;
        if (w) v *= w[i];
        x[(size_t)row * n + i] = v;
    }
}

// ggml rms_norm_f32 vec order: one float4 per thread, xor-shuffle warp sums,
// then the same 32-entry cross-warp tree in every warp.
template <int BS>
__global__ void inkwell_pf_rmsnorm_exact(float* __restrict__ x,
        const float* __restrict__ w, int n, float eps) {
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    float* xr = x + (size_t)blockIdx.x * n;
    const int n4 = n >> 2;
    float4 xs = make_float4(0.f, 0.f, 0.f, 0.f);
    float tmp = 0.f;
    if (tid < n4) {
        xs = ((const float4*)xr)[tid];
        tmp += xs.x * xs.x + xs.y * xs.y + xs.z * xs.z + xs.w * xs.w;
    }
#pragma unroll
    for (int m = 16; m; m >>= 1) tmp += __shfl_xor_sync(0xffffffffu, tmp, m);
    if (lane == 0) s_sum[wid] = tmp;
    __syncthreads();
    tmp = 0.f;
    if (lane < BS / 32) tmp = s_sum[lane];
#pragma unroll
    for (int m = 16; m; m >>= 1) tmp += __shfl_xor_sync(0xffffffffu, tmp, m);
    const float scale = rsqrtf(__fdiv_rn(tmp, (float)n) + eps);
    if (tid < n4) {
        float4 r = make_float4(scale * xs.x, scale * xs.y, scale * xs.z, scale * xs.w);
        if (w) {
            const int i = tid * 4;
            r.x *= w[i];
            r.y *= w[i + 1];
            r.z *= w[i + 2];
            r.w *= w[i + 3];
        }
        ((float4*)xr)[tid] = r;
    }
}

extern "C" void ink_launch_pf_rmsnorm(float* x, const float* w, int rows, int n,
        float eps, void* stream) {
    const int n4 = n >> 2;
    if ((n & 3) == 0 && n4 <= 1024) {
        const cudaStream_t st = (cudaStream_t)stream;
        if (n4 <= 32) inkwell_pf_rmsnorm_exact<32><<<rows, 32, 0, st>>>(x, w, n, eps);
        else if (n4 <= 128) inkwell_pf_rmsnorm_exact<128><<<rows, 128, 0, st>>>(x, w, n, eps);
        else if (n4 <= 256) inkwell_pf_rmsnorm_exact<256><<<rows, 256, 0, st>>>(x, w, n, eps);
        else if (n4 <= 512) inkwell_pf_rmsnorm_exact<512><<<rows, 512, 0, st>>>(x, w, n, eps);
        else inkwell_pf_rmsnorm_exact<1024><<<rows, 1024, 0, st>>>(x, w, n, eps);
        return;
    }
    inkwell_pf_rmsnorm<<<rows, 256, 0, (cudaStream_t)stream>>>(x, w, rows, n, eps);
}

extern "C" __global__ void inkwell_pf_clip_heads(float* __restrict__ x,
        const float* __restrict__ scale, int C, int heads, int d, float thr) {
    __shared__ float sm[16];
    const int row = blockIdx.x; // C*heads rows
    const int tid = threadIdx.x;
    float* xr = x + (size_t)row * d;
    const float* sr = scale ? scale + (size_t)(row % heads) * d : nullptr;
    float cv = 0.f;
    if (tid < d) {
        cv = xr[tid];
        if (sr) cv *= sr[tid];
    }
    float v = cv * cv;
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((tid & 31) == 0) sm[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float nsum = sm[0];
        for (int j = 1; j < 8; ++j) nsum += sm[j];
        sm[8] = fminf(1.f, thr / (sqrtf(nsum) + 1e-6f));
    }
    __syncthreads();
    const float f = sm[8];
    if (tid < d) xr[tid] = cv * f;
    (void)C;
}

extern "C" void ink_launch_pf_clip_heads(float* x, const float* scale, int C, int heads,
        int d, float thr, void* stream) {
    inkwell_pf_clip_heads<<<C * heads, 256, 0, (cudaStream_t)stream>>>(x, scale, C, heads, d, thr);
}

extern "C" __global__ void inkwell_pf_add(float* __restrict__ x, const float* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

extern "C" void ink_launch_pf_add(float* x, const float* y, long long n, void* stream) {
    inkwell_pf_add<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, y, n);
}

extern "C" __global__ void inkwell_pf_mul_sigmoid(float* __restrict__ x, const float* __restrict__ gate, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.f / (1.f + expf(-gate[i]));
}

extern "C" void ink_launch_pf_mul_sigmoid(float* x, const float* gate, long long n, void* stream) {
    inkwell_pf_mul_sigmoid<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, gate, n);
}

extern "C" __global__ void inkwell_pf_softcap(float* __restrict__ x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

extern "C" void ink_launch_pf_softcap(float* x, int n, float cap, void* stream) {
    inkwell_pf_softcap<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(x, n, cap);
}

extern "C" __global__ void inkwell_pf_tau_qr(float* __restrict__ q, float* __restrict__ r,
        int C, float log_alpha, float log_floor, int pos0) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float tau = logf(fmaxf((float)(pos0 + t + 1) / log_floor, 1.f)) * log_alpha + 1.f;
    if (i < 3072) q[(size_t)t * 3072 + i] *= tau;
    if (i < 384) r[(size_t)t * 384 + i] *= tau;
}

extern "C" void ink_launch_pf_tau_qr(float* q, float* r, int C, float log_alpha,
        float log_floor, int pos0, void* stream) {
    dim3 g(12, C);
    inkwell_pf_tau_qr<<<g, 256, 0, (cudaStream_t)stream>>>(q, r, C, log_alpha, log_floor, pos0);
}

// Small/medium row-major GEMM for dense BF16 control, router, sink, and conv-projection weights.
// x [M,K] f32, w [N,K] bf16 or f32 row-major, y [M,N] f32.
extern "C" __global__ void inkwell_pf_dense_gemm_tiled(const float* __restrict__ x,
        const void* __restrict__ w, float* __restrict__ y, int M, int N, int K, int w_is_bf16) {
    constexpr int TM = 16, TN = 16, TK = 16;
    __shared__ float xs[TM][TK + 1];
    __shared__ float ws[TN][TK + 1];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int row = blockIdx.y * TM + ty;
    const int col = blockIdx.x * TN + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int kk = k0 + tx;
        xs[ty][tx] = (row < M && kk < K) ? x[(size_t)row * K + kk] : 0.f;
        const int wr = blockIdx.x * TN + ty;
        if (wr < N && kk < K) {
            if (w_is_bf16) {
                const uint16_t v = ((const uint16_t*)w)[(size_t)wr * K + kk];
                ws[ty][tx] = __uint_as_float((uint32_t)v << 16);
            } else {
                ws[ty][tx] = ((const float*)w)[(size_t)wr * K + kk];
            }
        } else ws[ty][tx] = 0.f;
        __syncthreads();
#pragma unroll
        for (int z = 0; z < TK; ++z) acc = fmaf(xs[ty][z], ws[tx][z], acc);
        __syncthreads();
    }
    if (row < M && col < N) y[(size_t)row * N + col] = acc;
}

extern "C" void ink_launch_pf_dense_gemm_tiled(const float* x, const void* w, float* y,
        int M, int N, int K, int w_is_bf16, void* stream) {
    dim3 g((N + 15) / 16, (M + 15) / 16);
    inkwell_pf_dense_gemm_tiled<<<g, 256, 0, (cudaStream_t)stream>>>(x, w, y, M, N, K, w_is_bf16);
}

__device__ __forceinline__ float2 pf_dense_wload2(const float* w, size_t i) {
    return ((const float2*)w)[i];
}

__device__ __forceinline__ float2 pf_dense_wload2(const __nv_bfloat16* w, size_t i) {
    const __nv_bfloat162 v = ((const __nv_bfloat162*)w)[i];
    return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
}

// Batched dense projection in the serial mmvf reduction order. blockIdx.y is
// the token row, so every token sees the same pair order as a single-token GEMV.
template <typename T, int R, int BLOCK>
__global__ void inkwell_pf_dense_mmvf_exact(const float* __restrict__ x,
        const T* __restrict__ w, float* __restrict__ y, int M, int N, int K) {
    constexpr int NW = BLOCK / 32;
    constexpr int TM = 4;
    static_assert(R <= NW, "rows use one reduction warp each");
    __shared__ float buf[R][32];
    const int tok0 = blockIdx.y * TM;
    const int r0 = blockIdx.x * R;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    const int k2 = K >> 1;
    const int nt = (M - tok0) < TM ? (M - tok0) : TM;
    if (nt <= 0) return;

    const T* wr0 = w + (size_t)r0 * K;
    float sum[TM][R];
#pragma unroll
    for (int dt = 0; dt < TM; ++dt) {
#pragma unroll
        for (int r = 0; r < R; ++r) sum[dt][r] = 0.0f;
    }

    for (int col2 = tid; col2 < k2; col2 += BLOCK) {
        float2 xv[TM];
#pragma unroll
        for (int dt = 0; dt < TM; ++dt) {
            xv[dt] = (dt < nt)
                ? __ldg(&((const float2*)(x + (size_t)(tok0 + dt) * K))[col2])
                : make_float2(0.0f, 0.0f);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r < N) {
                const float2 wv = pf_dense_wload2(wr0 + (size_t)r * K, (size_t)col2);
#pragma unroll
                for (int dt = 0; dt < TM; ++dt) {
                    sum[dt][r] += wv.x * xv[dt].x;
                    sum[dt][r] += wv.y * xv[dt].y;
                }
            }
        }
    }

    for (int dt = 0; dt < nt; ++dt) {
        const int tok = tok0 + dt;
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) sum[dt][r] += __shfl_xor_sync(0xffffffffu, sum[dt][r], m);
        }
        if constexpr (BLOCK == 32) {
            if (tid == 0) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r0 + r < N) y[(size_t)tok * N + r0 + r] = sum[dt][r];
                }
            }
        } else {
            if (tid < 32) {
#pragma unroll
                for (int r = 0; r < R; ++r) buf[r][tid] = 0.0f;
            }
            __syncthreads();
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (lane == 0) buf[r][wid] = sum[dt][r];
            }
            __syncthreads();
            if (wid < R) {
                const int r = wid;
                float v = buf[r][lane];
#pragma unroll
                for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
                if (lane == 0 && r0 + r < N) y[(size_t)tok * N + r0 + r] = v;
            }
            __syncthreads();
        }
    }
}

static int pf_mmvf_block_size(int k) {
    int best = 32;
    int best_iter = (k + 2 * 32 - 1) / (2 * 32);
    for (int bs = 64; bs <= 256; bs += 32) {
        const int niter = (k + 2 * bs - 1) / (2 * bs);
        if (niter < best_iter) {
            best_iter = niter;
            best = bs;
        }
    }
    return best;
}

template <typename T, int BLOCK>
static void pf_launch_dense_exact_block(const float* x, const T* w, float* y,
        int M, int N, int K, cudaStream_t stream) {
    constexpr int R = BLOCK / 32 < 4 ? BLOCK / 32 : 4;
    dim3 g((unsigned)((N + R - 1) / R), (unsigned)((M + 3) / 4));
    inkwell_pf_dense_mmvf_exact<T, R, BLOCK><<<g, BLOCK, 0, stream>>>(x, w, y, M, N, K);
}

template <typename T>
static void pf_launch_dense_exact_type(const float* x, const T* w, float* y,
        int M, int N, int K, cudaStream_t stream) {
    switch (pf_mmvf_block_size(K)) {
        case 32:  pf_launch_dense_exact_block<T, 32>(x, w, y, M, N, K, stream); break;
        case 64:  pf_launch_dense_exact_block<T, 64>(x, w, y, M, N, K, stream); break;
        case 96:  pf_launch_dense_exact_block<T, 96>(x, w, y, M, N, K, stream); break;
        case 128: pf_launch_dense_exact_block<T, 128>(x, w, y, M, N, K, stream); break;
        case 160: pf_launch_dense_exact_block<T, 160>(x, w, y, M, N, K, stream); break;
        case 192: pf_launch_dense_exact_block<T, 192>(x, w, y, M, N, K, stream); break;
        case 224: pf_launch_dense_exact_block<T, 224>(x, w, y, M, N, K, stream); break;
        default:  pf_launch_dense_exact_block<T, 256>(x, w, y, M, N, K, stream); break;
    }
}

// Exact path requires even K and tightly packed row-major BF16/F32 weights.
extern "C" void ink_launch_pf_dense_gemm_exact(const float* x, const void* w, float* y,
        int M, int N, int K, int w_is_bf16, void* stream) {
    if (M <= 0 || N <= 0) return;
    if (w_is_bf16) {
        pf_launch_dense_exact_type(x, (const __nv_bfloat16*)w, y, M, N, K, (cudaStream_t)stream);
    } else {
        pf_launch_dense_exact_type(x, (const float*)w, y, M, N, K, (cudaStream_t)stream);
    }
}

extern "C" void ink_launch_pf_dense_gemm(const float* x, const void* w, float* y,
        int M, int N, int K, int w_is_bf16, void* stream) {
    if ((K & 1) == 0) {
        ink_launch_pf_dense_gemm_exact(x, w, y, M, N, K, w_is_bf16, stream);
    } else {
        ink_launch_pf_dense_gemm_tiled(x, w, y, M, N, K, w_is_bf16, stream);
    }
}

// Compute the two controller RMS scales once per token group.  This is the
// same reduction sequence formerly repeated by every row tile of rms2down.
__global__ void inkwell_pf_ctrl_scales(const float* __restrict__ cur,
        const float* __restrict__ msg, float* __restrict__ scalesg,
        int C, int n, float eps) {
    constexpr int TM = 4;
    __shared__ float s_sum[32];
    const int tok0 = blockIdx.x * TM;
    const int nt = (C - tok0) < TM ? (C - tok0) : TM;
    if (nt <= 0) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n4 = n >> 2;
    if (tid >= 24 && tid < 32) s_sum[tid] = 0.0f;

    for (int dt = 0; dt < nt; ++dt) {
        const int tok = tok0 + dt;
        const float* __restrict__ x0 = cur + (size_t)tok * n;
        const float* __restrict__ x1 = msg + (size_t)tok * n;
#pragma unroll
        for (int bx = 0; bx < 2; ++bx) {
            const float* x = bx ? x1 : x0;
            float p[3];
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int vt = tid + k * 256;
                if (vt < n4) {
                    const float4 q = __ldg(&((const float4*)x)[vt]);
                    p[k] = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
                } else {
                    p[k] = 0.0f;
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                float v = p[k];
#pragma unroll
                for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
                if (lane == 0) s_sum[wid + 8 * k] = v;
            }
            __syncthreads();
            float t = s_sum[lane];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) t += __shfl_xor_sync(0xffffffffu, t, m);
            if (tid == 0) scalesg[tok * 2 + bx] = rsqrtf(__fdiv_rn(t, (float)n) + eps);
        }
    }
}

// Batched loop-controller chain in the reduction order of the serial fused
// rms2down + gf_apply kernels, tiled over tokens to amortize BF16 weight loads.
__global__ void inkwell_pf_ctrl_rms2down(const float* __restrict__ cur,
        const float* __restrict__ msg, const float* __restrict__ scalesg,
        const __nv_bfloat16* __restrict__ w,
        float* __restrict__ gate, int C, int n, int rows, float eps) {
    constexpr int TK = 64;
    constexpr int TOKENS = 16;
    constexpr int TPC = 2;  // tokens per warp; each lane covers two output rows
    __shared__ float xs[TOKENS][TK + 1];
    __shared__ float ws[64][TK + 1];
    __shared__ float sc[TOKENS][2];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int tok0 = blockIdx.y * TOKENS;
    (void)eps;

    if (tid < TOKENS * 2) {
        const int dt = tid >> 1;
        const int bx = tid & 1;
        const int tok = tok0 + dt;
        sc[dt][bx] = (tok < C) ? __ldg(scalesg + tok * 2 + bx) : 0.0f;
    }
    __syncthreads();

    const int out0 = lane;
    const int out1 = lane + 32;
    const int t0 = wid * TPC;
    float acc[TPC][2];
#pragma unroll
    for (int z = 0; z < TPC; ++z) {
        acc[z][0] = 0.0f;
        acc[z][1] = 0.0f;
    }

    const int ktot = 2 * n;
    for (int k0 = 0; k0 < ktot; k0 += TK) {
        for (int idx = tid; idx < 32 * TK; idx += 256) {
            const int r = idx >> 5;
            const int kk = (idx & 31) * 2;
            const int k = k0 + kk;
            if (r < rows && k + 1 < ktot) {
                const float2 wv = pf_dense_wload2(w, ((size_t)r * ktot + k) / 2);
                ws[r][kk] = wv.x;
                ws[r][kk + 1] = wv.y;
            } else {
                ws[r][kk] = 0.0f;
                ws[r][kk + 1] = 0.0f;
            }
        }
        for (int idx = tid; idx < TOKENS * TK; idx += 256) {
            const int dt = idx >> 6;
            const int kk = idx & 63;
            const int k = k0 + kk;
            const int tok = tok0 + dt;
            float v = 0.0f;
            if (tok < C && k < ktot) {
                if (k < n) {
                    v = __fmul_rn(sc[dt][0], __ldg(cur + (size_t)tok * n + k));
                } else {
                    v = __fmul_rn(sc[dt][1], __ldg(msg + (size_t)tok * n + (k - n)));
                }
            }
            xs[dt][kk] = v;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < TK; ++kk) {
            const float w0 = ws[out0][kk];
            const float w1 = ws[out1][kk];
#pragma unroll
            for (int z = 0; z < TPC; ++z) {
                const float xv = xs[t0 + z][kk];
                acc[z][0] = fmaf(xv, w0, acc[z][0]);
                acc[z][1] = fmaf(xv, w1, acc[z][1]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int z = 0; z < TPC; ++z) {
        const int tok = tok0 + t0 + z;
        if (tok < C) {
            if (out0 < rows) gate[(size_t)tok * rows + out0] = acc[z][0];
            if (out1 < rows) gate[(size_t)tok * rows + out1] = acc[z][1];
        }
    }
}

__global__ void inkwell_pf_ctrl_gf_apply(const __nv_bfloat16* __restrict__ w,
        const float* __restrict__ gate, const float* __restrict__ cur,
        const float* __restrict__ msg, float* __restrict__ dst, int C, int n, int fo) {
    constexpr int K = 64;
    constexpr int TOKENS = 32;
    constexpr int TPC = 4;
    __shared__ float wg[32][K + 1];
    __shared__ float wf[32][K + 1];
    __shared__ float gt[TOKENS][K + 1];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int o0 = blockIdx.x * 32;
    const int tok0 = blockIdx.y * TOKENS;

    for (int idx = tid; idx < 32 * K; idx += 256) {
        const int o = idx >> 6;
        const int k = idx & 63;
        const int go = o0 + o;
        wg[o][k] = (go < n) ? __bfloat162float(w[(size_t)go * K + k]) : 0.0f;
        wf[o][k] = (go < n) ? __bfloat162float(w[(size_t)(fo + go) * K + k]) : 0.0f;
    }
    for (int idx = tid; idx < TOKENS * K; idx += 256) {
        const int dt = idx >> 6;
        const int k = idx & 63;
        const int tok = tok0 + dt;
        gt[dt][k] = (tok < C) ? __ldg(gate + (size_t)tok * K + k) : 0.0f;
    }
    __syncthreads();

    const int out = o0 + lane;
    const int t0 = wid * TPC;
    float accg[TPC], accf[TPC];
#pragma unroll
    for (int z = 0; z < TPC; ++z) {
        accg[z] = 0.0f;
        accf[z] = 0.0f;
    }
#pragma unroll 8
    for (int k = 0; k < K; ++k) {
        const float ag = wg[lane][k];
        const float af = wf[lane][k];
#pragma unroll
        for (int z = 0; z < TPC; ++z) {
            const float x = gt[t0 + z][k];
            accg[z] = fmaf(x, ag, accg[z]);
            accf[z] = fmaf(x, af, accf[z]);
        }
    }
    if (out >= n) return;
#pragma unroll
    for (int z = 0; z < TPC; ++z) {
        const int tok = tok0 + t0 + z;
        if (tok < C) {
            const float gn = tanhf(accg[z]), fg = tanhf(accf[z]);
            const float cv = cur[(size_t)tok * n + out];
            const float mv = msg[(size_t)tok * n + out];
            const float r = __fsub_rn(cv, __fmul_rn(fg, cv));
            dst[(size_t)tok * n + out] = __fadd_rn(r, __fmul_rn(gn, mv));
        }
    }
}

extern "C" void ink_launch_pf_ctrl_exact(const float* cur, const float* msg,
        const void* down, const void* gf, float* gate, float* dst,
        int C, int n, int rows, float eps, int fo, void* stream) {
    const cudaStream_t st = (cudaStream_t)stream;
    float* scalesg = gate + (size_t)rows * C;
    inkwell_pf_ctrl_scales<<<dim3((unsigned)((C + 3) / 4)), 256, 0, st>>>(
        cur, msg, scalesg, C, n, eps);
    inkwell_pf_ctrl_rms2down<<<dim3(1, (unsigned)((C + 15) / 16)), 256, 0, st>>>(
        cur, msg, scalesg, (const __nv_bfloat16*)down, gate, C, n, rows, eps);
    inkwell_pf_ctrl_gf_apply<<<dim3((unsigned)((n + 31) / 32), (unsigned)((C + 31) / 32)), 256, 0, st>>>(
        (const __nv_bfloat16*)gf, gate, cur, msg, dst, C, n, fo);
}

extern "C" __global__ void inkwell_pf_ctrl_apply(const float* __restrict__ gain,
        const float* __restrict__ forget, float* __restrict__ cur,
        const float* __restrict__ msg, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = tanhf(gain[i]);
        const float f = tanhf(forget[i]);
        cur[i] = (1.f - f) * cur[i] + g * msg[i];
    }
}

extern "C" void ink_launch_pf_ctrl_apply(const float* gain, const float* forget,
        float* cur, const float* msg, long long n, void* stream) {
    inkwell_pf_ctrl_apply<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(gain, forget, cur, msg, n);
}

// Concatenated gain/forget GEMM output is token-major [C,2n]: gain occupies
// columns [0,n) and forget columns [n,2n) of each row.
extern "C" __global__ void inkwell_pf_ctrl_apply_cat(const float* __restrict__ gf,
        float* __restrict__ cur, const float* __restrict__ msg, int C, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < C && i < n) {
        const size_t x = (size_t)t * n + i;
        const float g = tanhf(gf[(size_t)t * 2 * n + i]);
        const float f = tanhf(gf[(size_t)t * 2 * n + n + i]);
        cur[x] = (1.f - f) * cur[x] + g * msg[x];
    }
}

extern "C" void ink_launch_pf_ctrl_apply_cat(const float* gf, float* cur,
        const float* msg, int C, int n, void* stream) {
    dim3 g((n + 255) / 256, C);
    inkwell_pf_ctrl_apply_cat<<<g, 256, 0, (cudaStream_t)stream>>>(gf, cur, msg, C, n);
}

extern "C" __global__ void inkwell_pf_router8(const float* __restrict__ logits,
        const float* __restrict__ bias, int C, float wscale, float wfloor,
        int* __restrict__ ids, float* __restrict__ weights) {
    if (threadIdx.x != 0) return;
    const int t = blockIdx.x;
    if (t >= C) return;
    float prob[128];
    bool used[128];
    for (int e = 0; e < 128; ++e) {
        const float z = logits[(size_t)t * 128 + e];
        prob[e] = 1.f / (1.f + __expf(-z));
        used[e] = false;
    }
    float sum = 0.f;
    int idv[8];
    for (int j = 0; j < 8; ++j) {
        int best = -1;
        float bs = -1e30f;
        for (int e = 0; e < 128; ++e) {
            if (!used[e] && prob[e] + bias[e] > bs) { bs = prob[e] + bias[e]; best = e; }
        }
        used[best] = true;
        idv[j] = best;
        sum += prob[best];
    }
    sum = fmaxf(sum, wfloor);
    const float inv = wscale / sum;
    for (int j = 0; j < 8; ++j) {
        ids[t * 8 + j] = idv[j];
        weights[t * 8 + j] = prob[idv[j]] * inv;
    }
}

extern "C" void ink_launch_pf_router8(const float* logits, const float* bias, int C,
        float wscale, float wfloor, int* ids, float* weights, void* stream) {
    inkwell_pf_router8<<<C, 32, 0, (cudaStream_t)stream>>>(logits, bias, C, wscale, wfloor, ids, weights);
}

extern "C" __global__ void inkwell_pf_write_kv(const float* __restrict__ k,
        const float* __restrict__ v, void* __restrict__ kd, void* __restrict__ vd,
        int C, int pos0, int cap, int swa) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // h*128+d
    if (t >= C || i >= 1024) return;
    const int h = i >> 7, d = i & 127;
    const int logical = pos0 + t;
    const int slot = swa > 0 ? logical % swa : logical;
    if (slot >= cap) return;
    const size_t dst = ((size_t)h * cap + slot) * 128 + d;
    const size_t src = (size_t)t * 1024 + i;
    ((__half*)kd)[dst] = __float2half_rn(k[src]);
    ((__half*)vd)[dst] = __float2half_rn(v[src]);
}

extern "C" void ink_launch_pf_write_kv(const float* k, const float* v, void* kd, void* vd,
        int C, int pos0, int cap, int swa, void* stream) {
    dim3 g(4, C);
    inkwell_pf_write_kv<<<g, 256, 0, (cudaStream_t)stream>>>(k, v, kd, vd, C, pos0, cap, swa);
}

extern "C" __global__ void inkwell_pf_expand_rel(const float* __restrict__ src,
        float* __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1025 * 16) return;
    for (int h = 0; h < 24; ++h) dst[(size_t)h * 1025 * 16 + i] = src[i];
}

extern "C" void ink_launch_pf_expand_rel(const float* src, float* dst, void* stream) {
    inkwell_pf_expand_rel<<<(1025 * 16 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(src, dst);
}

// BF16-table variant.  `attn_rel_bias_pad` stays BF16 on the device for the
// fused decode sink/relbias kernel, so the prefill driver must convert it
// instead of reinterpreting those bytes as FP32.
extern "C" __global__ void inkwell_pf_expand_rel_bf16(const uint16_t* __restrict__ src,
        float* __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1025 * 16) return;
    const float v = __bfloat162float(*(const __nv_bfloat16*)(src + i));
    for (int h = 0; h < 24; ++h) dst[(size_t)h * 1025 * 16 + i] = v;
}

extern "C" void ink_launch_pf_expand_rel_bf16(const void* src, float* dst, void* stream) {
    inkwell_pf_expand_rel_bf16<<<(1025 * 16 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const uint16_t*)src, dst);
}

// Depth-stack mix. states[j] are [C,n]; query rows are qcols contiguous n-vectors.
// Output layout is [qcol][C][n]. scores are softmax over rows of RMS-normalized states.
struct PfMixStates { const float* p[8]; };

extern "C" __global__ void inkwell_pf_mix(PfMixStates states,
        int rows, const float* __restrict__ query, float* __restrict__ out,
        int C, int n, int qcols, float scale, float eps) {
    __shared__ float ss[8];
    __shared__ float sc[8];
    __shared__ float mx, den;
    const int tq = blockIdx.x;
    const int t = tq % C;
    const int qc = tq / C;
    if (qc >= qcols) return;
    if (threadIdx.x == 0) {
        const float* q = query + (size_t)qc * n;
        float m = -1e30f;
        for (int j = 0; j < rows; ++j) {
            const float* x = states.p[j] + (size_t)t * n;
            float ns = 0.f, dot = 0.f;
            for (int i = 0; i < n; ++i) {
                const float v = x[i];
                ns = fmaf(v, v, ns);
            }
            const float inv = 1.f / sqrtf(ns / (float)n + eps);
            for (int i = 0; i < n; ++i) dot = fmaf(x[i] * inv, q[i], dot);
            sc[j] = dot * scale;
            ss[j] = inv;
            m = fmaxf(m, sc[j]);
        }
        mx = m;
        float ds = 0.f;
        for (int j = 0; j < rows; ++j) ds += expf(sc[j] - m);
        den = ds;
    }
    __syncthreads();
    const float m = mx, ds = den;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < rows; ++j) {
            const float p = expf(sc[j] - m) / ds;
            acc = fmaf(p, states.p[j][(size_t)t * n + i] * ss[j], acc);
        }
        out[((size_t)qc * C + t) * n + i] = acc;
    }
}

#define PF_MIX_NPL 72

// Batched equivalent of inkwell_mix_fused: same lane-strided associations,
// xor-shuffle trees, sequential softmax, and row accumulation order.  One CTA
// per (qcol, token) avoids redundant rms/dot recomputation across embd tiles.
__global__ void __launch_bounds__(256) inkwell_pf_mix_exact(PfMixStates states,
        int rows, const float* __restrict__ q, float* __restrict__ out,
        int C, int embd, float scale) {
    __shared__ float s_rms[64];
    __shared__ float s_p[64];
    const int qc = blockIdx.x;
    const int token = blockIdx.y;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float qq[PF_MIX_NPL];
#pragma unroll
    for (int t = 0; t < PF_MIX_NPL; ++t) qq[t] = q[(size_t)qc * embd + lane + 32 * t];
    for (int j = w; j < rows; j += nw) {
        const float* col = states.p[j] + (size_t)token * embd;
        float c[PF_MIX_NPL];
#pragma unroll
        for (int t = 0; t < PF_MIX_NPL; ++t) c[t] = col[lane + 32 * t];
        float ss = 0.f;
#pragma unroll
        for (int t = 0; t < PF_MIX_NPL; ++t) ss += c[t] * c[t];
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(~0u, ss, m);
        const float rms = sqrtf(ss / (float)embd + 1e-6f);
        float dp = 0.f;
#pragma unroll
        for (int t = 0; t < PF_MIX_NPL; ++t) dp += c[t] / rms * qq[t];
        for (int m = 16; m; m >>= 1) dp += __shfl_xor_sync(~0u, dp, m);
        if (lane == 0) { s_rms[j] = rms; s_p[j] = dp; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < rows; ++j) { s_p[j] *= scale; mx = fmaxf(mx, s_p[j]); }
        float d = 0.f;
        for (int j = 0; j < rows; ++j) { s_p[j] = __expf(s_p[j] - mx); d += s_p[j]; }
        for (int j = 0; j < rows; ++j) s_p[j] /= d;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < embd; e += blockDim.x) {
        float a = 0.f;
        for (int j0 = 0; j0 < rows; j0 += 8) {
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) v[u] = (j0 + u < rows) ? states.p[j0 + u][(size_t)token * embd + e] : 0.f;
#pragma unroll
            for (int u = 0; u < 8; ++u) if (j0 + u < rows) a += v[u] / s_rms[j0 + u] * s_p[j0 + u];
        }
        out[((size_t)qc * C + token) * embd + e] = a;
    }
}

extern "C" void ink_launch_pf_mix(const void* const* states, int rows, const float* query,
        float* out, int C, int n, int qcols, float scale, float eps, void* stream) {
    PfMixStates tmp{};
    for (int i = 0; i < rows && i < 8; ++i) tmp.p[i] = (const float*)states[i];
    if (n == 32 * PF_MIX_NPL && rows <= 64) {
        dim3 g((unsigned)qcols, (unsigned)C);
        inkwell_pf_mix_exact<<<g, 256, 0, (cudaStream_t)stream>>>(tmp, rows, query, out, C, n, scale);
    } else {
        inkwell_pf_mix<<<C * qcols, 256, 0, (cudaStream_t)stream>>>(tmp, rows, query, out, C, n, qcols, scale, eps);
    }
}

// Phase-2 H2D staging-ring contract (stub in Phase 1):
// The driver owns a 32 MiB device ring split into two 16 MiB buffers.  For each
// 32-expert tile t it must (1) wait for tile t-2 compute before reusing a
// buffer, (2) copy tile t+1 host-pinned trellis/suh/svh bytes on the side copy
// stream, (3) rewrite tile descriptors to the device ring addresses, and
// (4) record events so grouped GEMM on the main stream never reads an
// incomplete tile.  Phase 1 kernels take already valid device descriptors and
// intentionally know nothing about the ring.
