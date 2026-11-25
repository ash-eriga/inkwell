// Chunked-prefill kernels for sm_61.  These are independent of the decode
// kernels in kernels.cu; nothing here runs unless a prefill driver invokes it.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

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

// Exact 128-point Hadamard in the same pairwise order as inkwell_exl3_had_in.
__device__ __forceinline__ void pf_had128_smem(float* sm, int t) {
    for (int s = 1; s < 128; s <<= 1) {
        const float u = (t & s) ? sm[t - s] - sm[t] : sm[t] + sm[t + s];
        __syncthreads();
        sm[t] = u;
        __syncthreads();
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

    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        const int tn_i = tn0 + dtile;
        if (tn_i < tn && du < w) {
            rings[dtile][du] = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
        }
        for (int idx = tid; idx < 16 * BM; idx += 128) {
            const int mm = idx >> 4;
            const int rr = idx & 15;
            const int gm = m0 + mm;
            Xs[rr][mm] = (gm < M) ? x[(size_t)gm * tk * 16 + tk_i * 16 + rr] : 0.f;
        }
        __syncthreads();
        if (tn_i < tn) {
            const uint32_t* rg = rings[dtile];
            const int col = dtile * 16 + dc;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int dmz = dm + 2 * z;
                float f0, f1, f2, f3;
                pf_decode4(rg, plan[z], f0, f1, f2, f3);
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
            for (int in = 0; in < RN; ++in) wv[in] = Ws[r][tx + in * 8];
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
                const int gn = tn0 * 16 + tx + in * 8;
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
        if (tn_i < tn && du < w) {
            rings[dtile][du] = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
        }
        for (int idx = tid; idx < 16 * BM; idx += 128) {
            const int mm = idx >> 4;
            const int rr = idx & 15;
            const int gm = m0 + mm;
            Xs[rr][mm] = (gm < M) ? x[(size_t)gm * tk * 16 + tk_i * 16 + rr] : 0.f;
        }
        __syncthreads();
        if (tn_i < tn) {
            const uint32_t* rg = rings[dtile];
            const int col = dtile * 16 + dc;
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int dmz = dm + 2 * z;
                float f0, f1, f2, f3;
                pf_decode4(rg, plan[z], f0, f1, f2, f3);
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
    inkwell_exl3_had_in_prefill<<<gi, 128, 0, st>>>(x, (const uint16_t*)suh, xh, M, (int)k);
    const int tk = (int)(k / 16), tn = (int)(n / 16);
    if (M <= 4) pf_launch_gemm_core<4>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else if (M <= 8) pf_launch_gemm_core<8>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else if (M <= 16) pf_launch_gemm_core<16>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else if (M <= 32) pf_launch_gemm_core<32>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
    else {
        if (tn <= 72) {
            dim3 g((tn + 3) / 4, (M + 31) / 32);
            inkwell_exl3_gemm_narrow_core<<<g, 128, 0, st>>>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk);
        } else if ((int)ka >= 3 && (int)halfk) {
            pf_launch_gemm_core<64>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk, st);
        } else {
            dim3 g((tn + 3) / 4, (M + 63) / 64);
            inkwell_exl3_gemm_wide_core<<<g, 128, 0, st>>>((const uint32_t*)trel, xh, t, M, tk, tn, (int)ka, (int)halfk);
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

// Expanded-row Hadamard input for one expert descriptor table.
extern "C" __global__ void inkwell_moe_had_in_rows(
        const uint64_t* __restrict__ descs, const int* __restrict__ tile_expert_id,
        const int* __restrict__ sorted_token, const float* __restrict__ x,
        float* __restrict__ xh, int P, int k) {
    __shared__ float sm[128];
    const int row = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    const int ex = tile_expert_id[row];
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 1];
    const float xv = (row < P && e < k) ? x[(size_t)sorted_token[row] * k + e] : 0.f;
    sm[t] = (row < P && e < k) ? xv * __half2float(__ushort_as_half(suh[e])) : 0.f;
    __syncthreads();
    pf_had128_smem(sm, t);
    if (row < P && e < k) xh[(size_t)row * k + e] = sm[t] * PF_INV_SQRT128;
}

extern "C" __global__ void inkwell_moe_had_out_rows(
        const uint64_t* __restrict__ descs, const int* __restrict__ row_expert,
        const float* __restrict__ tmat, float* __restrict__ y, int P, int n) {
    __shared__ float sm[128];
    const int row = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    const int ex = row_expert[row];
    const uint16_t* svh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 2];
    sm[t] = (row < P && e < n) ? tmat[(size_t)row * n + e] : 0.f;
    __syncthreads();
    pf_had128_smem(sm, t);
    if (row < P && e < n) y[(size_t)row * n + e] = sm[t] * PF_INV_SQRT128
        * __half2float(__ushort_as_half(svh[e]));
}

// Grouped EXL3 core.  The externally visible launch contract still supplies
// one CTA per 64 output columns and 256 threads; even column CTAs cover 128
// columns and odd CTAs exit.  Rows are processed in unpadded 32-row pieces,
// which matches the realistic mean of 32 routed rows per expert.
__global__ void __launch_bounds__(256) inkwell_exl3_moe_grouped_gemm_kernel(
        const uint64_t* __restrict__ descs,
        const int* __restrict__ tile_expert, const int* __restrict__ tile_start,
        const int* __restrict__ tile_count, const int* __restrict__ tile_n,
        const float* __restrict__ xh, float* __restrict__ y,
        int tk, int tn, int tile_base_index) {
    if (blockIdx.x & 1) return;
    constexpr int BM = 32, BN = 128, RM = 4, RN = 4;
    const int mt = tile_base_index + blockIdx.y;
    if (mt >= *tile_n) return;
    const int ex = tile_expert[mt];
    const int row0 = tile_start[mt];
    const int rows = tile_count[mt];
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
    __shared__ float Ws[16][BN + 1];
    __shared__ float Xs[16][BM + 1];
    __shared__ uint32_t rings[8][80];

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
            if (tn_i < tn && du < w) {
                rings[dtile][du] = __ldg(trel + ((size_t)tk_i * tn + tn_i) * (size_t)w + du);
            }
            for (int idx = tid; idx < 16 * BM; idx += 256) {
                const int mm = idx >> 4;
                const int rr = idx & 15;
                Xs[rr][mm] = (mm < rcnt) ? xh[(size_t)(row0 + rbase + mm) * K + tk_i * 16 + rr] : 0.f;
            }
            __syncthreads();
            if (tn_i < tn) {
                const uint32_t* rg = rings[dtile];
                const int col = dtile * 16 + dc;
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int dmz = dm + 2 * z;
                    float f0, f1, f2, f3;
                    pf_decode4(rg, plan[z], f0, f1, f2, f3);
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
                for (int in = 0; in < RN; ++in) wv[in] = Ws[r][tx + in * 32];
#pragma unroll
                for (int im = 0; im < RM; ++im)
#pragma unroll
                    for (int in = 0; in < RN; ++in)
                        acc[im][in] = fmaf(xv[im], wv[in], acc[im][in]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int im = 0; im < RM; ++im) {
            const int lr = rbase + ty * RM + im;
            if (lr < rows) {
                const float off = KOFF * sx[im];
#pragma unroll
                for (int in = 0; in < RN; ++in) {
                    const int gn = tn0 * 16 + tx + in * 32;
                    if (gn < N) y[(size_t)(row0 + lr) * N + gn] = fmaf(KINV, acc[im][in], off);
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
        float beta, float lin_beta, int P, int ncols) {
    __shared__ float sm[128];
    const int row = blockIdx.y;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    const int ex = row_expert[row];
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs_d[ex * 4 + 1];
    float v = 0.f;
    if (row < P && e < ncols) {
        const float g = gu[(size_t)row * 2 * ncols + e];
        const float u = gu[(size_t)row * 2 * ncols + ncols + e];
        const float a = (beta * tanhf(g / beta)) / (1.0f + expf(-g));
        const float b = lin_beta * tanhf(u / lin_beta);
        v = (a * b) * __half2float(__ushort_as_half(suh[e]));
    }
    sm[t] = v;
    __syncthreads();
    pf_had128_smem(sm, t);
    if (row < P && e < ncols) xh[(size_t)row * ncols + e] = sm[t] * PF_INV_SQRT128;
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
__global__ void inkwell_moe_had_out_pair_rows(const uint64_t*, const uint64_t*, const int*, const float*, const float*, float*, int, int);

// Full grouped MoE pipeline over rows already sorted by inkwell_exl3_moe_group_sort.
// Workspace layout (floats unless noted): A[P*2304], B[P*2304], T[P*2304],
// then row_expert[P] as int.  Stream ordering lets the short-lived arrays
// share those three regions: xhg=A, xhu=B, tg/tu=T, gu=A, xhd=B, td=A,
// dn=T.  Peak MoE scratch is therefore 3*P*2304 floats rather than the
// naive Phase-1 layout; Phase 2 still owns the arena and expert streaming.
extern "C" void ink_launch_exl3_moe_grouped_gemm(
        const void* descs_g, const void* descs_u, const void* descs_d,
        const float* x, int C,
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
    float* dn = tg;
    int* row_expert = (int*)(tg + (size_t)P * (2 * NG));
    // row_expert is derived on-device: one row per expanded selection.
    // tile_start/count are inclusive contiguous ranges; expand them with a tiny kernel below.
    inkwell_moe_fill_row_expert<<<1, 128, 0, st>>>(tile_expert, tile_start, tile_count, tile_n, row_expert);

    dim3 hig(KG / 128, P), hug(KG / 128, P);
    inkwell_moe_had_in_rows<<<hig, 128, 0, st>>>((const uint64_t*)descs_g, row_expert, sorted_token, x, xhg, P, KG);
    inkwell_moe_had_in_rows<<<hug, 128, 0, st>>>((const uint64_t*)descs_u, row_expert, sorted_token, x, xhu, P, KG);
    const int max_tiles = P / 64 + 128;
    dim3 gg((NG / 16 + 3) / 4, max_tiles);
    inkwell_exl3_moe_grouped_gemm_kernel<<<gg, 256, 0, st>>>((const uint64_t*)descs_g,
        tile_expert, tile_start, tile_count, tile_n, xhg, tg, KG / 16, NG / 16, 0);
    inkwell_exl3_moe_grouped_gemm_kernel<<<gg, 256, 0, st>>>((const uint64_t*)descs_u,
        tile_expert, tile_start, tile_count, tile_n, xhu, tu, KG / 16, NG / 16, 0);
    dim3 hog(NG / 128, P, 2);
    // Write gate into gu[:,0,:] and up into gu[:,1,:] using a strided temporary layout.
    // tg/tu are contiguous rows; the epilogue kernels below interleave them into gu halves.
    inkwell_moe_had_out_pair_rows<<<hog, 128, 0, st>>>((const uint64_t*)descs_g, (const uint64_t*)descs_u,
        row_expert, tg, tu, gu, P, NG);
    dim3 hd(NG / 128, P);
    inkwell_moe_situ_had_rows<<<hd, 128, 0, st>>>((const uint64_t*)descs_d, row_expert, gu, xhd, 4.f, 25.f, P, NG);
    dim3 gd((ND / 16 + 3) / 4, max_tiles);
    inkwell_exl3_moe_grouped_gemm_kernel<<<gd, 256, 0, st>>>((const uint64_t*)descs_d,
        tile_expert, tile_start, tile_count, tile_n, xhd, td, NG / 16, ND / 16, 0);
    dim3 hod(ND / 128, P);
    inkwell_moe_had_out_rows<<<hod, 128, 0, st>>>((const uint64_t*)descs_d, row_expert, td, dn, P, ND);
    dim3 gs((ND + 255) / 256, C);
    inkwell_moe_weighted_scatter<<<gs, 256, 0, st>>>(dn, sorted_weight, inverse_row, out, C, ND);
}

__global__ void inkwell_moe_fill_row_expert(const int* tile_expert, const int* tile_start,
        const int* tile_count, const int* tile_n, int* row_expert) {
    if (threadIdx.x != 0) return;
    for (int t = 0; t < *tile_n; ++t)
        for (int i = 0; i < tile_count[t]; ++i)
            row_expert[tile_start[t] + i] = tile_expert[t];
}

__global__ void inkwell_moe_had_out_pair_rows(const uint64_t* descs_g, const uint64_t* descs_u,
        const int* row_expert, const float* tg, const float* tu, float* gu, int P, int n) {
    __shared__ float sm[128];
    const int row = blockIdx.y;
    const int half = blockIdx.z;
    const int base = blockIdx.x * 128;
    const int t = threadIdx.x;
    const int e = base + t;
    const int ex = row_expert[row];
    const uint64_t* descs = half ? descs_u : descs_g;
    const uint16_t* svh = (const uint16_t*)(uintptr_t)descs[ex * 4 + 2];
    const float* tm = half ? tu : tg;
    sm[t] = (row < P && e < n) ? tm[(size_t)row * n + e] : 0.f;
    __syncthreads();
    pf_had128_smem(sm, t);
    if (row < P && e < n) gu[(size_t)row * 2 * n + half * n + e] = sm[t] * PF_INV_SQRT128
        * __half2float(__ushort_as_half(svh[e]));
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

// ---------------- GQA-3 flash prefill attention ----------------
// Correctness-first GQA-3 kernel.  One CTA owns a 32-query tile and one KV
// head; K/V tiles are loaded once and reused for all three query heads.
// rel_w layout is [nh][1025][16]; r layout is [C,nh,16].  sinks is [C,nh].
// The effective key bias is dot(r[q,h], rel_w[h][dist]) - sinks[q,h]; the
// sink column is the fixed zero reference.  SWA physical slot = logical_pos % swa_win.
extern "C" __global__ void __launch_bounds__(128) inkwell_flash_prefill_gqa3(
        const __half* __restrict__ K, const __half* __restrict__ V,
        const float* __restrict__ Q, const float* __restrict__ r,
        const float* __restrict__ rel_w, const float* __restrict__ sinks,
        float* __restrict__ out, int C, int pos0, int kv_cap, int swa_win, float scale,
        const __half* __restrict__ KS, const __half* __restrict__ VS, int snap_start, int snap_len) {
    // Register-tiled flash attention.  One CTA owns BQ=32 queries for one KV
    // head and reuses every BK=32 K/V tile for the three GQA query heads.
    // QK uses a 4x4 register tile on 64 threads (two shared-memory loads per
    // four FMAs); PV uses a 4x8 register tile on all 128 threads with bank-
    // conflict-free strided columns.  Kf/Vf share one smem allocation.
    constexpr int BQ = 32, BK = 32;
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
                const __half* kbase = osnap ? KS + (size_t)hk * snap_len * 128 : Kh;
                const size_t krow = osnap ? (size_t)(logical - snap_start) : (size_t)slot;
                const uint2 packed = *(const uint2*)(kbase + krow * 128 + d);
                KVh[d][kr] = __ushort_as_half((unsigned short)(packed.x & 0xffffu));
                KVh[d + 1][kr] = __ushort_as_half((unsigned short)(packed.x >> 16));
                KVh[d + 2][kr] = __ushort_as_half((unsigned short)(packed.y & 0xffffu));
                KVh[d + 3][kr] = __ushort_as_half((unsigned short)(packed.y >> 16));
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
                const __half* vbase = vsnap ? VS + (size_t)hk * snap_len * 128 : Vh;
                const size_t vrow = vsnap ? (size_t)(logical - snap_start) : (size_t)slot;
                const uint2 packed = *(const uint2*)(vbase + vrow * 128 + d);
                KVh[d][kr] = __ushort_as_half((unsigned short)(packed.x & 0xffffu));
                KVh[d + 1][kr] = __ushort_as_half((unsigned short)(packed.x >> 16));
                KVh[d + 2][kr] = __ushort_as_half((unsigned short)(packed.y & 0xffffu));
                KVh[d + 3][kr] = __ushort_as_half((unsigned short)(packed.y >> 16));
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
    inkwell_flash_prefill_gqa3<<<g, 128, 0, (cudaStream_t)stream>>>((const __half*)K,
        (const __half*)V, Q, r, rel_w, sinks, out, C, pos0, kv_cap, swa_win, scale,
        nullptr, nullptr, 0, 0);
}

extern "C" void ink_launch_flash_prefill_gqa3_snap(const void* K, const void* V, const float* Q,
        const float* r, const float* rel_w, const float* sinks, float* out,
        int C, int pos0, int kv_cap, int swa_win, float scale,
        const void* KS, const void* VS, int snap_start, int snap_len, void* stream) {
    dim3 g((C + 31) / 32, 8);
    inkwell_flash_prefill_gqa3<<<g, 128, 0, (cudaStream_t)stream>>>((const __half*)K,
        (const __half*)V, Q, r, rel_w, sinks, out, C, pos0, kv_cap, swa_win, scale,
        (const __half*)KS, (const __half*)VS, snap_start, snap_len);
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
    __shared__ float sm[128];
    const int row = blockIdx.x; // C*heads rows
    const int tid = threadIdx.x;
    float* xr = x + (size_t)row * d;
    const float* sr = scale ? scale + (size_t)(row % heads) * d : nullptr;
    float ss = 0.f;
    for (int i = tid; i < d; i += blockDim.x) {
        const float v = xr[i] * (sr ? sr[i] : 1.f);
        ss = fmaf(v, v, ss);
    }
    sm[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const float norm = sqrtf(sm[0]) + 1e-6f;
    const float f = fminf(1.f, thr / norm);
    for (int i = tid; i < d; i += blockDim.x) xr[i] = xr[i] * (sr ? sr[i] : 1.f) * f;
    (void)C;
}

extern "C" void ink_launch_pf_clip_heads(float* x, const float* scale, int C, int heads,
        int d, float thr, void* stream) {
    inkwell_pf_clip_heads<<<C * heads, 128, 0, (cudaStream_t)stream>>>(x, scale, C, heads, d, thr);
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
extern "C" __global__ void inkwell_pf_dense_gemm(const float* __restrict__ x,
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

extern "C" void ink_launch_pf_dense_gemm(const float* x, const void* w, float* y,
        int M, int N, int K, int w_is_bf16, void* stream) {
    dim3 g((N + 15) / 16, (M + 15) / 16);
    inkwell_pf_dense_gemm<<<g, 256, 0, (cudaStream_t)stream>>>(x, w, y, M, N, K, w_is_bf16);
}

// Batched loop-controller chain in the reduction order of the serial fused
// rms2down + gf_apply kernels. blockIdx.y is the token row.
__global__ void inkwell_pf_ctrl_rms2down(const float* __restrict__ cur,
        const float* __restrict__ msg, const __nv_bfloat16* __restrict__ w,
        float* __restrict__ gate, int n, int rows, float eps) {
    __shared__ float s_sum[32];
    __shared__ float s_buf[4][32];
    const int tok = blockIdx.y;
    const float* __restrict__ x0 = cur + (size_t)tok * n;
    const float* __restrict__ x1 = msg + (size_t)tok * n;
    float* __restrict__ dst = gate + (size_t)tok * rows;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n4 = n >> 2;
    if (tid >= 24 && tid < 32) s_sum[tid] = 0.0f;

    float scales[2];
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
        scales[bx] = rsqrtf(__fdiv_rn(t, (float)n) + eps);
    }

    if (tid < 32) {
#pragma unroll
        for (int r = 0; r < 4; ++r) s_buf[r][tid] = 0.0f;
    }
    __syncthreads();

    const int r0 = blockIdx.x * 4;
    const size_t stride2 = (size_t)n;
    const __nv_bfloat162* wbase = (const __nv_bfloat162*)w + (size_t)r0 * stride2;
    const float2* x0_2 = (const float2*)x0;
    const float2* x1_2 = (const float2*)x1;
    const int half2 = n >> 1;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int c = tid; c < n; c += 256) {
        const bool in0 = c < half2;
        const float2 raw = in0 ? __ldg(&x0_2[c]) : __ldg(&x1_2[c - half2]);
        const float sc = in0 ? scales[0] : scales[1];
        const float2 tg = make_float2(__fmul_rn(sc, raw.x), __fmul_rn(sc, raw.y));
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const __nv_bfloat162 tw = wbase[(size_t)r * stride2 + c];
            acc[r] += __bfloat162float(tw.x) * tg.x;
            acc[r] += __bfloat162float(tw.y) * tg.y;
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], m);
        if (lane == 0) s_buf[r][wid] = acc[r];
    }
    __syncthreads();
    if (wid < 4) {
        const int r = wid;
        float v = s_buf[r][lane];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0 && r0 + r < rows) dst[r0 + r] = v;
    }
}

__global__ void inkwell_pf_ctrl_gf_apply(const __nv_bfloat16* __restrict__ w,
        const float* __restrict__ gate, const float* __restrict__ cur,
        const float* __restrict__ msg, float* __restrict__ dst, int n, int fo) {
    const int tok = blockIdx.y;
    const float* __restrict__ y = gate + (size_t)tok * 64;
    const float* __restrict__ cr = cur + (size_t)tok * n;
    const float* __restrict__ mr = msg + (size_t)tok * n;
    float* __restrict__ dr = dst + (size_t)tok * n;
    const int wtid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * 8 + wtid;
    if (i >= n) return;
    const float2 t = __ldg(&((const float2*)y)[lane]);
    const __nv_bfloat162 vg = ((const __nv_bfloat162*)(w + (size_t)i * 64))[lane];
    const __nv_bfloat162 vf = ((const __nv_bfloat162*)(w + (size_t)(fo + i) * 64))[lane];
    float g = 0.0f, f = 0.0f;
    g += __bfloat162float(vg.x) * t.x;
    g += __bfloat162float(vg.y) * t.y;
    f += __bfloat162float(vf.x) * t.x;
    f += __bfloat162float(vf.y) * t.y;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, m);
        f += __shfl_xor_sync(0xffffffffu, f, m);
    }
    if (lane == 0) {
        const float gn = tanhf(g), fg = tanhf(f);
        const float r = __fsub_rn(cr[i], __fmul_rn(fg, cr[i]));
        dr[i] = __fadd_rn(r, __fmul_rn(gn, mr[i]));
    }
}

extern "C" void ink_launch_pf_ctrl_exact(const float* cur, const float* msg,
        const void* down, const void* gf, float* gate, float* dst,
        int C, int n, int rows, float eps, int fo, void* stream) {
    const cudaStream_t st = (cudaStream_t)stream;
    inkwell_pf_ctrl_rms2down<<<dim3((unsigned)(rows / 4), (unsigned)C), 256, 0, st>>>(
        cur, msg, (const __nv_bfloat16*)down, gate, n, rows, eps);
    inkwell_pf_ctrl_gf_apply<<<dim3((unsigned)((n + 7) / 8), (unsigned)C), 256, 0, st>>>(
        (const __nv_bfloat16*)gf, gate, cur, msg, dst, n, fo);
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
        prob[e] = 1.f / (1.f + expf(-z));
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
    for (int j = 0; j < 8; ++j) {
        ids[t * 8 + j] = idv[j];
        weights[t * 8 + j] = prob[idv[j]] / sum * wscale;
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
// xor-shuffle trees, sequential softmax, and row accumulation order.  Grid
// z is the token; states are token-major [C,n] planes.
__global__ void __launch_bounds__(256) inkwell_pf_mix_exact(PfMixStates states,
        int rows, const float* __restrict__ q, float* __restrict__ out,
        int C, int embd, float scale) {
    __shared__ float s_rms[64];
    __shared__ float s_p[64];
    const int token = blockIdx.z;
    const int qc = blockIdx.y;
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
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= embd) return;
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

extern "C" void ink_launch_pf_mix(const void* const* states, int rows, const float* query,
        float* out, int C, int n, int qcols, float scale, float eps, void* stream) {
    PfMixStates tmp{};
    for (int i = 0; i < rows && i < 8; ++i) tmp.p[i] = (const float*)states[i];
    if (n == 32 * PF_MIX_NPL && rows <= 64) {
        dim3 g((n + 255) / 256, qcols, C);
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
