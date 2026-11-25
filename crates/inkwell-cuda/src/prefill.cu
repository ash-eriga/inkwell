// Chunked-prefill kernels for sm_61.  These are independent of the decode
// kernels in kernels.cu; nothing here runs unless a prefill driver invokes it.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
// Workspace layout (floats unless noted): xhg[P*2304], xhu[P*2304],
// tg[P*1152], tu[P*1152], gu[P*2304], xhd[P*1152], td[P*2304], dn[P*2304],
// then row_expert[P] as int.  This Phase-1 layout is deliberately simple;
// Phase 2 must stream it in four 32-expert tiles and reuse the arena.
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
    float* gu = tu + (size_t)P * NG;
    float* xhd = gu + (size_t)P * (2 * NG);
    float* td = xhd + (size_t)P * NG;
    float* dn = td + (size_t)P * ND;
    int* row_expert = (int*)(dn + (size_t)P * ND);
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
        float* __restrict__ out, int C, int pos0, int kv_cap, int swa_win, float scale) {
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
                const uint2 packed = *(const uint2*)(Kh + (size_t)slot * 128 + d);
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
                const uint2 packed = *(const uint2*)(Vh + (size_t)slot * 128 + d);
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
        (const __half*)V, Q, r, rel_w, sinks, out, C, pos0, kv_cap, swa_win, scale);
}

// Phase-2 H2D staging-ring contract (stub in Phase 1):
// The driver owns a 32 MiB device ring split into two 16 MiB buffers.  For each
// 32-expert tile t it must (1) wait for tile t-2 compute before reusing a
// buffer, (2) copy tile t+1 host-pinned trellis/suh/svh bytes on the side copy
// stream, (3) rewrite tile descriptors to the device ring addresses, and
// (4) record events so grouped GEMM on the main stream never reads an
// incomplete tile.  Phase 1 kernels take already valid device descriptors and
// intentionally know nothing about the ring.
