#pragma once
// Candidate kernels + exact ggml mul_mat_vec_f reference for the
// mmvf + MoE Hadamard study on GTX 1060 (sm_61).
// Expects crates/inkwell-cuda/src/kernels.cu to be included before this header.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace mh {

// ============================================================================
// 0. Exact ggml mul_mat_vec_f<__nv_bfloat16, float, 1, 256> replica
//    (matches ~/lfm-gtx1060-bench/src-inkling/ggml/src/ggml-cuda/mmvf.cu)
// ============================================================================
__global__ void mmvf_bf16_ref(
        const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ y,
        float* __restrict__ dst,
        int ncols2) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float buf_iw[];
    if (tid < 32) buf_iw[tid] = 0.0f;
    __syncthreads();

    const __nv_bfloat162* x2 = (const __nv_bfloat162*)(x + (size_t)row * (ncols2 * 2));
    const float2* y2 = (const float2*)y;
    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const __nv_bfloat162 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        sumf += __bfloat162float(tmpx.x) * tmpy.x;
        sumf += __bfloat162float(tmpx.y) * tmpy.y;
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) sumf += __shfl_xor_sync(0xffffffffu, sumf, m);
    buf_iw[tid >> 5] = sumf;
    __syncthreads();
    if (tid < 32) {
        sumf = buf_iw[tid];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sumf += __shfl_xor_sync(0xffffffffu, sumf, m);
    }
    if (tid == 0) dst[row] = sumf;
}

// ============================================================================
// 1. Multi-row BF16 GEMV (exact mmvf-256 reduction order -> BIT-IDENTICAL)
//    Loads y2[col2] ONCE per R rows, hiding BF16 weight load latency via ILP=R.
// ============================================================================
template <int R>
__global__ void mmvf_bf16_mr(
        const __nv_bfloat16* __restrict__ x,
        const float* __restrict__ y,
        float* __restrict__ dst,
        int nrows, int ncols2) {
    const int r0 = blockIdx.x * R;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    __shared__ float buf[R][32];
    if (tid < 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) buf[r][tid] = 0.0f;
    }
    __syncthreads();

    const float2* y2 = (const float2*)y;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.0f;

    const size_t stride2 = (size_t)ncols2;
    const __nv_bfloat162* base = (const __nv_bfloat162*)x + (size_t)r0 * stride2;

    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const float2 tmpy = __ldg(&y2[col2]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r < nrows) {
                const __nv_bfloat162 tw = base[(size_t)r * stride2 + col2];
                sum[r] += __bfloat162float(tw.x) * tmpy.x;
                sum[r] += __bfloat162float(tw.y) * tmpy.y;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], m);
        if (lane == 0) buf[r][wid] = sum[r];
    }
    __syncthreads();
    if (wid < R) {
        const int r = wid;
        float v = buf[r][lane];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0 && r0 + r < nrows) dst[r0 + r] = v;
    }
}

// ============================================================================
// 2. Parallel 4-warp router (128 threads, exact tie-break -> BIT-IDENTICAL)
// ============================================================================
__device__ __forceinline__ void router_par_128_body(
        const float* __restrict__ logits,
        const float* __restrict__ bias,
        float* __restrict__ out,
        int k, float wscale, float wfloor) {
    __shared__ float s_prb[128];
    __shared__ float s_wv[4];
    __shared__ int   s_wi[4];
    __shared__ int   s_best;
    __shared__ int   s_ids[16];

    const int t = threadIdx.x;
    const int wid = t >> 5, lane = t & 31;
    float sel = -INFINITY;
    if (t < 128) {
        const float l = logits[t];
        const float p = 1.0f / (1.0f + __expf(-l));
        s_prb[t] = p;
        sel = p + bias[t];
    }
    __syncthreads();

    for (int j = 0; j < k; ++j) {
        if (t < 128) {
            float bv = sel;
            int bi = t;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, m);
                const int   oi = __shfl_xor_sync(0xffffffffu, bi, m);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { s_wv[wid] = bv; s_wi[wid] = bi; }
        }
        __syncthreads();
        if (t == 0) {
            float bv = s_wv[0];
            int bi = s_wi[0];
#pragma unroll
            for (int w = 1; w < 4; ++w) {
                if (s_wv[w] > bv || (s_wv[w] == bv && s_wi[w] < bi)) {
                    bv = s_wv[w]; bi = s_wi[w];
                }
            }
            s_best = bi;
            s_ids[j] = bi;
        }
        __syncthreads();
        if (t == s_best) sel = -INFINITY;
    }
    if (t == 0) {
        float wsum = 0.0f;
        for (int j = 0; j < k; ++j) wsum += s_prb[s_ids[j]];
        if (wsum < wfloor) wsum = wfloor;
        const float inv = wscale / wsum;
        for (int j = 0; j < k; ++j) {
            out[j] = s_prb[s_ids[j]] * inv;
            out[k + j] = __int_as_float(s_ids[j]);
        }
    }
}

__global__ void router_par_128(
        const float* __restrict__ logits,
        const float* __restrict__ bias,
        float* __restrict__ out,
        int k, float wscale, float wfloor) {
    router_par_128_body(logits, bias, out, k, wscale, wfloor);
}

// ============================================================================
// 3. Fused mlp.gate [128, 2304] BF16 + parallel router (1 launch, BIT-IDENTICAL)
//    Atomic ticket counter lets the last block run router_par_128 immediately.
// ============================================================================
template <int R>
__global__ void gate_router_fused_exact(
        const __nv_bfloat16* __restrict__ w,
        const float* __restrict__ x,
        const float* __restrict__ bias,
        float* __restrict__ scratch_logits,
        unsigned int* __restrict__ ticket_ctr,
        float* __restrict__ out,
        int ncols2, int k, float wscale, float wfloor) {
    const int r0 = blockIdx.x * R;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    __shared__ float buf[R][32];
    __shared__ unsigned int ticket;
    if (tid < 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) buf[r][tid] = 0.0f;
    }
    __syncthreads();

    const float2* y2 = (const float2*)x;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.0f;

    const size_t stride2 = (size_t)ncols2;
    const __nv_bfloat162* base = (const __nv_bfloat162*)w + (size_t)r0 * stride2;

    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const float2 tmpy = __ldg(&y2[col2]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const __nv_bfloat162 tw = base[(size_t)r * stride2 + col2];
            sum[r] += __bfloat162float(tw.x) * tmpy.x;
            sum[r] += __bfloat162float(tw.y) * tmpy.y;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], m);
        if (lane == 0) buf[r][wid] = sum[r];
    }
    __syncthreads();
    if (wid < R) {
        const int r = wid;
        float v = buf[r][lane];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) scratch_logits[r0 + r] = v;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(ticket_ctr, 1u);
    __syncthreads();
    if (ticket != gridDim.x - 1) return;
    __threadfence();
    if (tid == 0) *ticket_ctr = 0u;

    // Load logits via __ldcg (written by other blocks) into smem then run exact router
    __shared__ float s_log[128];
    if (tid < 128) s_log[tid] = __ldcg(&scratch_logits[tid]);
    __syncthreads();
    router_par_128_body(s_log, bias, out, k, wscale, wfloor);
}

// ============================================================================
// 4. Fused mlp.gate + router with LDG.128 (warp-per-row, REORDERED - FLAGGED)
//    grid=(16,1,1), blk=256 (8 warps = 8 rows/blk). Stages x[2304] in smem.
// ============================================================================
__device__ __forceinline__ float bf16_lo(uint32_t u) {
    return __uint_as_float(u << 16);
}
__device__ __forceinline__ float bf16_hi(uint32_t u) {
    return __uint_as_float(u & 0xffff0000u);
}

__global__ void gate_router_fused_ldg128(
        const __nv_bfloat16* __restrict__ w,
        const float* __restrict__ x,
        const float* __restrict__ bias,
        float* __restrict__ scratch_logits,
        unsigned int* __restrict__ ticket_ctr,
        float* __restrict__ out,
        int k, float wscale, float wfloor) {
    __shared__ __align__(16) float sx[2304];
    __shared__ unsigned int ticket;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    // Stage x[2304] = 576 float4s
    for (int i = tid; i < 576; i += 256) {
        ((float4*)sx)[i] = ((const float4*)x)[i];
    }
    __syncthreads();

    const int row = blockIdx.x * 8 + wid;
    // 2304 bf16 = 288 uint4 (8 bf16 per uint4); 32 lanes -> 9 iters/lane
    const uint4* w4 = (const uint4*)(w + (size_t)row * 2304);
    float acc = 0.0f;
#pragma unroll
    for (int it = 0; it < 9; ++it) {
        const int idx = it * 32 + lane;
        const uint4 qw = __ldg(&w4[idx]);
        const float4 xa = ((const float4*)sx)[idx * 2];
        const float4 xb = ((const float4*)sx)[idx * 2 + 1];
        acc = fmaf(bf16_lo(qw.x), xa.x, acc);
        acc = fmaf(bf16_hi(qw.x), xa.y, acc);
        acc = fmaf(bf16_lo(qw.y), xa.z, acc);
        acc = fmaf(bf16_hi(qw.y), xa.w, acc);
        acc = fmaf(bf16_lo(qw.z), xb.x, acc);
        acc = fmaf(bf16_hi(qw.z), xb.y, acc);
        acc = fmaf(bf16_lo(qw.w), xb.z, acc);
        acc = fmaf(bf16_hi(qw.w), xb.w, acc);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) scratch_logits[row] = acc;
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(ticket_ctr, 1u);
    __syncthreads();
    if (ticket != gridDim.x - 1) return;
    __threadfence();
    if (tid == 0) *ticket_ctr = 0u;

    __shared__ float s_log[128];
    if (tid < 128) s_log[tid] = __ldcg(&scratch_logits[tid]);
    __syncthreads();
    router_par_128_body(s_log, bias, out, k, wscale, wfloor);
}

// ============================================================================
// 5. Controller rms2down: 0-sg-smem + multi-row R (BIT-IDENTICAL to shipped!)
//    Drops smem from 18,688 B -> ~256 B (100% occupancy), recomputes scales
//    with exact 1024-thr emulated tree, scales x0/x1 on the fly with __fmul_rn.
//    Supports batched B=1 (single) or B=3 (q,k,v via blockIdx.y) controllers!
// ============================================================================
template <int R, int B>
__global__ void rms2down_nosmem_mr(
        const float* const* __restrict__ x0_arr,
        const float* const* __restrict__ x1_arr,
        const __nv_bfloat16* const* __restrict__ w_arr,
        float* const* __restrict__ dst_arr,
        int n, float eps) {
    __shared__ float s_sum[32];
    __shared__ float s_buf[R][32];
    const int b = (B == 1) ? 0 : blockIdx.y;
    const float* __restrict__ x0 = x0_arr[b];
    const float* __restrict__ x1 = x1_arr[b];
    const __nv_bfloat16* __restrict__ w = w_arr[b];
    float* __restrict__ dst = dst_arr[b];

    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const int n4 = n >> 2; // 576
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
        for (int r = 0; r < R; ++r) s_buf[r][tid] = 0.0f;
    }
    __syncthreads();

    const int r0 = blockIdx.x * R;
    const size_t stride2 = (size_t)n; // 2304 pairs = 4608 bf16
    const __nv_bfloat162* wbase = (const __nv_bfloat162*)w + (size_t)r0 * stride2;
    const float2* x0_2 = (const float2*)x0;
    const float2* x1_2 = (const float2*)x1;
    const int half2 = n >> 1; // 1152

    float s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) s[r] = 0.0f;

    for (int c = tid; c < n; c += 256) {
        const bool in0 = (c < half2);
        const float2 raw = in0 ? __ldg(&x0_2[c]) : __ldg(&x1_2[c - half2]);
        const float sc = in0 ? scales[0] : scales[1];
        const float2 tg = make_float2(__fmul_rn(sc, raw.x), __fmul_rn(sc, raw.y));
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const __nv_bfloat162 tw = wbase[(size_t)r * stride2 + c];
            s[r] += __bfloat162float(tw.x) * tg.x;
            s[r] += __bfloat162float(tw.y) * tg.y;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) s[r] += __shfl_xor_sync(0xffffffffu, s[r], m);
        if (lane == 0) s_buf[r][wid] = s[r];
    }
    __syncthreads();
    if (wid < R) {
        const int r = wid;
        float v = s_buf[r][lane];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) dst[r0 + r] = v;
    }
}

// Reordered LDG.128 rms2down (1 warp/row, 8 rows/blk, grid=(8, B) -> REORDERED - FLAGGED)
template <int B>
__global__ void rms2down_ldg128(
        const float* const* __restrict__ x0_arr,
        const float* const* __restrict__ x1_arr,
        const __nv_bfloat16* const* __restrict__ w_arr,
        float* const* __restrict__ dst_arr,
        int n, float eps) {
    __shared__ float s_sum[32];
    const int b = (B == 1) ? 0 : blockIdx.y;
    const float* __restrict__ x0 = x0_arr[b];
    const float* __restrict__ x1 = x1_arr[b];
    const __nv_bfloat16* __restrict__ w = w_arr[b];
    float* __restrict__ dst = dst_arr[b];

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
            } else p[k] = 0.0f;
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

    const int row = blockIdx.x * 8 + wid;
    const uint4* w4 = (const uint4*)(w + (size_t)row * 4608);
    const float4* x0_4 = (const float4*)x0;
    const float4* x1_4 = (const float4*)x1;
    float acc = 0.0f;
    // 2304 elems = 288 uint4 -> 9 iters of 32 lanes
#pragma unroll
    for (int it = 0; it < 9; ++it) {
        const int idx = it * 32 + lane;
        const uint4 qw = __ldg(&w4[idx]);
        const float4 xa = __ldg(&x0_4[idx * 2]);
        const float4 xb = __ldg(&x0_4[idx * 2 + 1]);
        const float sc = scales[0];
        acc = fmaf(bf16_lo(qw.x), sc * xa.x, acc);
        acc = fmaf(bf16_hi(qw.x), sc * xa.y, acc);
        acc = fmaf(bf16_lo(qw.y), sc * xa.z, acc);
        acc = fmaf(bf16_hi(qw.y), sc * xa.w, acc);
        acc = fmaf(bf16_lo(qw.z), sc * xb.x, acc);
        acc = fmaf(bf16_hi(qw.z), sc * xb.y, acc);
        acc = fmaf(bf16_lo(qw.w), sc * xb.z, acc);
        acc = fmaf(bf16_hi(qw.w), sc * xb.w, acc);
    }
#pragma unroll
    for (int it = 0; it < 9; ++it) {
        const int idx = it * 32 + lane;
        const uint4 qw = __ldg(&w4[288 + idx]);
        const float4 xa = __ldg(&x1_4[idx * 2]);
        const float4 xb = __ldg(&x1_4[idx * 2 + 1]);
        const float sc = scales[1];
        acc = fmaf(bf16_lo(qw.x), sc * xa.x, acc);
        acc = fmaf(bf16_hi(qw.x), sc * xa.y, acc);
        acc = fmaf(bf16_lo(qw.y), sc * xa.z, acc);
        acc = fmaf(bf16_hi(qw.y), sc * xa.w, acc);
        acc = fmaf(bf16_lo(qw.z), sc * xb.x, acc);
        acc = fmaf(bf16_hi(qw.z), sc * xb.y, acc);
        acc = fmaf(bf16_lo(qw.w), sc * xb.z, acc);
        acc = fmaf(bf16_hi(qw.w), sc * xb.w, acc);
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) dst[row] = acc;
}

// ============================================================================
// 6. Controller gf_apply: batched B=1/3 BIT-IDENTICAL (0 barriers, 11 regs)
//    + LDG.128 (REORDERED)
// ============================================================================
template <int B>
__global__ void gf_apply_opt(
        const __nv_bfloat16* const* __restrict__ w_arr,
        const float* const* __restrict__ y_arr,
        const float* const* __restrict__ cur_arr,
        const float* const* __restrict__ msg_arr,
        float* const* __restrict__ dst_arr,
        int n, int fo) {
    const int b = (B == 1) ? 0 : blockIdx.y;
    const __nv_bfloat16* __restrict__ x = w_arr[b];
    const float* __restrict__ y = y_arr[b];
    const float* __restrict__ cur = cur_arr[b];
    const float* __restrict__ msg = msg_arr[b];
    float* __restrict__ dst = dst_arr[b];

    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * 8 + w;
    if (i >= n) return;
    const float2 t = __ldg(&((const float2*)y)[lane]);
    const __nv_bfloat162 vg = ((const __nv_bfloat162*)(x + (size_t)i * 64))[lane];
    const __nv_bfloat162 vf = ((const __nv_bfloat162*)(x + (size_t)(fo + i) * 64))[lane];
    float g = 0.0f, f = 0.0f;
    g += __bfloat162float(vg.x) * t.x; g += __bfloat162float(vg.y) * t.y;
    f += __bfloat162float(vf.x) * t.x; f += __bfloat162float(vf.y) * t.y;
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, m);
        f += __shfl_xor_sync(0xffffffffu, f, m);
    }
    if (lane == 0) {
        const float gn = tanhf(g), fg = tanhf(f);
        const float r = __fsub_rn(cur[i], __fmul_rn(fg, cur[i]));
        dst[i] = __fadd_rn(r, __fmul_rn(gn, msg[i]));
    }
}

// 8 threads per output row -> 1 uint4 (8 bf16 = 16B) per thread, 128B coalesced!
// 32 outputs per 256-thr block -> grid=(72, B, 1). REORDERED (FLAGGED).
template <int B>
__global__ void gf_apply_ldg128(
        const __nv_bfloat16* const* __restrict__ w_arr,
        const float* const* __restrict__ y_arr,
        const float* const* __restrict__ cur_arr,
        const float* const* __restrict__ msg_arr,
        float* const* __restrict__ dst_arr,
        int n, int fo) {
    __shared__ __align__(16) float sy[64];
    __shared__ float sout[32];
    const int b = (B == 1) ? 0 : blockIdx.y;
    const __nv_bfloat16* __restrict__ x = w_arr[b];
    const float* __restrict__ y = y_arr[b];
    const float* __restrict__ cur = cur_arr[b];
    const float* __restrict__ msg = msg_arr[b];
    float* __restrict__ dst = dst_arr[b];

    if (threadIdx.x < 64) sy[threadIdx.x] = y[threadIdx.x];
    __syncthreads();

    const int tid = threadIdx.x;
    const int sub = tid & 7;      // 0..7 -> 8 bf16 = 1 uint4
    const int loc = tid >> 3;     // 0..31 output within block
    const int i = blockIdx.x * 32 + loc;
    if (i < n) {
        const uint4 qg = __ldg(&((const uint4*)(x + (size_t)i * 64))[sub]);
        const uint4 qf = __ldg(&((const uint4*)(x + (size_t)(fo + i) * 64))[sub]);
        const float4 ya = ((const float4*)sy)[sub * 2];
        const float4 yb = ((const float4*)sy)[sub * 2 + 1];
        float g = 0.0f, f = 0.0f;
        g = fmaf(bf16_lo(qg.x), ya.x, g); g = fmaf(bf16_hi(qg.x), ya.y, g);
        g = fmaf(bf16_lo(qg.y), ya.z, g); g = fmaf(bf16_hi(qg.y), ya.w, g);
        g = fmaf(bf16_lo(qg.z), yb.x, g); g = fmaf(bf16_hi(qg.z), yb.y, g);
        g = fmaf(bf16_lo(qg.w), yb.z, g); g = fmaf(bf16_hi(qg.w), yb.w, g);

        f = fmaf(bf16_lo(qf.x), ya.x, f); f = fmaf(bf16_hi(qf.x), ya.y, f);
        f = fmaf(bf16_lo(qf.y), ya.z, f); f = fmaf(bf16_hi(qf.y), ya.w, f);
        f = fmaf(bf16_lo(qf.z), yb.x, f); f = fmaf(bf16_hi(qf.z), yb.y, f);
        f = fmaf(bf16_lo(qf.w), yb.z, f); f = fmaf(bf16_hi(qf.w), yb.w, f);

#pragma unroll
        for (int m = 4; m > 0; m >>= 1) {
            g += __shfl_xor_sync(0xffffffffu, g, m);
            f += __shfl_xor_sync(0xffffffffu, f, m);
        }
        if (sub == 0) {
            const float gn = tanhf(g), fg = tanhf(f);
            const float r = __fsub_rn(cur[i], __fmul_rn(fg, cur[i]));
            sout[loc] = __fadd_rn(r, __fmul_rn(gn, msg[i]));
        }
    }
    __syncthreads();
    if (tid < 32) {
        const int idx = blockIdx.x * 32 + tid;
        if (idx < n) dst[idx] = sout[tid];
    }
}

// ============================================================================
// 7. Fused attn_sink_proj [24, 2304] BF16 + rel_bias_pad [1025, 16] BF16 * r
//    + gather(rel_pos[0..n_kv]) - sinks -> kq_b[n_kv, 24]
//    Replaces 3 launches: sgemm_32x32x32_NT_vec + mul_mat_vec_f + relbias_gather!
//    grid=(24, 1, 1), blk=256.
// ============================================================================
__global__ void sink_relbias_fused(
        const __nv_bfloat16* __restrict__ w_sink,      // [24, 2304] BF16
        const float* __restrict__ inp0,                // [2304] F32
        const __nv_bfloat16* __restrict__ rel_bias_bf, // [1025, 16] BF16 (col-major: 16 contiguous per row)
        const float* __restrict__ r,                   // [16, 24] F32 (16 contiguous per head)
        const int* __restrict__ rel_pos,               // [n_kv] i32
        float* __restrict__ kq_b,                      // [n_kv, 24] F32
        int n_kv, int ncols2) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float buf_iw[32];
    __shared__ float s_sink;
    __shared__ float sr[16];
    if (tid < 32) buf_iw[tid] = 0.0f;
    if (tid < 16) sr[tid] = r[h * 16 + tid];
    __syncthreads();

    // Exact mmvf-256 dot for sinks[h]
    const __nv_bfloat162* x2 = (const __nv_bfloat162*)(w_sink + (size_t)h * (ncols2 * 2));
    const float2* y2 = (const float2*)inp0;
    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const __nv_bfloat162 tmpx = x2[col2];
        const float2 tmpy = __ldg(&y2[col2]);
        sumf += __bfloat162float(tmpx.x) * tmpy.x;
        sumf += __bfloat162float(tmpx.y) * tmpy.y;
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) sumf += __shfl_xor_sync(0xffffffffu, sumf, m);
    buf_iw[tid >> 5] = sumf;
    __syncthreads();
    if (tid < 32) {
        sumf = buf_iw[tid];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sumf += __shfl_xor_sync(0xffffffffu, sumf, m);
        if (tid == 0) s_sink = sumf;
    }
    __syncthreads();

    const float sk = s_sink;
    const float4 r0 = ((const float4*)sr)[0];
    const float4 r1 = ((const float4*)sr)[1];
    const float4 r2 = ((const float4*)sr)[2];
    const float4 r3 = ((const float4*)sr)[3];
    for (int j = tid; j < n_kv; j += 256) {
        const int pos = __ldg(&rel_pos[j]);
        const uint4* rb = (const uint4*)(rel_bias_bf + (size_t)pos * 16);
        const uint4 q0 = __ldg(&rb[0]);
        const uint4 q1 = __ldg(&rb[1]);
        float acc = 0.0f;
        acc = fmaf(bf16_lo(q0.x), r0.x, acc);
        acc = fmaf(bf16_hi(q0.x), r0.y, acc);
        acc = fmaf(bf16_lo(q0.y), r0.z, acc);
        acc = fmaf(bf16_hi(q0.y), r0.w, acc);
        acc = fmaf(bf16_lo(q0.z), r1.x, acc);
        acc = fmaf(bf16_hi(q0.z), r1.y, acc);
        acc = fmaf(bf16_lo(q0.w), r1.z, acc);
        acc = fmaf(bf16_hi(q0.w), r1.w, acc);
        acc = fmaf(bf16_lo(q1.x), r2.x, acc);
        acc = fmaf(bf16_hi(q1.x), r2.y, acc);
        acc = fmaf(bf16_lo(q1.y), r2.z, acc);
        acc = fmaf(bf16_hi(q1.y), r2.w, acc);
        acc = fmaf(bf16_lo(q1.z), r3.x, acc);
        acc = fmaf(bf16_hi(q1.z), r3.y, acc);
        acc = fmaf(bf16_lo(q1.w), r3.z, acc);
        acc = fmaf(bf16_hi(q1.w), r3.w, acc);
        kq_b[(size_t)h * n_kv + j] = __fsub_rn(acc, sk);
    }
}

// ============================================================================
// 8. Batched k/v_sconv.weight_proj [32, 2304] BF16 x 2 (1 launch, BIT-IDENTICAL)
// ============================================================================
template <int R>
__global__ void kv_sconv_batched(
        const __nv_bfloat16* __restrict__ wk,
        const __nv_bfloat16* __restrict__ wv,
        const float* __restrict__ xk,
        const float* __restrict__ xv,
        float* __restrict__ dk,
        float* __restrict__ dv,
        int nrows, int ncols2) {
    const __nv_bfloat16* x = blockIdx.y ? wv : wk;
    const float* y = blockIdx.y ? xv : xk;
    float* dst = blockIdx.y ? dv : dk;
    const int r0 = blockIdx.x * R;
    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;
    __shared__ float buf[R][32];
    if (tid < 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) buf[r][tid] = 0.0f;
    }
    __syncthreads();

    const float2* y2 = (const float2*)y;
    float sum[R];
#pragma unroll
    for (int r = 0; r < R; ++r) sum[r] = 0.0f;
    const size_t stride2 = (size_t)ncols2;
    const __nv_bfloat162* base = (const __nv_bfloat162*)x + (size_t)r0 * stride2;
    for (int col2 = tid; col2 < ncols2; col2 += 256) {
        const float2 tmpy = __ldg(&y2[col2]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r < nrows) {
                const __nv_bfloat162 tw = base[(size_t)r * stride2 + col2];
                sum[r] += __bfloat162float(tw.x) * tmpy.x;
                sum[r] += __bfloat162float(tw.y) * tmpy.y;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], m);
        if (lane == 0) buf[r][wid] = sum[r];
    }
    __syncthreads();
    if (wid < R) {
        const int r = wid;
        float v = buf[r][lane];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0 && r0 + r < nrows) dst[r0 + r] = v;
    }
}

// ============================================================================
// 9. MoE Hadamard Fixes (ALL 100% BIT-IDENTICAL!)
// ============================================================================

// H3b-1: Vectorized standalone had_in_moe (float4 + uint2, 0 idle warps!)
// grid=(k/256, nslots, nsel) with 64 thr/blk (2 warps) OR grid=(k/128, nslots, nsel) with 32 thr/blk.
// Using blk=64 for k=2304 (9 blocks*2 warps=18 exact) or blk=96 for k=1152 (3 blocks*3 warps=9 exact).
__global__ void had_in_moe_vec(
        const uint64_t* __restrict__ descs,
        const uint64_t* __restrict__ descs2,
        const int* __restrict__ ids,
        const float* __restrict__ x,
        float* __restrict__ xh, float* __restrict__ xh2,
        int xh2s, int k) {
    const int slot = blockIdx.y;
    const uint64_t* dd = blockIdx.z ? descs2 : descs;
    float* xho = blockIdx.z ? xh2 + (size_t)slot * xh2s : xh + (size_t)slot * k;
    const uint16_t* suh = (const uint16_t*)(uintptr_t)dd[ids[slot] * 4 + 1];
    const size_t xoff = descs2 ? 0 : (size_t)slot * k;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int base = (blockIdx.x * wpb + wid) * 128;
    if (base >= k) return;
    const int e0 = base + lane * 4;
    const float4 xv = *(const float4*)(x + xoff + e0);
    const uint2 sv = *(const uint2*)(suh + e0);
    float v[4];
    v[0] = xv.x * __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu)));
    v[1] = xv.y * __half2float(__ushort_as_half((uint16_t)(sv.x >> 16)));
    v[2] = xv.z * __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu)));
    v[3] = xv.w * __half2float(__ushort_as_half((uint16_t)(sv.y >> 16)));
    had128_warp(v, lane * 4);
    *(float4*)(xho + e0) = make_float4(
        v[0] * 0.08838834764831845f,
        v[1] * 0.08838834764831845f,
        v[2] * 0.08838834764831845f,
        v[3] * 0.08838834764831845f);
}

// H3b-2: Fused situ_pair + had_in_moe(Down) (1 launch replacing 2, BIT-IDENTICAL!)
// k = 1152 = 9 * 128. grid=(3, nslots, 1), blk=96 (3 warps = 3*128 = 384 elems/blk -> 0 idle warps!)
__global__ void situ_had_in_moe(
        const float* __restrict__ gu,
        const uint64_t* __restrict__ descs_d,
        const int* __restrict__ ids,
        float* __restrict__ xh,
        float situ_beta, float situ_lin_beta, int ncols) {
    const int slot = blockIdx.y;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int base = (blockIdx.x * 3 + wid) * 128;
    const int e0 = base + lane * 4;
    const float4 gv = *(const float4*)(gu + (size_t)slot * 2 * ncols + e0);
    const float4 uv = *(const float4*)(gu + (size_t)slot * 2 * ncols + ncols + e0);
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs_d[ids[slot] * 4 + 1];
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
        const float a = (situ_beta * tanhf(gf[i] / situ_beta)) / (1.0f + expf(-gf[i]));
        const float b = situ_lin_beta * tanhf(uf[i] / situ_lin_beta);
        const float h = a * b;
        v[i] = h * __half2float(__ushort_as_half(sh[i]));
    }
    had128_warp(v, lane * 4);
    *(float4*)(xh + (size_t)slot * ncols + e0) = make_float4(
        v[0] * 0.08838834764831845f,
        v[1] * 0.08838834764831845f,
        v[2] * 0.08838834764831845f,
        v[3] * 0.08838834764831845f);
}

// H3b-3: Fused had_out_moe(Down) + expert_wsum (1 launch replacing 2, BIT-IDENTICAL!)
// grid=(18, 1, 1), blk=256 (8 warps = 1 warp per slot s=0..7 on the same 128-strip!)
__global__ void had_out_wsum_moe(
        const uint64_t* __restrict__ descs_d,
        const int* __restrict__ ids,
        const float* __restrict__ dn,
        const float* __restrict__ w,
        float* __restrict__ out,
        int n) {
    __shared__ float sm[8][128];
    const int slot = threadIdx.x >> 5; // 0..7
    const int lane = threadIdx.x & 31;
    const int base = blockIdx.x * 128;
    const int e0 = lane * 4;
    const uint16_t* svh = (const uint16_t*)(uintptr_t)descs_d[ids[slot] * 4 + 2];
    const float4 dv = *(const float4*)(dn + (size_t)slot * n + base + e0);
    float v[4] = {dv.x, dv.y, dv.z, dv.w};
    had128_warp(v, e0);
    const uint2 sv = *(const uint2*)(svh + base + e0);
    sm[slot][e0 + 0] = v[0] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu)));
    sm[slot][e0 + 1] = v[1] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.x >> 16)));
    sm[slot][e0 + 2] = v[2] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu)));
    sm[slot][e0 + 3] = v[3] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.y >> 16)));
    __syncthreads();
    if (threadIdx.x < 128) {
        const int c = threadIdx.x;
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < 8; ++s) acc += w[s] * sm[s][c];
        out[base + c] = acc;
    }
}

// H3b-1b: Vectorized standalone had_out_moe (float4 + uint2, 0 idle warps!)
// grid=(n/384, nslots, nsel) with blk=96 (3 warps = 384 elems/blk) -> exact for n=1152 (3 blks) and n=2304 (6 blks).
__global__ void had_out_moe_vec(
        const uint64_t* __restrict__ descs,
        const uint64_t* __restrict__ descs2,
        const int* __restrict__ ids,
        const float* __restrict__ t,
        float* __restrict__ y, int n, int stride, int off,
        int tslot, int tzoff) {
    const int slot = blockIdx.y;
    const uint64_t* dd = blockIdx.z ? descs2 : descs;
    const uint16_t* svh = (const uint16_t*)(uintptr_t)dd[ids[slot] * 4 + 2];
    off += blockIdx.z * n;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wpb = blockDim.x >> 5;
    const int base = (blockIdx.x * wpb + wid) * 128;
    if (base >= n) return;
    const int e0 = base + lane * 4;
    const float4 tv = *(const float4*)(t + (size_t)slot * tslot + (size_t)blockIdx.z * tzoff + e0);
    float v[4] = {tv.x, tv.y, tv.z, tv.w};
    had128_warp(v, lane * 4);
    const uint2 sv = *(const uint2*)(svh + e0);
    *(float4*)(y + (size_t)slot * stride + off + e0) = make_float4(
        v[0] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu))),
        v[1] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.x >> 16))),
        v[2] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu))),
        v[3] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.y >> 16))));
}

// H2 & H3a: inkwell_exl3_gemv_moe_s with templated FUSE_IN (had_in prologue)
// and FUSE_OUT (had_out epilogue), nky=1, pw16=0 -> 100% BIT-IDENTICAL!
// __launch_bounds__(256, 6) keeps <=40 regs (6 blocks/SM, matching shipped gemv_moe_s!).
template <bool FUSE_IN, bool FUSE_OUT>
__global__ void __launch_bounds__(256, 6) gemv_moe_s_fused(
        const uint64_t* __restrict__ descs,
        const uint64_t* __restrict__ descs2,
        const int* __restrict__ ids,
        const float* __restrict__ x_in,   // raw x if FUSE_IN, else xh
        const float* __restrict__ xh2, int xh2s,
        float* __restrict__ y_out, int yslot, int yzoff,
        int tk, int tn, int k, int n) {
    const int slot = blockIdx.y;
    const int zsel = blockIdx.z;
    const uint64_t* d = (zsel ? descs2 : descs) + ids[slot] * 4;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff), hk = (int)((d[3] >> 8) & 1);
    const int kk = ka * 2 + hk;

    __shared__ __align__(16) float xs[SGEMV_MAXK];
    __shared__ float red[8];

    const int tid = threadIdx.x;
    const int wid = tid >> 5, lane = tid & 31;

    float sxl = 0.0f;
    if constexpr (FUSE_IN) {
        const uint16_t* suh = (const uint16_t*)(uintptr_t)d[1];
        const float* xsrc = descs2 ? x_in : (x_in + (size_t)slot * k);
        const int nb128 = k >> 7;
        for (int b = wid; b < nb128; b += 8) {
            const int e0 = b * 128 + lane * 4;
            const float4 xv = __ldg((const float4*)(xsrc + e0));
            const uint2 sv = __ldg((const uint2*)(suh + e0));
            float v[4];
            v[0] = xv.x * __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu)));
            v[1] = xv.y * __half2float(__ushort_as_half((uint16_t)(sv.x >> 16)));
            v[2] = xv.z * __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu)));
            v[3] = xv.w * __half2float(__ushort_as_half((uint16_t)(sv.y >> 16)));
            had128_warp(v, lane * 4);
            *(float4*)(xs + e0) = make_float4(
                v[0] * 0.08838834764831845f,
                v[1] * 0.08838834764831845f,
                v[2] * 0.08838834764831845f,
                v[3] * 0.08838834764831845f);
        }
        __syncthreads();
        for (int e = tid; e < k; e += 256) sxl += xs[e];
    } else {
        const float* xg = zsel ? xh2 + (size_t)slot * xh2s : x_in + (size_t)slot * k;
        for (int e = tid; e < k; e += 256) {
            const float v = xg[e];
            sxl += v;
            xs[e] = v;
        }
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) sxl += __shfl_xor_sync(0xffffffffu, sxl, s);
    if (lane == 0) red[wid] = sxl;
    __syncthreads();
    float sx = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx += red[i];

    const int tn_i = blockIdx.x * 8 + wid;
    float a0 = 0.0f, a1 = 0.0f;
    if (tn_i < tn) {
        switch (kk) {
            case 3:  sgemv_run<1, 1>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            case 4:  sgemv_run<2, 0>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            case 5:  sgemv_run<2, 1>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            case 6:  sgemv_run<3, 0>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            case 7:  sgemv_run<3, 1>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            case 8:  sgemv_run<4, 0>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            case 10: sgemv_run<5, 0>(tb, tn, tn_i, 0, tk, xs, a0, a1); break;
            default: __trap();
        }
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        a0 += __shfl_xor_sync(0xffffffffu, a0, 2); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    }

    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.0f * KINV + __half2float(__ushort_as_half(0xc931));
    const float off = KOFF * sx;

    if constexpr (FUSE_OUT) {
        __syncthreads();
        if (tn_i < tn && (lane & 3) == 0) {
            const int scol = wid * 16 + (lane >> 2);
            xs[scol]     = fmaf(KINV, a0, off);
            xs[scol + 8] = fmaf(KINV, a1, off);
        }
        __syncthreads();
        if (wid == 0) {
            const uint64_t* d2 = (zsel ? descs2 : descs) + ids[slot] * 4;
            const uint16_t* svh = (const uint16_t*)(uintptr_t)d2[2];
            const int base = blockIdx.x * 128;
            const int e0 = lane * 4;
            const float4 tv = *(const float4*)(xs + e0);
            float v[4] = {tv.x, tv.y, tv.z, tv.w};
            had128_warp(v, e0);
            const uint2 sv = *(const uint2*)(svh + base + e0);
            *(float4*)(y_out + (size_t)slot * yslot + (size_t)zsel * yzoff + base + e0) = make_float4(
                v[0] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.x & 0xffffu))),
                v[1] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.x >> 16))),
                v[2] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.y & 0xffffu))),
                v[3] * 0.08838834764831845f * __half2float(__ushort_as_half((uint16_t)(sv.y >> 16))));
        }
    } else {
        if (tn_i < tn && (lane & 3) == 0) {
            float* dst = &y_out[(size_t)slot * yslot + (size_t)zsel * yzoff + tn_i * 16 + (lane >> 2)];
            dst[0] = fmaf(KINV, a0, off);
            dst[8] = fmaf(KINV, a1, off);
        }
    }
}

} // namespace mh
