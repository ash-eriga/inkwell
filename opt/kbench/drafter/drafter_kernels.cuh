#pragma once
// Hand-tuned Pascal sm_61 (GTX 1060 6GB) kernels for speculative decoding drafter.
// All kernels are 100% CUDA-graph-capturable (zero host sync, device-side argmax output).
//
// Conventions:
//   - Weight matrices W[N, K] are row-major with K % 128 == 0, N % 128 == 0.
//   - int8 weights: symmetric per-output-row scale scale_w[N] (f32).
//   - int4 weights: unsigned nibble-packed [N, K/2] with offset +8:
//       each uint32_t word holds 8 weights:
//         low nibbles  (p & 0x0F0F0F0Fu)       = bytes [u0, u1, u2, u3] for cols k..k+3
//         high nibbles ((p >> 4) & 0x0F0F0F0Fu)= bytes [u4, u5, u6, u7] for cols k+4..k+7
//       where u_i = clamp(round(w_i / scale_w[row]) + 8, 0, 15).
//       Dot product: dp4a.s32.u32(xq_s8, u_u4, acc) - 8 * sum(xq_s8).
//   - Activations: [B, K] row-major int8_t (dp4a) or int16_t (dp2a), per-row scale_x[B] (f32)
//       and sum_x[B] (int32, sum of int8 elements for int4 offset correction).

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace drf {

// ---- PTX Intrinsics (sm_61) ----
__device__ __forceinline__ int dp4a_ss(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ int dp4a_su(int a, unsigned int b, int c) {
    int d;
    asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ unsigned int dp4a_uu(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int d;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ int dp2a_lo_ss(int a16x2, int b8x4, int c) {
    int d;
    asm("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a16x2), "r"(b8x4), "r"(c));
    return d;
}

__device__ __forceinline__ int dp2a_hi_ss(int a16x2, int b8x4, int c) {
    int d;
    asm("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a16x2), "r"(b8x4), "r"(c));
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

__device__ __forceinline__ float block_reduce_sum(float v, float* red8) {
#pragma unroll
    for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
    if ((threadIdx.x & 31) == 0) red8[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = (threadIdx.x < 8) ? red8[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) {
#pragma unroll
        for (int o = 4; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) red8[0] = s;
    }
    __syncthreads();
    return red8[0];
}

__device__ __forceinline__ float block_reduce_max(float v, float* red8) {
#pragma unroll
    for (int s = 16; s; s >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, s));
    if ((threadIdx.x & 31) == 0) red8[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = (threadIdx.x < 8) ? red8[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) {
#pragma unroll
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (threadIdx.x == 0) red8[0] = m;
    }
    __syncthreads();
    return red8[0];
}

__device__ __forceinline__ int block_reduce_sum_i32(int v, int* red8) {
#pragma unroll
    for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
    if ((threadIdx.x & 31) == 0) red8[threadIdx.x >> 5] = v;
    __syncthreads();
    int s = (threadIdx.x < 8) ? red8[threadIdx.x] : 0;
    if (threadIdx.x < 32) {
#pragma unroll
        for (int o = 4; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (threadIdx.x == 0) red8[0] = s;
    }
    __syncthreads();
    return red8[0];
}

// ============================================================================
// 1. FUSED RMSNORM + OPTIONAL RESIDUAL + QUANTIZE (int8 OR int16) + sum_x
//    Grid: (B), Block: 256 threads. Single DRAM pass over x, res, gamma!
//    K must be a multiple of 128, up to 9216 (uses smem buffer for K floats).
// ============================================================================
template <bool QUANT_I16>
__global__ void __launch_bounds__(256, 4) fused_rmsnorm_quant(
    const float* __restrict__ x,
    const float* __restrict__ res,      // optional (nullable)
    const float* __restrict__ gamma,    // [K]
    float* __restrict__ res_out,        // optional (nullable)
    void* __restrict__ xq_out,          // int8_t[B, K] or int16_t[B, K]
    float* __restrict__ scale_x,        // [B]
    int* __restrict__ sum_x,            // [B] (int32 sum of quantized elems)
    int K, float eps)
{
    extern __shared__ float y_sh[];     // [K] floats (<= 36 KB for K=9216, 9 KB for K=2304)
    __shared__ float red8[8];
    __shared__ int ired8[8];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const float* xb = x + (size_t)b * K;
    const float* rb = res ? (res + (size_t)b * K) : nullptr;
    float* rob = res_out ? (res_out + (size_t)b * K) : nullptr;

    float ss = 0.f;
    const int k4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    const float4* r4 = reinterpret_cast<const float4*>(rb);
    float4* ro4 = reinterpret_cast<float4*>(rob);
    float4* y4 = reinterpret_cast<float4*>(y_sh);

    for (int i = tid; i < k4; i += 256) {
        float4 v = x4[i];
        if (r4) {
            float4 rv = r4[i];
            v.x += rv.x; v.y += rv.y; v.z += rv.z; v.w += rv.w;
            if (ro4) ro4[i] = v;
        }
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        y4[i] = v;
    }
    const float inv_rms = rsqrtf(block_reduce_sum(ss, red8) / (float)K + eps);

    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    float amax = 0.f;
    for (int i = tid; i < k4; i += 256) {
        float4 v = y4[i];
        float4 g = g4[i];
        v.x *= inv_rms * g.x;
        v.y *= inv_rms * g.y;
        v.z *= inv_rms * g.z;
        v.w *= inv_rms * g.w;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        y4[i] = v;
    }
    const float gmax = block_reduce_max(amax, red8);

    if constexpr (!QUANT_I16) {
        const float sc = (gmax > 1e-12f) ? (gmax / 127.0f) : 1.0f;
        const float inv_sc = (gmax > 1e-12f) ? (127.0f / gmax) : 0.0f;
        int32_t* q4 = reinterpret_cast<int32_t*>(static_cast<int8_t*>(xq_out) + (size_t)b * K);
        int lsum = 0;
        for (int i = tid; i < k4; i += 256) {
            float4 v = y4[i];
            int q0 = max(-127, min(127, __float2int_rn(v.x * inv_sc)));
            int q1 = max(-127, min(127, __float2int_rn(v.y * inv_sc)));
            int q2 = max(-127, min(127, __float2int_rn(v.z * inv_sc)));
            int q3 = max(-127, min(127, __float2int_rn(v.w * inv_sc)));
            lsum += q0 + q1 + q2 + q3;
            uint32_t pk = ((uint32_t)(q0 & 0xFF))
                        | ((uint32_t)(q1 & 0xFF) << 8)
                        | ((uint32_t)(q2 & 0xFF) << 16)
                        | ((uint32_t)(q3 & 0xFF) << 24);
            q4[i] = (int32_t)pk;
        }
        int tsum = block_reduce_sum_i32(lsum, ired8);
        if (tid == 0) {
            scale_x[b] = sc;
            if (sum_x) sum_x[b] = tsum;
        }
    } else {
        const float sc = (gmax > 1e-12f) ? (gmax / 32767.0f) : 1.0f;
        const float inv_sc = (gmax > 1e-12f) ? (32767.0f / gmax) : 0.0f;
        int2* q4 = reinterpret_cast<int2*>(static_cast<int16_t*>(xq_out) + (size_t)b * K);
        for (int i = tid; i < k4; i += 256) {
            float4 v = y4[i];
            int q0 = max(-32767, min(32767, __float2int_rn(v.x * inv_sc)));
            int q1 = max(-32767, min(32767, __float2int_rn(v.y * inv_sc)));
            int q2 = max(-32767, min(32767, __float2int_rn(v.z * inv_sc)));
            int q3 = max(-32767, min(32767, __float2int_rn(v.w * inv_sc)));
            int2 pk;
            pk.x = (int32_t)(((uint32_t)(q0 & 0xFFFF)) | ((uint32_t)(q1 & 0xFFFF) << 16));
            pk.y = (int32_t)(((uint32_t)(q2 & 0xFFFF)) | ((uint32_t)(q3 & 0xFFFF) << 16));
            q4[i] = pk;
        }
        if (tid == 0) {
            scale_x[b] = sc;
            if (sum_x) sum_x[b] = 0;
        }
    }
}

// Standalone quantize (e.g., after SwiGLU or attention out) -> int8/int16
template <bool QUANT_I16>
__global__ void __launch_bounds__(256, 6) quantize_act(
    const float* __restrict__ x,
    void* __restrict__ xq_out,
    float* __restrict__ scale_x,
    int* __restrict__ sum_x,
    int K)
{
    __shared__ float red8[8];
    __shared__ int ired8[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int k4 = K >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)b * K);
    float amax = 0.f;
    for (int i = tid; i < k4; i += 256) {
        float4 v = x4[i];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    const float gmax = block_reduce_max(amax, red8);
    if constexpr (!QUANT_I16) {
        const float sc = (gmax > 1e-12f) ? (gmax / 127.0f) : 1.0f;
        const float inv_sc = (gmax > 1e-12f) ? (127.0f / gmax) : 0.0f;
        int32_t* q4 = reinterpret_cast<int32_t*>(static_cast<int8_t*>(xq_out) + (size_t)b * K);
        int lsum = 0;
        for (int i = tid; i < k4; i += 256) {
            float4 v = x4[i];
            int q0 = max(-127, min(127, __float2int_rn(v.x * inv_sc)));
            int q1 = max(-127, min(127, __float2int_rn(v.y * inv_sc)));
            int q2 = max(-127, min(127, __float2int_rn(v.z * inv_sc)));
            int q3 = max(-127, min(127, __float2int_rn(v.w * inv_sc)));
            lsum += q0 + q1 + q2 + q3;
            q4[i] = (int32_t)(((uint32_t)(q0 & 0xFF)) | ((uint32_t)(q1 & 0xFF) << 8)
                            | ((uint32_t)(q2 & 0xFF) << 16) | ((uint32_t)(q3 & 0xFF) << 24));
        }
        int tsum = block_reduce_sum_i32(lsum, ired8);
        if (tid == 0) { scale_x[b] = sc; if (sum_x) sum_x[b] = tsum; }
    } else {
        const float sc = (gmax > 1e-12f) ? (gmax / 32767.0f) : 1.0f;
        const float inv_sc = (gmax > 1e-12f) ? (32767.0f / gmax) : 0.0f;
        int2* q4 = reinterpret_cast<int2*>(static_cast<int16_t*>(xq_out) + (size_t)b * K);
        for (int i = tid; i < k4; i += 256) {
            float4 v = x4[i];
            int q0 = max(-32767, min(32767, __float2int_rn(v.x * inv_sc)));
            int q1 = max(-32767, min(32767, __float2int_rn(v.y * inv_sc)));
            int q2 = max(-32767, min(32767, __float2int_rn(v.z * inv_sc)));
            int q3 = max(-32767, min(32767, __float2int_rn(v.w * inv_sc)));
            int2 pk;
            pk.x = (int32_t)(((uint32_t)(q0 & 0xFFFF)) | ((uint32_t)(q1 & 0xFFFF) << 16));
            pk.y = (int32_t)(((uint32_t)(q2 & 0xFFFF)) | ((uint32_t)(q3 & 0xFFFF) << 16));
            q4[i] = pk;
        }
        if (tid == 0) { scale_x[b] = sc; if (sum_x) sum_x[b] = 0; }
    }
}

// Fused SiLU(gate) * up + quantize to int8 (saves an entire DRAM roundtrip in FFN!)
__global__ void __launch_bounds__(256, 6) fused_swiglu_quant_i8(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int8_t* __restrict__ xq_out,
    float* __restrict__ scale_x,
    int* __restrict__ sum_x,
    int K)
{
    extern __shared__ float y_sh[];
    __shared__ float red8[8];
    __shared__ int ired8[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int k4 = K >> 2;
    const float4* g4 = reinterpret_cast<const float4*>(gate + (size_t)b * K);
    const float4* u4 = reinterpret_cast<const float4*>(up + (size_t)b * K);
    float4* y4 = reinterpret_cast<float4*>(y_sh);
    float amax = 0.f;
    for (int i = tid; i < k4; i += 256) {
        float4 g = g4[i], u = u4[i], v;
        v.x = (g.x / (1.f + __expf(-g.x))) * u.x;
        v.y = (g.y / (1.f + __expf(-g.y))) * u.y;
        v.z = (g.z / (1.f + __expf(-g.z))) * u.z;
        v.w = (g.w / (1.f + __expf(-g.w))) * u.w;
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        y4[i] = v;
    }
    const float gmax = block_reduce_max(amax, red8);
    const float sc = (gmax > 1e-12f) ? (gmax / 127.0f) : 1.0f;
    const float inv_sc = (gmax > 1e-12f) ? (127.0f / gmax) : 0.0f;
    int32_t* q4 = reinterpret_cast<int32_t*>(xq_out + (size_t)b * K);
    int lsum = 0;
    for (int i = tid; i < k4; i += 256) {
        float4 v = y4[i];
        int q0 = max(-127, min(127, __float2int_rn(v.x * inv_sc)));
        int q1 = max(-127, min(127, __float2int_rn(v.y * inv_sc)));
        int q2 = max(-127, min(127, __float2int_rn(v.z * inv_sc)));
        int q3 = max(-127, min(127, __float2int_rn(v.w * inv_sc)));
        lsum += q0 + q1 + q2 + q3;
        q4[i] = (int32_t)(((uint32_t)(q0 & 0xFF)) | ((uint32_t)(q1 & 0xFF) << 8)
                        | ((uint32_t)(q2 & 0xFF) << 16) | ((uint32_t)(q3 & 0xFF) << 24));
    }
    int tsum = block_reduce_sum_i32(lsum, ired8);
    if (tid == 0) { scale_x[b] = sc; if (sum_x) sum_x[b] = tsum; }
}

// ============================================================================
// 2. INT8 x INT8 DP4A GEMV (B = 1..4 rows, 8 warps/block = 8 output rows/block)
//    16-byte coalesced int4 loads along K; xq staged in smem for B rows.
//    Grid: (N / 8), Block: 256 threads.
// ============================================================================
template <int B>
__global__ void __launch_bounds__(256, (B <= 3 ? 6 : 5)) gemv_i8_dp4a(
    const int8_t* __restrict__ W,       // [N, K]
    const float* __restrict__ scale_w,  // [N]
    const int8_t* __restrict__ xq,      // [B, K]
    const float* __restrict__ scale_x,  // [B]
    float* __restrict__ y,              // [B, N]
    int N, int K)
{
    extern __shared__ int4 xs_i4[];     // [B * (K/16)] int4s (e.g. B=3, K=2304 -> 6912 B)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp;
    const int k16 = K >> 4;

    // Coalesced load of xq[B, K] into shared memory
    const int4* xg = reinterpret_cast<const int4*>(xq);
    const int tot_x = B * k16;
    for (int i = tid; i < tot_x; i += 256) xs_i4[i] = xg[i];
    __syncthreads();

    if (row >= N) return;

    const int4* wp = reinterpret_cast<const int4*>(W + (size_t)row * K);
    int acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0;

    // 1 warp reads 32 * 16B = 512B contiguous per step -> 100% coalesced
    for (int i = lane; i < k16; i += 32) {
        const int4 wv = __ldg(wp + i);
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int4 xv = xs_i4[b * k16 + i];
            acc[b] = dp4a_ss(xv.x, wv.x, acc[b]);
            acc[b] = dp4a_ss(xv.y, wv.y, acc[b]);
            acc[b] = dp4a_ss(xv.z, wv.z, acc[b]);
            acc[b] = dp4a_ss(xv.w, wv.w, acc[b]);
        }
    }

    const float sw = scale_w[row];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        int v = acc[b];
#pragma unroll
        for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
        if (lane == 0) y[(size_t)b * N + row] = (float)v * (sw * scale_x[b]);
    }
}

// ============================================================================
// 3. INT16 x INT8 DP2A GEMV (B = 1..3 rows, 8 warps/block = 8 output rows/block)
//    Same 1 byte/weight DRAM footprint as dp4a, 256x higher activation precision!
// ============================================================================
template <int B>
__global__ void __launch_bounds__(256, 6) gemv_i8_dp2a(
    const int8_t* __restrict__ W,       // [N, K]
    const float* __restrict__ scale_w,  // [N]
    const int16_t* __restrict__ xq16,   // [B, K]
    const float* __restrict__ scale_x,  // [B]
    float* __restrict__ y,              // [B, N]
    int N, int K)
{
    extern __shared__ int4 xs16_i4[];   // [B * (K/8)] int4s (each int4 = 8 int16s = 16 B)
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp;
    const int k16 = K >> 4;             // number of 16-weight chunks
    const int k8  = K >> 3;             // number of int4 (16B) x16 chunks = 2 * k16

    const int4* xg = reinterpret_cast<const int4*>(xq16);
    const int tot_x = B * k8;
    for (int i = tid; i < tot_x; i += 256) xs16_i4[i] = xg[i];
    __syncthreads();

    if (row >= N) return;

    const int4* wp = reinterpret_cast<const int4*>(W + (size_t)row * K);
    int acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0;

    for (int i = lane; i < k16; i += 32) {
        const int4 wv = __ldg(wp + i);  // 16 int8 weights: wv.x(0..3), wv.y(4..7), wv.z(8..11), wv.w(12..15)
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int4 x0 = xs16_i4[b * k8 + 2 * i];     // 8 int16s: cols 0..7
            const int4 x1 = xs16_i4[b * k8 + 2 * i + 1]; // 8 int16s: cols 8..15
            acc[b] = dp2a_lo_ss(x0.x, wv.x, acc[b]);
            acc[b] = dp2a_hi_ss(x0.y, wv.x, acc[b]);
            acc[b] = dp2a_lo_ss(x0.z, wv.y, acc[b]);
            acc[b] = dp2a_hi_ss(x0.w, wv.y, acc[b]);
            acc[b] = dp2a_lo_ss(x1.x, wv.z, acc[b]);
            acc[b] = dp2a_hi_ss(x1.y, wv.z, acc[b]);
            acc[b] = dp2a_lo_ss(x1.z, wv.w, acc[b]);
            acc[b] = dp2a_hi_ss(x1.w, wv.w, acc[b]);
        }
    }

    const float sw = scale_w[row];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        int v = acc[b];
#pragma unroll
        for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
        if (lane == 0) y[(size_t)b * N + row] = (float)v * (sw * scale_x[b]);
    }
}

// ============================================================================
// 4. INT4 (NIBBLE-PACKED +8) x INT8 DP4A GEMV (B = 1..3 rows)
//    0.5 bytes/weight! One uint4 (16B) load = 32 weights!
//    Word layout: each uint32_t p holds 8 weights:
//      low  nibbles (p & 0x0F0F0F0Fu)       -> cols [0..3]
//      high nibbles ((p>>4) & 0x0F0F0F0Fu)  -> cols [4..7]
// ============================================================================
template <int B>
__global__ void __launch_bounds__(256, 6) gemv_i4_dp4a(
    const uint8_t* __restrict__ W4,     // [N, K/2] packed
    const float* __restrict__ scale_w,  // [N]
    const int8_t* __restrict__ xq,      // [B, K]
    const float* __restrict__ scale_x,  // [B]
    const int* __restrict__ sum_x,      // [B]
    float* __restrict__ y,              // [B, N]
    int N, int K)
{
    extern __shared__ int4 xs_i4[];     // [B * (K/16)] int4s
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * 8 + warp;
    const int k16 = K >> 4;
    const int k32 = K >> 5;

    const int4* xg = reinterpret_cast<const int4*>(xq);
    const int tot_x = B * k16;
    for (int i = tid; i < tot_x; i += 256) xs_i4[i] = xg[i];
    __syncthreads();

    if (row >= N) return;

    const uint4* wp = reinterpret_cast<const uint4*>(W4 + (size_t)row * (K >> 1));
    int acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0;

    constexpr uint32_t MASK = 0x0F0F0F0Fu;
    for (int i = lane; i < k32; i += 32) {
        const uint4 wv = __ldg(wp + i); // 32 weights
        const uint32_t w0l = wv.x & MASK, w0h = (wv.x >> 4) & MASK;
        const uint32_t w1l = wv.y & MASK, w1h = (wv.y >> 4) & MASK;
        const uint32_t w2l = wv.z & MASK, w2h = (wv.z >> 4) & MASK;
        const uint32_t w3l = wv.w & MASK, w3h = (wv.w >> 4) & MASK;
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const int4 x0 = xs_i4[b * k16 + 2 * i];     // cols 0..15
            const int4 x1 = xs_i4[b * k16 + 2 * i + 1]; // cols 16..31
            acc[b] = dp4a_su(x0.x, w0l, acc[b]);
            acc[b] = dp4a_su(x0.y, w0h, acc[b]);
            acc[b] = dp4a_su(x0.z, w1l, acc[b]);
            acc[b] = dp4a_su(x0.w, w1h, acc[b]);
            acc[b] = dp4a_su(x1.x, w2l, acc[b]);
            acc[b] = dp4a_su(x1.y, w2h, acc[b]);
            acc[b] = dp4a_su(x1.z, w3l, acc[b]);
            acc[b] = dp4a_su(x1.w, w3h, acc[b]);
        }
    }

    const float sw = scale_w[row];
#pragma unroll
    for (int b = 0; b < B; ++b) {
        int v = acc[b];
#pragma unroll
        for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
        if (lane == 0) y[(size_t)b * N + row] = (float)(v - 8 * sum_x[b]) * (sw * scale_x[b]);
    }
}

// ============================================================================
// 5. FUSED HEAD + ARGMAX + VOCAB_ID REMAP (SINGLE LAUNCH, ZERO HOST SYNC!)
//    Atomic-ticket last-block finalize: zero logit buffer in DRAM!
//    Supports: (a) int8 dp4a, (b) int8 dp2a (int16 act), (c) int4 dp4a.
//    Grid: G blocks (e.g. 120 or 160), Block: 256 threads (8 warps).
//    Scratch: G * B floats (blk_val), G * B uint32_t (blk_id), 1 uint32_t ticket (zeroed initially, self-resetting).
// ============================================================================
template <int B>
__device__ __forceinline__ void finalize_head_argmax(
    float* best_v, uint32_t* best_i,
    const uint32_t* __restrict__ vocab_ids,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    float* red_v, uint32_t* red_i)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int G = gridDim.x;

    // Reduce across 8 warps in this block (only lane 0 of each warp holds that warp's best)
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            red_v[b * 8 + warp] = best_v[b];
            red_i[b * 8 + warp] = best_i[b];
        }
    }
    __syncthreads();

    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float bv = -1e30f; uint32_t bi = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                float v = red_v[b * 8 + w];
                if (v > bv) { bv = v; bi = red_i[b * 8 + w]; }
            }
            blk_val[b * G + blockIdx.x] = bv;
            blk_id[b * G + blockIdx.x] = vocab_ids ? vocab_ids[bi] : bi;
        }
        __threadfence();
        uint32_t done = atomicAdd(ticket, 1u) + 1u;
        red_i[0] = (done == (uint32_t)G) ? 1u : 0u;
    }
    __syncthreads();

    if (red_i[0]) {
        // Last block finishes global argmax across G block partials
        for (int b = 0; b < B; ++b) {
            float bv = -1e30f; uint32_t bi = 0;
            for (int g = tid; g < G; g += 256) {
                float v = __ldcg(&blk_val[b * G + g]);
                uint32_t id = __ldcg(&blk_id[b * G + g]);
                if (v > bv) { bv = v; bi = id; }
            }
#pragma unroll
            for (int s = 16; s; s >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, s);
                uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, s);
                if (ov > bv) { bv = ov; bi = oi; }
            }
            if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
            __syncthreads();
            if (tid == 0) {
                float fbv = -1e30f; uint32_t fbi = 0;
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    if (red_v[w] > fbv) { fbv = red_v[w]; fbi = red_i[w]; }
                }
                out_tok[b] = fbi;
            }
            __syncthreads();
        }
        if (tid == 0) *ticket = 0u;
    }
}

template <int B>
__global__ void __launch_bounds__(256, 6) head_argmax_i8_dp4a(
    const int8_t* __restrict__ W,
    const float* __restrict__ scale_w,
    const int8_t* __restrict__ xq,
    const float* __restrict__ scale_x,
    const uint32_t* __restrict__ vocab_ids,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    int V, int K)
{
    extern __shared__ int4 xs_i4[];
    __shared__ float red_v[B * 8];
    __shared__ uint32_t red_i[B * 8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k16 = K >> 4;

    const int4* xg = reinterpret_cast<const int4*>(xq);
    const int tot_x = B * k16;
    for (int i = tid; i < tot_x; i += 256) xs_i4[i] = xg[i];
    __syncthreads();

    float best_v[B]; uint32_t best_i[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { best_v[b] = -1e30f; best_i[b] = 0; }

    // Note: scale_x[b] > 0 is constant across V -> argmax(logit) == argmax(acc * scale_w[row])!
    // Still multiply by scale_w[row] (row-dependent!)
    const int stride = gridDim.x * 8;
    for (int row = blockIdx.x * 8 + warp; row < V; row += stride) {
        const int4* wp = reinterpret_cast<const int4*>(W + (size_t)row * K);
        int acc[B];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = 0;
        for (int i = lane; i < k16; i += 32) {
            const int4 wv = __ldg(wp + i);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int4 xv = xs_i4[b * k16 + i];
                acc[b] = dp4a_ss(xv.x, wv.x, acc[b]);
                acc[b] = dp4a_ss(xv.y, wv.y, acc[b]);
                acc[b] = dp4a_ss(xv.z, wv.z, acc[b]);
                acc[b] = dp4a_ss(xv.w, wv.w, acc[b]);
            }
        }
        const float sw = scale_w[row];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int v = acc[b];
#pragma unroll
            for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0) {
                float logit = (float)v * sw; // scale_x[b] > 0 cancels in argmax
                if (logit > best_v[b]) { best_v[b] = logit; best_i[b] = (uint32_t)row; }
            }
        }
    }

    finalize_head_argmax<B>(best_v, best_i, vocab_ids, blk_val, blk_id, ticket, out_tok, red_v, red_i);
}

template <int B>
__global__ void __launch_bounds__(256, 5) head_argmax_i8_dp2a(
    const int8_t* __restrict__ W,
    const float* __restrict__ scale_w,
    const int16_t* __restrict__ xq16,
    const float* __restrict__ scale_x,
    const uint32_t* __restrict__ vocab_ids,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    int V, int K)
{
    extern __shared__ int4 xs16_i4[];
    __shared__ float red_v[B * 8];
    __shared__ uint32_t red_i[B * 8];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k16 = K >> 4;
    const int k8  = K >> 3;

    const int4* xg = reinterpret_cast<const int4*>(xq16);
    const int tot_x = B * k8;
    for (int i = tid; i < tot_x; i += 256) xs16_i4[i] = xg[i];
    __syncthreads();

    float best_v[B]; uint32_t best_i[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { best_v[b] = -1e30f; best_i[b] = 0; }

    const int stride = gridDim.x * 8;
    for (int row = blockIdx.x * 8 + warp; row < V; row += stride) {
        const int4* wp = reinterpret_cast<const int4*>(W + (size_t)row * K);
        int acc[B];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = 0;
        for (int i = lane; i < k16; i += 32) {
            const int4 wv = __ldg(wp + i);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int4 x0 = xs16_i4[b * k8 + 2 * i];
                const int4 x1 = xs16_i4[b * k8 + 2 * i + 1];
                acc[b] = dp2a_lo_ss(x0.x, wv.x, acc[b]);
                acc[b] = dp2a_hi_ss(x0.y, wv.x, acc[b]);
                acc[b] = dp2a_lo_ss(x0.z, wv.y, acc[b]);
                acc[b] = dp2a_hi_ss(x0.w, wv.y, acc[b]);
                acc[b] = dp2a_lo_ss(x1.x, wv.z, acc[b]);
                acc[b] = dp2a_hi_ss(x1.y, wv.z, acc[b]);
                acc[b] = dp2a_lo_ss(x1.z, wv.w, acc[b]);
                acc[b] = dp2a_hi_ss(x1.w, wv.w, acc[b]);
            }
        }
        const float sw = scale_w[row];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int v = acc[b];
#pragma unroll
            for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0) {
                float logit = (float)v * sw;
                if (logit > best_v[b]) { best_v[b] = logit; best_i[b] = (uint32_t)row; }
            }
        }
    }

    finalize_head_argmax<B>(best_v, best_i, vocab_ids, blk_val, blk_id, ticket, out_tok, red_v, red_i);
}

template <int B>
__global__ void __launch_bounds__(256, 6) head_argmax_i4_dp4a(
    const uint8_t* __restrict__ W4,
    const float* __restrict__ scale_w,
    const int8_t* __restrict__ xq,
    const int* __restrict__ sum_x,
    const uint32_t* __restrict__ vocab_ids,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    int V, int K)
{
    extern __shared__ int4 xs_i4[];
    __shared__ float red_v[B * 8];
    __shared__ uint32_t red_i[B * 8];
    __shared__ int sx_sh[B];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k16 = K >> 4;
    const int k32 = K >> 5;

    const int4* xg = reinterpret_cast<const int4*>(xq);
    const int tot_x = B * k16;
    for (int i = tid; i < tot_x; i += 256) xs_i4[i] = xg[i];
    if (tid < B) sx_sh[tid] = 8 * sum_x[tid];
    __syncthreads();

    float best_v[B]; uint32_t best_i[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { best_v[b] = -1e30f; best_i[b] = 0; }

    constexpr uint32_t MASK = 0x0F0F0F0Fu;
    const int stride = gridDim.x * 8;
    for (int row = blockIdx.x * 8 + warp; row < V; row += stride) {
        const uint4* wp = reinterpret_cast<const uint4*>(W4 + (size_t)row * (K >> 1));
        int acc[B];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = 0;
        for (int i = lane; i < k32; i += 32) {
            const uint4 wv = __ldg(wp + i);
            const uint32_t w0l = wv.x & MASK, w0h = (wv.x >> 4) & MASK;
            const uint32_t w1l = wv.y & MASK, w1h = (wv.y >> 4) & MASK;
            const uint32_t w2l = wv.z & MASK, w2h = (wv.z >> 4) & MASK;
            const uint32_t w3l = wv.w & MASK, w3h = (wv.w >> 4) & MASK;
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int4 x0 = xs_i4[b * k16 + 2 * i];
                const int4 x1 = xs_i4[b * k16 + 2 * i + 1];
                acc[b] = dp4a_su(x0.x, w0l, acc[b]);
                acc[b] = dp4a_su(x0.y, w0h, acc[b]);
                acc[b] = dp4a_su(x0.z, w1l, acc[b]);
                acc[b] = dp4a_su(x0.w, w1h, acc[b]);
                acc[b] = dp4a_su(x1.x, w2l, acc[b]);
                acc[b] = dp4a_su(x1.y, w2h, acc[b]);
                acc[b] = dp4a_su(x1.z, w3l, acc[b]);
                acc[b] = dp4a_su(x1.w, w3h, acc[b]);
            }
        }
        const float sw = scale_w[row];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int v = acc[b];
#pragma unroll
            for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0) {
                float logit = (float)(v - sx_sh[b]) * sw;
                if (logit > best_v[b]) { best_v[b] = logit; best_i[b] = (uint32_t)row; }
            }
        }
    }

    finalize_head_argmax<B>(best_v, best_i, vocab_ids, blk_val, blk_id, ticket, out_tok, red_v, red_i);
}

// Group-128 symmetric int4 (+8 nibble-packed) fused Head + Argmax + Vocab-ID remap!
// Q5 proved group-128 int4 preserves 100% d1 (12/12) and 94.4% d1..3 (34/36) match vs FP32,
// whereas per-row int4 dropped to 67% d1! Overhead: 1 fp16 scale per 128 weights (+3.1% VRAM).
// Warp-coalesced 128-byte loads: lanes 0..15 cover group 2g (128 weights), lanes 16..31 cover group 2g+1!
template <int B>
__global__ void __launch_bounds__(256, 6) head_argmax_i4g128_dp4a(
    const uint8_t* __restrict__ W4,       // [V, K/2] packed (+8 unsigned nibbles)
    const __half* __restrict__ scale_g,   // [V, K/128] fp16 per-128-group scale
    const int8_t* __restrict__ xq,        // [B, K]
    const int* __restrict__ sum_xg,       // [B, K/128] int32 group sums (8 * sum_xg subtracted per group)
    const uint32_t* __restrict__ vocab_ids,
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    int V, int K)
{
    extern __shared__ int32_t xs_i32[];   // [B * (K/4)] int32s
    __shared__ float red_v[B * 8];
    __shared__ uint32_t red_i[B * 8];
    __shared__ int sxg_sh[B * 32];        // up to 32 groups (K <= 4096)

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k4 = K >> 2;
    const int ng = K >> 7;                // K / 128 (must be even: 4 for 512, 8 for 1024, 18 for 2304)

    const int32_t* xg = reinterpret_cast<const int32_t*>(xq);
    const int tot_x = B * k4;
    for (int i = tid; i < tot_x; i += 256) xs_i32[i] = xg[i];
    for (int i = tid; i < B * ng; i += 256) sxg_sh[i] = 8 * sum_xg[i];
    __syncthreads();

    float best_v[B]; uint32_t best_i[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { best_v[b] = -1e30f; best_i[b] = 0; }

    constexpr uint32_t MASK = 0x0F0F0F0Fu;
    const int stride = gridDim.x * 8;
    const int half = lane >> 4;           // 0 for lanes 0..15, 1 for lanes 16..31
    const int sub  = lane & 15;           // 0..15 within half-warp (each lane = 8 weights = 2 int32s of x)

    for (int row = blockIdx.x * 8 + warp; row < V; row += stride) {
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(W4 + (size_t)row * (K >> 1));
        const __half* sgp = scale_g + (size_t)row * ng;
        float facc[B];
#pragma unroll
        for (int b = 0; b < B; ++b) facc[b] = 0.f;

        // Each iter processes 2 groups of 128 weights (256 weights = 32 uint32s = 128B coalesced!)
        for (int g = 0; g < ng; g += 2) {
            const uint32_t wv = __ldg(wp + g * 16 + lane);
            const uint32_t wl = wv & MASK;
            const uint32_t wh = (wv >> 4) & MASK;
            const int g_idx = g + half;
            const float sg = __half2float(__ldg(sgp + g_idx));
            const int x_base = g_idx * 32 + sub * 2;
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int x0 = xs_i32[b * k4 + x_base];
                const int x1 = xs_i32[b * k4 + x_base + 1];
                int dot = dp4a_su(x0, wl, 0);
                dot = dp4a_su(x1, wh, dot);
                // Subtract group offset once on sub==0 of each half-warp!
                if (sub == 0) dot -= sxg_sh[b * ng + g_idx];
                facc[b] = fmaf((float)dot, sg, facc[b]);
            }
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = facc[b];
#pragma unroll
            for (int s = 16; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0 && v > best_v[b]) { best_v[b] = v; best_i[b] = (uint32_t)row; }
        }
    }

    finalize_head_argmax<B>(best_v, best_i, vocab_ids, blk_val, blk_id, ticket, out_tok, red_v, red_i);
}

// ============================================================================
// 6. OPTION 2a: RESIDENT EXL3 UNEMBED (K=4.0, 200,058 x 2304) BATCHED B=1..4
//    M=16-Paired C0c exact FP32 kernel specialized for K=4.0 (nw=32 words/tile,
//    0%-bloat L1-rotated or inline K=4 aligned pairs) + fused svh + block argmax!
//    At K=4.0: each tile has 256 4-bit nibbles = 1024 bits = 32 uint32 words.
//    2-way in-block tk-split, 8 tiles/block, 256 threads/block (40 regs, 6 blk/SM).
// ============================================================================
static constexpr float KINV_F = 0.00676727294921875f;
static constexpr float KOFF_C = -0.0018157958984375f;

// Note: EXL3 K=4.0 unembed has shape tk=144 (K=2304), tn=12568 (N=201088, valid V=200058).
// For K=4.0, b0(i) = ((i+1)*4 - 16) mod 1024. After L1 rotation (or reading words w0,w1,w2):
// Each sub-lane (0..15) owns 16 consecutive windows: 2 pairs of (D, D+16) per 32-bit word!
// Exact K=4.0 offsets for 16 consecutive windows (j=0..15, stride 4 bits = 64 bits = 2 words):
//   j=0,4 -> W0 (0,16), j=1,5 -> shf(W1,W0,4), j=2,6 -> shf(W1,W0,8), j=3,7 -> shf(W1,W0,12)
//   j=8,12 -> W1 (0,16), j=9,13 -> shf(W2,W1,4), j=10,14 -> shf(W2,W1,8), j=11,15 -> shf(W2,W1,12)
// Rows (tile_perm): j=0..7 -> rows {0,1,8,9,2,3,10,11} (sub even -> 0..3/8..11, sub odd -> 4..7/12..15)
// Wait: sub in 0..15 covers 16*16=256 windows; tile_perm maps i = sub*16 + j:
//   tt = i >> 3 = 2*sub + (j >> 3); col_base = (tt >> 2) = sub >> 1;
//   j_lo = j & 7: r = ((2*sub + (j>>3)) & 3)*2 + (j&1) + ((j&2)?8:0); c = (sub>>1) + ((j&4)?8:0).
// So for fixed sub:
//   sub even: j=0..7 has (tt&3)=0 -> r in {0,1,8,9}, j=8..15 has (tt&3)=1 -> r in {2,3,10,11}
//   sub odd : j=0..7 has (tt&3)=2 -> r in {4,5,12,13}, j=8..15 has (tt&3)=3 -> r in {6,7,14,15}
// And in BOTH halves, j&4==0 -> col c0 = sub>>1, j&4==4 -> col c1 = (sub>>1) + 8!
// Because K=4, j and j+4 are 16 bits apart -> EXACT (D, D+16) halfword pairs in ONE word!!
// That means U.H1 = window j (col c0), U.H0 = window j+4 (col c1), BOTH using the SAME x[r]!!
template <int B>
__global__ void __launch_bounds__(256, 6) head_exl3_k4_batch_c0c(
    const uint32_t* __restrict__ trel,  // [tk, tn, 32] L1-rotated K=4.0 trellis
    const float* __restrict__ xgb,      // [tk*16, B] interleaved Hadamard-transformed x
    float* __restrict__ out_logits,     // [B, N] (written before H128/svh or directly timed)
    int tk, int tn, int k, int n, uint32_t CM)
{
    __shared__ __align__(16) float xs[32 * 16 * B];
    __shared__ float sx_sh[B];
    __shared__ float red[8];

    for (int b = 0; b < B; ++b) {
        float s = 0.f;
        for (int e = threadIdx.x; e < k; e += 256) s += xgb[e * B + b];
        float tot = block_reduce_sum(s, red);
        if (threadIdx.x == 0) sx_sh[b] = tot;
        __syncthreads();
    }

    const int part = threadIdx.x >> 7;
    const int local_tid = threadIdx.x & 127;
    const int t_in_blk = local_tid >> 4, sub = local_tid & 15;
    const int tn_i = blockIdx.x * 8 + t_in_blk;
    const int r_base = (sub & 1) ? 4 : 0;

    const size_t stride = (size_t)tn * 32;
    const int wbase = sub * 2;
    const uint32_t* p0 = trel + ((size_t)part * tn + tn_i) * 32 + wbase;
    const uint32_t* p1 = trel + ((size_t)part * tn + tn_i) * 32 + wbase + 1;
    const uint32_t* p2 = trel + ((size_t)part * tn + tn_i) * 32 + ((wbase + 2) & 31);
    const size_t step2 = 2 * stride;

    float a0[B], a1[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { a0[b] = 0.f; a1[b] = 0.f; }

    for (int tbase = 0; tbase < tk; tbase += 32) {
        const int nloc = min(32, tk - tbase);
        for (int e = threadIdx.x; e < nloc * 16 * B; e += 256) xs[e] = xgb[(size_t)tbase * 16 * B + e];
        __syncthreads();
        if (tn_i < tn) {
#pragma unroll 1
            for (int ti = part; ti < nloc; ti += 2, p0 += step2, p1 += step2, p2 += step2) {
                const uint32_t W0 = __ldg(p0), W1 = __ldg(p1), W2 = __ldg(p2);
                const float* xt = &xs[ti * 16 * B];
                auto do_pair = [&](uint32_t U, int r) {
                    float f0 = (float)(int)dp4a_uu(mul16_h1(U, CM), 0x01010101u, (uint32_t)(-510));
                    float f1 = (float)(int)dp4a_uu(mul16_h0(U, CM), 0x01010101u, (uint32_t)(-510));
                    if constexpr (B == 1) {
                        float x = xt[r]; a0[0] = fmaf(x, f0, a0[0]); a1[0] = fmaf(x, f1, a1[0]);
                    } else if constexpr (B == 2) {
                        float2 x = *(const float2*)&xt[r * 2];
                        a0[0] = fmaf(x.x, f0, a0[0]); a0[1] = fmaf(x.y, f0, a0[1]);
                        a1[0] = fmaf(x.x, f1, a1[0]); a1[1] = fmaf(x.y, f1, a1[1]);
                    } else if constexpr (B == 3) {
                        float x0 = xt[r * 3 + 0], x1 = xt[r * 3 + 1], x2 = xt[r * 3 + 2];
                        a0[0] = fmaf(x0, f0, a0[0]); a0[1] = fmaf(x1, f0, a0[1]); a0[2] = fmaf(x2, f0, a0[2]);
                        a1[0] = fmaf(x0, f1, a1[0]); a1[1] = fmaf(x1, f1, a1[1]); a1[2] = fmaf(x2, f1, a1[2]);
                    } else {
                        float4 x = *(const float4*)&xt[r * 4];
                        a0[0] = fmaf(x.x, f0, a0[0]); a0[1] = fmaf(x.y, f0, a0[1]); a0[2] = fmaf(x.z, f0, a0[2]); a0[3] = fmaf(x.w, f0, a0[3]);
                        a1[0] = fmaf(x.x, f1, a1[0]); a1[1] = fmaf(x.y, f1, a1[1]); a1[2] = fmaf(x.z, f1, a1[2]); a1[3] = fmaf(x.w, f1, a1[3]);
                    }
                };
                // First half (j=0..7): pairs (0,4),(1,5),(2,6),(3,7) -> rows r_base+{0,1,8,9}
                do_pair(W0,                         r_base + 0);
                do_pair(__funnelshift_l(W1, W0, 4), r_base + 1);
                do_pair(__funnelshift_l(W1, W0, 8), r_base + 8);
                do_pair(__funnelshift_l(W1, W0,12), r_base + 9);
                // Second half (j=8..15): pairs (8,12),(9,13),(10,14),(11,15) -> rows r_base+{2,3,10,11}
                do_pair(W1,                         r_base + 2);
                do_pair(__funnelshift_l(W2, W1, 4), r_base + 3);
                do_pair(__funnelshift_l(W2, W1, 8), r_base + 10);
                do_pair(__funnelshift_l(W2, W1,12), r_base + 11);
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
            float* dst = &out_logits[(size_t)b * n + tn_i * 16 + (sub >> 1)];
            dst[0] = fmaf(KINV_F, s0, off);
            dst[8] = fmaf(KINV_F, s1, off);
        }
    }
}

// Fast H128 + svh + argmax post-pass for batched EXL3 unembed (B rows x V=200058)
// 1 block per 128-element Hadamard group (or persistent 160 blocks with fused argmax!)
template <int B>
__global__ void __launch_bounds__(256, 6) exl3_had_out_argmax(
    const float* __restrict__ t,        // [B, N_pad]
    const uint16_t* __restrict__ svh,   // [N_pad] fp16 signs/scales
    float* __restrict__ blk_val,
    uint32_t* __restrict__ blk_id,
    uint32_t* __restrict__ ticket,
    uint32_t* __restrict__ out_tok,
    int V, int N_pad)
{
    __shared__ float hbuf[128];
    __shared__ float red_v[B * 8];
    __shared__ uint32_t red_i[B * 8];
    const int tid = threadIdx.x;
    const int ngrp = N_pad >> 7;
    const float scale = 0.08838834764831845f; // 1 / sqrt(128)

    float bv[B]; uint32_t bi[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { bv[b] = -1e30f; bi[b] = 0; }

    for (int grp = blockIdx.x; grp < ngrp; grp += gridDim.x) {
        const int base = grp << 7;
        float sv = 1.f;
        if (tid < 128) {
            // fp16 to f32
            uint16_t h = svh[base + tid];
            int e = (h >> 10) & 31, m = h & 1023;
            float v = e ? ldexpf(1.f + m * (1.f / 1024.f), e - 15) : ldexpf(m * (1.f / 1024.f), -14);
            sv = (h & 0x8000) ? -v : v;
        }
        for (int b = 0; b < B; ++b) {
            if (tid < 128) hbuf[tid] = t[(size_t)b * N_pad + base + tid];
            __syncthreads();
            // 7-stage H128 Walsh-Hadamard butterfly
            for (int st = 1; st < 128; st <<= 1) {
                float v0 = 0.f;
                if (tid < 128) {
                    int partner = tid ^ st;
                    float a = hbuf[tid], p = hbuf[partner];
                    v0 = (tid < partner) ? (a + p) : (p - a);
                }
                __syncthreads();
                if (tid < 128) hbuf[tid] = v0;
                __syncthreads();
            }
            if (tid < 128) {
                int idx = base + tid;
                if (idx < V) {
                    float logit = hbuf[tid] * scale * sv;
                    if (logit > bv[b]) { bv[b] = logit; bi[b] = (uint32_t)idx; }
                }
            }
            __syncthreads();
        }
    }

    // Reduce within block (threads 0..127 -> 4 warps)
    const int warp = tid >> 5;
    const int lane = tid & 31;
#pragma unroll
    for (int b = 0; b < B; ++b) {
        float v = (tid < 128) ? bv[b] : -1e30f;
        uint32_t id = (tid < 128) ? bi[b] : 0;
#pragma unroll
        for (int s = 16; s; s >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, v, s);
            uint32_t oi = __shfl_xor_sync(0xffffffffu, id, s);
            if (ov > v) { v = ov; id = oi; }
        }
        bv[b] = v; bi[b] = id;
    }
    finalize_head_argmax<B>(bv, bi, nullptr, blk_val, blk_id, ticket, out_tok, red_v, red_i);
}

// ============================================================================
// 7. SHORT-WINDOW ATTENTION FOR LIGHT DRAFTER BODY (SWA <= 128 tokens)
//    One block per (row b, head h), 128 threads/block (4 warps).
//    Fuses Q-norm + RoPE-like dot + softmax + weighted V sum!
// ============================================================================
__global__ void __launch_bounds__(128, 8) drafter_swa_attn(
    const float* __restrict__ q,    // [B, nh, hd]
    const float* __restrict__ kc,   // [win, nkv, hd]
    const float* __restrict__ vc,   // [win, nkv, hd]
    float* __restrict__ out,        // [B, nh, hd]
    int win, int nh, int nkv, int hd)
{
    __shared__ float scores[256];
    __shared__ float red4[4];
    const int b = blockIdx.y;
    const int h = blockIdx.x;
    const int kvh = h / (nh / nkv);
    const int tid = threadIdx.x;
    const float* qr = q + ((size_t)b * nh + h) * hd;
    const float scale = rsqrtf((float)hd);

    // Compute QK^T scores for t in [0..win)
    for (int t = tid; t < win; t += 128) {
        const float* kp = kc + ((size_t)t * nkv + kvh) * hd;
        float d = 0.f;
        for (int e = 0; e < hd; ++e) d += qr[e] * kp[e];
        scores[t] = d * scale;
    }
    __syncthreads();

    float mx = -1e30f;
    for (int t = tid; t < win; t += 128) mx = fmaxf(mx, scores[t]);
#pragma unroll
    for (int s = 16; s; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s));
    if ((tid & 31) == 0) red4[tid >> 5] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(red4[0], red4[1]), fmaxf(red4[2], red4[3]));

    float sm = 0.f;
    for (int t = tid; t < win; t += 128) {
        float e = __expf(scores[t] - mx);
        scores[t] = e;
        sm += e;
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, s);
    if ((tid & 31) == 0) red4[tid >> 5] = sm;
    __syncthreads();
    const float inv = 1.f / (red4[0] + red4[1] + red4[2] + red4[3] + 1e-12f);

    if (tid < hd) {
        float acc = 0.f;
        for (int t = 0; t < win; ++t) {
            acc += scores[t] * vc[((size_t)t * nkv + kvh) * hd + tid];
        }
        out[((size_t)b * nh + h) * hd + tid] = acc * inv;
    }
}

// Device-side token embedding lookup + tap fuse (reads device out_tok[0], adds target tap)
__global__ void __launch_bounds__(256, 6) drafter_embed_add(
    const int8_t* __restrict__ emb_i8,  // [V_emb, D]
    const float* __restrict__ emb_sc,   // [V_emb]
    const uint32_t* __restrict__ tok_d, // device ptr to previous drafted token id
    const float* __restrict__ tap_proj, // [D]
    float* __restrict__ out,            // [D]
    int D, int V_emb)
{
    uint32_t tok = tok_d[0];
    if (tok >= (uint32_t)V_emb) tok = 0;
    const float sc = emb_sc[tok];
    const int8_t* row = emb_i8 + (size_t)tok * D;
    for (int c = threadIdx.x; c < D; c += 256) {
        out[c] = (float)row[c] * sc + tap_proj[c];
    }
}

// ============================================================================
// 8. TWO-STAGE SHORTLIST + RESCORE HEAD (Stage 1: r=256 int8 -> top-M=64;
//    Stage 2: rescore top-M=64 rows from int8 full-dim head -> argmax)
// ============================================================================
template <int B, int TOPM>
__global__ void __launch_bounds__(256, 6) head_shortlist_stage1(
    const int8_t* __restrict__ W_lr,    // [V, R] (R=256)
    const float* __restrict__ sc_lr,    // [V]
    const int8_t* __restrict__ xq_lr,   // [B, R]
    float* __restrict__ cand_v,         // [B, G * TOPM_BLK] -> here G=64 blocks, 1 top/block -> 64 cands!
    uint32_t* __restrict__ cand_i,      // [B, G]
    int V, int R)
{
    __shared__ int4 xs_i4[B * 16]; // R=256 -> 16 int4s per row
    __shared__ float red_v[B * 8];
    __shared__ uint32_t red_i[B * 8];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k16 = R >> 4;
    const int4* xg = reinterpret_cast<const int4*>(xq_lr);
    for (int i = tid; i < B * k16; i += 256) xs_i4[i] = xg[i];
    __syncthreads();

    float bv[B]; uint32_t bi[B];
#pragma unroll
    for (int b = 0; b < B; ++b) { bv[b] = -1e30f; bi[b] = 0; }

    const int stride = gridDim.x * 8;
    for (int row = blockIdx.x * 8 + warp; row < V; row += stride) {
        const int4* wp = reinterpret_cast<const int4*>(W_lr + (size_t)row * R);
        int acc[B];
#pragma unroll
        for (int b = 0; b < B; ++b) acc[b] = 0;
        if (lane < k16) {
            const int4 wv = __ldg(wp + lane);
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int4 xv = xs_i4[b * k16 + lane];
                acc[b] = dp4a_ss(xv.x, wv.x, acc[b]);
                acc[b] = dp4a_ss(xv.y, wv.y, acc[b]);
                acc[b] = dp4a_ss(xv.z, wv.z, acc[b]);
                acc[b] = dp4a_ss(xv.w, wv.w, acc[b]);
            }
        }
        const float sw = sc_lr[row];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int v = acc[b];
#pragma unroll
            for (int s = 8; s; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0) {
                float logit = (float)v * sw;
                if (logit > bv[b]) { bv[b] = logit; bi[b] = (uint32_t)row; }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) { red_v[b * 8 + warp] = bv[b]; red_i[b * 8 + warp] = bi[b]; }
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float m = -1e30f; uint32_t mi = 0;
            for (int w = 0; w < 8; ++w) if (red_v[b * 8 + w] > m) { m = red_v[b * 8 + w]; mi = red_i[b * 8 + w]; }
            cand_v[b * gridDim.x + blockIdx.x] = m;
            cand_i[b * gridDim.x + blockIdx.x] = mi;
        }
    }
}

// Stage 2: rescore TOPM (e.g., 64) shortlisted rows from full [V, K] int8/int4 head -> argmax!
template <int B>
__global__ void __launch_bounds__(256, 4) head_shortlist_stage2_rescore(
    const int8_t* __restrict__ W_full,  // [V, K]
    const float* __restrict__ sc_full,  // [V]
    const int8_t* __restrict__ xq,      // [B, K]
    const uint32_t* __restrict__ cand_i,// [B, M]
    const uint32_t* __restrict__ vocab_ids,
    uint32_t* __restrict__ out_tok,     // [B]
    int M, int K)
{
    extern __shared__ int4 xs_i4[];
    __shared__ float red_v[8];
    __shared__ uint32_t red_i[8];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k16 = K >> 4;
    const int4* xg = reinterpret_cast<const int4*>(xq + (size_t)b * K);
    for (int i = tid; i < k16; i += 256) xs_i4[i] = xg[i];
    __syncthreads();

    float bv = -1e30f; uint32_t bi = 0;
    for (int m = warp; m < M; m += 8) {
        uint32_t row = cand_i[b * M + m];
        const int4* wp = reinterpret_cast<const int4*>(W_full + (size_t)row * K);
        int acc = 0;
        for (int i = lane; i < k16; i += 32) {
            const int4 wv = __ldg(wp + i);
            const int4 xv = xs_i4[i];
            acc = dp4a_ss(xv.x, wv.x, acc);
            acc = dp4a_ss(xv.y, wv.y, acc);
            acc = dp4a_ss(xv.z, wv.z, acc);
            acc = dp4a_ss(xv.w, wv.w, acc);
        }
#pragma unroll
        for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, s);
        if (lane == 0) {
            float logit = (float)acc * sc_full[row];
            if (logit > bv) { bv = logit; bi = row; }
        }
    }
    if (lane == 0) { red_v[warp] = bv; red_i[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float fm = -1e30f; uint32_t fi = 0;
        for (int w = 0; w < 8; ++w) if (red_v[w] > fm) { fm = red_v[w]; fi = red_i[w]; }
        out_tok[b] = vocab_ids ? vocab_ids[fi] : fi;
    }
}

} // namespace drf
