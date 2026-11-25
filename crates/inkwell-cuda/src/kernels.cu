// Elementwise + small-vector kernels for inkwell.
// Reference implementations — correctness first (golden-test path),
// fusion comes later per the autotuning ledger.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "exl3_gemv_v2.cuh"

extern "C" {

// rms_norm over rows: y[c] = x[c] * w[c] / sqrt(mean(x^2) + eps)
__global__ void inkwell_rmsnorm(const float* __restrict__ x,
                                const float* __restrict__ w,
                                float* __restrict__ y,
                                int n, float eps) {
    extern __shared__ float sm[];
    const int t = threadIdx.x;
    float acc = 0.f;
    for (int i = t; i < n; i += blockDim.x) { const float v = x[i]; acc += v * v; }
    sm[t] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) sm[t] += sm[t + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / n + eps);
    for (int i = t; i < n; i += blockDim.x) y[i] = x[i] * inv * (w ? w[i] : 1.f);
}

// clip to L2 norm <= threshold per row of `dim` (optionally per-channel scale)
__global__ void inkwell_clip(float* __restrict__ x,
                             const float* __restrict__ scale,
                             int dim, int n_rows, float threshold) {
    extern __shared__ float sm[];
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    float* row = x + (size_t)r * dim;
    const int t = threadIdx.x;
    if (scale) {
        for (int i = t; i < dim; i += blockDim.x) row[i] *= scale[i];
    }
    __syncthreads();
    float acc = 0.f;
    for (int i = t; i < dim; i += blockDim.x) acc += row[i] * row[i];
    sm[t] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) sm[t] += sm[t + s];
        __syncthreads();
    }
    const float norm = sqrtf(sm[0]);
    const float f = fminf(threshold / (norm + 1e-6f), 1.f);
    for (int i = t; i < dim; i += blockDim.x) row[i] *= f;
}

// y = a * sigmoid(b)
__global__ void inkwell_gate_mul(float* __restrict__ y,
                                 const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * (1.f / (1.f + expf(-b[i])));
}

// logits softcap: y = cap * tanh(x / cap)
__global__ void inkwell_softcap(float* __restrict__ x, int n, float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// SITU-ish elementwise: y = x * sigmoid(beta * x)  (silu family; beta=1 -> silu)
__global__ void inkwell_silu(float* __restrict__ x, int n, float beta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.f / (1.f + expf(-beta * x[i]));
}

// y = a + b
__global__ void inkwell_add(float* __restrict__ y,
                            const float* __restrict__ a,
                            const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

} // extern "C"

// host-side launchers (extern "C" entry points for FFI)
extern "C" {
void inkwell_rmsnorm_launch(const float* x, const float* w, float* y, int n, float eps) {
    const int blk = 256;
    inkwell_rmsnorm<<<1, blk, blk * 4>>>(x, w, y, n, eps);
}
void inkwell_clip_launch(float* x, const float* scale, int dim, int rows, float thr) {
    const int blk = 256;
    inkwell_clip<<<rows, blk, blk * 4>>>(x, scale, dim, rows, thr);
}
void inkwell_gate_mul_launch(float* y, const float* a, const float* b, int n) {
    inkwell_gate_mul<<<(n + 255) / 256, 256>>>(y, a, b, n);
}
void inkwell_softcap_launch(float* x, int n, float cap) {
    inkwell_softcap<<<(n + 255) / 256, 256>>>(x, n, cap);
}
void inkwell_silu_launch(float* x, int n, float beta) {
    inkwell_silu<<<(n + 255) / 256, 256>>>(x, n, beta);
}
void inkwell_add_launch(float* y, const float* a, const float* b, int n) {
    inkwell_add<<<(n + 255) / 256, 256>>>(y, a, b, n);
}
}

// ---------------- quantized dequant kernels (own port, golden-tested vs ggml) --
#include <cstdint>
#include "iq_luts.h"

struct block_iq2_xxs { uint16_t d; uint16_t qs[32]; }; // ggml_half = u16 bits

__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    unsigned int s = (h & 0x8000) << 16;
    unsigned int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    unsigned int f;
    if (e == 0) f = s | ((m == 0) ? 0u : (unsigned int)__float_as_uint((float)m * 5.9604644775390625e-8f));
    else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    return __uint_as_float(f);
}

__global__ void inkwell_dequant_iq2_xxs(const void* __restrict__ vx, float* __restrict__ y, int64_t nblocks) {
    const block_iq2_xxs* x = (const block_iq2_xxs*)vx;
    const int64_t ibs = blockIdx.x;
    const int tid = threadIdx.x;            // 32 threads per block
    const int il = tid / 8, ib = tid % 8;
    float* yy = y + ibs * 256;
    float* yl = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*)q2;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | ((uint32_t)q2[3] << 16);
    const float d = fp16_to_f32(x[ibs].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        yl[j] = d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
}

extern "C" void ink_launch_dequant_iq2_xxs(const void* vx, float* y, long long nblocks, void* stream) {
    inkwell_dequant_iq2_xxs<<<(unsigned)nblocks, 32, 0, (cudaStream_t)stream>>>(vx, y, nblocks);
}

// ---------------- fused dequant+dot GEMV (own) --------------------------------
// y[r] = sum_c dequant(W[r,c]) * x[c]; one warp per row.

struct block_q2_0 { uint16_t d; uint8_t qs[16]; }; // 64 vals, 18 bytes

__global__ void inkwell_mmvq_iq2_xxs(const void* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int64_t nrows, int64_t ncols) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;                    // 32 lanes
    const int nb_row = (int)(ncols / 256);           // blocks per row
    const block_iq2_xxs* w = (const block_iq2_xxs*)W + row * nb_row;
    const int ngroups = (int)(ncols / 8);            // 8-vals per group
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g / 32, gb = g % 32;           // block, group-in-block
        const int ib = gb / 4, il = gb % 4;
        const uint16_t* q2 = w[b].qs + 4 * ib;
        const uint8_t* aux8 = (const uint8_t*)q2;
        const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
        const uint32_t aux32 = q2[2] | ((uint32_t)q2[3] << 16);
        const float d = fp16_to_f32(w[b].d) * (0.5f + (aux32 >> 28)) * 0.25f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
        const float* xv = x + g * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f) * xv[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

__global__ void inkwell_mmvq_q2_0(const void* __restrict__ W,
                                  const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int64_t nrows, int64_t ncols) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int nb_row = (int)(ncols / 64);
    const block_q2_0* w = (const block_q2_0*)W + row * nb_row;
    const int nbytes = (int)(ncols / 4);             // 4 vals per byte
    float acc = 0.f;
    for (int bi = lane; bi < nbytes; bi += 32) {
        const int b = bi / 16;
        const uint8_t byte = w[b].qs[bi % 16];
        const float d = fp16_to_f32(w[b].d);
        const float* xv = x + bi * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc += d * (((byte >> (j * 2)) & 3) - 1) * xv[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

extern "C" void ink_launch_mmvq_iq2_xxs(const void* W, const float* x, float* y,
                                        long long nrows, long long ncols, void* stream) {
    inkwell_mmvq_iq2_xxs<<<(unsigned)nrows, 32, 0, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}
extern "C" void ink_launch_mmvq_q2_0(const void* W, const float* x, float* y,
                                     long long nrows, long long ncols, void* stream) {
    inkwell_mmvq_q2_0<<<(unsigned)nrows, 32, 0, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}

// ---------------- IQ1_S (gate_up experts' other format) ------------------------
struct block_iq1_s { uint16_t d; uint8_t qs[32]; uint16_t qh[8]; }; // 50B/256

#define IQ1S_DELTA 0.125f

__global__ void inkwell_dequant_iq1_s(const void* __restrict__ vx, float* __restrict__ y, int64_t nblocks) {
    const block_iq1_s* x = (const block_iq1_s*)vx;
    const int64_t ibs = blockIdx.x;
    const int tid = threadIdx.x;
    const int il = tid / 8, ib = tid % 8;
    float* yl = y + ibs * 256 + 32 * ib + 8 * il;
    const float delta = (x[ibs].qh[ib] & 0x8000) ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
    const float d = fp16_to_f32(x[ibs].d) * (2 * ((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) yl[j] = d * (q[j] + delta);
}

__global__ void inkwell_mmvq_iq1_s(const void* __restrict__ W,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int64_t nrows, int64_t ncols) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int nb_row = (int)(ncols / 256);
    const block_iq1_s* w = (const block_iq1_s*)W + row * nb_row;
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g / 32, gb = g % 32;
        const int ib = gb / 4, il = gb % 4;
        const float delta = (w[b].qh[ib] & 0x8000) ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
        const float d = fp16_to_f32(w[b].d) * (2 * ((w[b].qh[ib] >> 12) & 7) + 1);
        uint32_t grid32[2];
        const int8_t* q = (const int8_t*)grid32;
        grid32[0] = iq1s_grid_gpu[w[b].qs[4 * ib + il] | (((w[b].qh[ib] >> 3 * il) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        const float* xv = x + g * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += d * (q[j] + delta) * xv[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

extern "C" void ink_launch_dequant_iq1_s(const void* vx, float* y, long long nblocks, void* stream) {
    inkwell_dequant_iq1_s<<<(unsigned)nblocks, 32, 0, (cudaStream_t)stream>>>(vx, y, nblocks);
}
extern "C" void ink_launch_mmvq_iq1_s(const void* W, const float* x, float* y,
                                      long long nrows, long long ncols, void* stream) {
    inkwell_mmvq_iq1_s<<<(unsigned)nrows, 32, 0, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}

// ---------------- NVFP4 (down experts) ----------------------------------------
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; }; // 36B/64 vals

__device__ __forceinline__ float ue4m3_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.f;
    const int e = (x >> 3) & 0xF, m = x & 0x7;
    return (e == 0 ? ldexpf((float)m, -9) : ldexpf(1.0f + m / 8.f, e - 7)) / 2.f;
}

__device__ const float ue4m3_lut[256] = {0.000000000e+00f, 9.765625000e-04f, 1.953125000e-03f, 2.929687500e-03f, 3.906250000e-03f, 4.882812500e-03f, 5.859375000e-03f, 6.835937500e-03f, 7.812500000e-03f, 8.789062500e-03f, 9.765625000e-03f, 1.074218750e-02f, 1.171875000e-02f, 1.269531250e-02f, 1.367187500e-02f, 1.464843750e-02f, 1.562500000e-02f, 1.757812500e-02f, 1.953125000e-02f, 2.148437500e-02f, 2.343750000e-02f, 2.539062500e-02f, 2.734375000e-02f, 2.929687500e-02f, 3.125000000e-02f, 3.515625000e-02f, 3.906250000e-02f, 4.296875000e-02f, 4.687500000e-02f, 5.078125000e-02f, 5.468750000e-02f, 5.859375000e-02f, 6.250000000e-02f, 7.031250000e-02f, 7.812500000e-02f, 8.593750000e-02f, 9.375000000e-02f, 1.015625000e-01f, 1.093750000e-01f, 1.171875000e-01f, 1.250000000e-01f, 1.406250000e-01f, 1.562500000e-01f, 1.718750000e-01f, 1.875000000e-01f, 2.031250000e-01f, 2.187500000e-01f, 2.343750000e-01f, 2.500000000e-01f, 2.812500000e-01f, 3.125000000e-01f, 3.437500000e-01f, 3.750000000e-01f, 4.062500000e-01f, 4.375000000e-01f, 4.687500000e-01f, 5.000000000e-01f, 5.625000000e-01f, 6.250000000e-01f, 6.875000000e-01f, 7.500000000e-01f, 8.125000000e-01f, 8.750000000e-01f, 9.375000000e-01f, 1.000000000e+00f, 1.125000000e+00f, 1.250000000e+00f, 1.375000000e+00f, 1.500000000e+00f, 1.625000000e+00f, 1.750000000e+00f, 1.875000000e+00f, 2.000000000e+00f, 2.250000000e+00f, 2.500000000e+00f, 2.750000000e+00f, 3.000000000e+00f, 3.250000000e+00f, 3.500000000e+00f, 3.750000000e+00f, 4.000000000e+00f, 4.500000000e+00f, 5.000000000e+00f, 5.500000000e+00f, 6.000000000e+00f, 6.500000000e+00f, 7.000000000e+00f, 7.500000000e+00f, 8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f, 1.800000000e+01f, 2.000000000e+01f, 2.200000000e+01f, 2.400000000e+01f, 2.600000000e+01f, 2.800000000e+01f, 3.000000000e+01f, 3.200000000e+01f, 3.600000000e+01f, 4.000000000e+01f, 4.400000000e+01f, 4.800000000e+01f, 5.200000000e+01f, 5.600000000e+01f, 6.000000000e+01f, 6.400000000e+01f, 7.200000000e+01f, 8.000000000e+01f, 8.800000000e+01f, 9.600000000e+01f, 1.040000000e+02f, 1.120000000e+02f, 1.200000000e+02f, 1.280000000e+02f, 1.440000000e+02f, 1.600000000e+02f, 1.760000000e+02f, 1.920000000e+02f, 2.080000000e+02f, 2.240000000e+02f, 0.000000000e+00f, 0.000000000e+00f, 9.765625000e-04f, 1.953125000e-03f, 2.929687500e-03f, 3.906250000e-03f, 4.882812500e-03f, 5.859375000e-03f, 6.835937500e-03f, 7.812500000e-03f, 8.789062500e-03f, 9.765625000e-03f, 1.074218750e-02f, 1.171875000e-02f, 1.269531250e-02f, 1.367187500e-02f, 1.464843750e-02f, 1.562500000e-02f, 1.757812500e-02f, 1.953125000e-02f, 2.148437500e-02f, 2.343750000e-02f, 2.539062500e-02f, 2.734375000e-02f, 2.929687500e-02f, 3.125000000e-02f, 3.515625000e-02f, 3.906250000e-02f, 4.296875000e-02f, 4.687500000e-02f, 5.078125000e-02f, 5.468750000e-02f, 5.859375000e-02f, 6.250000000e-02f, 7.031250000e-02f, 7.812500000e-02f, 8.593750000e-02f, 9.375000000e-02f, 1.015625000e-01f, 1.093750000e-01f, 1.171875000e-01f, 1.250000000e-01f, 1.406250000e-01f, 1.562500000e-01f, 1.718750000e-01f, 1.875000000e-01f, 2.031250000e-01f, 2.187500000e-01f, 2.343750000e-01f, 2.500000000e-01f, 2.812500000e-01f, 3.125000000e-01f, 3.437500000e-01f, 3.750000000e-01f, 4.062500000e-01f, 4.375000000e-01f, 4.687500000e-01f, 5.000000000e-01f, 5.625000000e-01f, 6.250000000e-01f, 6.875000000e-01f, 7.500000000e-01f, 8.125000000e-01f, 8.750000000e-01f, 9.375000000e-01f, 1.000000000e+00f, 1.125000000e+00f, 1.250000000e+00f, 1.375000000e+00f, 1.500000000e+00f, 1.625000000e+00f, 1.750000000e+00f, 1.875000000e+00f, 2.000000000e+00f, 2.250000000e+00f, 2.500000000e+00f, 2.750000000e+00f, 3.000000000e+00f, 3.250000000e+00f, 3.500000000e+00f, 3.750000000e+00f, 4.000000000e+00f, 4.500000000e+00f, 5.000000000e+00f, 5.500000000e+00f, 6.000000000e+00f, 6.500000000e+00f, 7.000000000e+00f, 7.500000000e+00f, 8.000000000e+00f, 9.000000000e+00f, 1.000000000e+01f, 1.100000000e+01f, 1.200000000e+01f, 1.300000000e+01f, 1.400000000e+01f, 1.500000000e+01f, 1.600000000e+01f, 1.800000000e+01f, 2.000000000e+01f, 2.200000000e+01f, 2.400000000e+01f, 2.600000000e+01f, 2.800000000e+01f, 3.000000000e+01f, 3.200000000e+01f, 3.600000000e+01f, 4.000000000e+01f, 4.400000000e+01f, 4.800000000e+01f, 5.200000000e+01f, 5.600000000e+01f, 6.000000000e+01f, 6.400000000e+01f, 7.200000000e+01f, 8.000000000e+01f, 8.800000000e+01f, 9.600000000e+01f, 1.040000000e+02f, 1.120000000e+02f, 1.200000000e+02f, 1.280000000e+02f, 1.440000000e+02f, 1.600000000e+02f, 1.760000000e+02f, 1.920000000e+02f, 2.080000000e+02f, 2.240000000e+02f, 2.400000000e+02f};
__device__ __forceinline__ float ue4m3_f(uint8_t x) { return ue4m3_lut[x]; }


__global__ void inkwell_mmvq_nvfp4(const void* __restrict__ W,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int64_t nrows, int64_t ncols) {
    const int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    const int nb_row = (int)(ncols / 64);
    const block_nvfp4* w = (const block_nvfp4*)W + row * nb_row;
    const int npairs = (int)(ncols / 2);             // 2 vals per byte
    float acc = 0.f;
    for (int p = lane; p < npairs; p += 32) {
        const int b = p / 32, sub = (p % 32) / 8, j = p % 8;
        // byte covers vals [sub*16 + j, sub*16 + j + 8]
        const float d = ue4m3_to_f32(w[b].d[sub]);
        const uint8_t q = w[b].qs[sub * 8 + j];
        const float* xv = x + b * 64 + sub * 16 + j;
        acc += d * kvalues_fp4[q & 0x0F] * xv[0]
             + d * kvalues_fp4[q >> 4]  * xv[8];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

__global__ void inkwell_dequant_nvfp4(const void* __restrict__ vx, float* __restrict__ y, int64_t nblocks) {
    const block_nvfp4* x = (const block_nvfp4*)vx;
    const int64_t ibs = blockIdx.x;
    const int tid = threadIdx.x;                    // 32 threads
    const int sub = tid / 8, j = tid % 8;
    const float d = ue4m3_to_f32(x[ibs].d[sub]);
    const uint8_t q = x[ibs].qs[sub * 8 + j];
    float* yl = y + ibs * 64 + sub * 16 + j;
    yl[0] = d * kvalues_fp4[q & 0x0F];
    yl[8] = d * kvalues_fp4[q >> 4];
}

extern "C" void ink_launch_mmvq_nvfp4(const void* W, const float* x, float* y,
                                      long long nrows, long long ncols, void* stream) {
    inkwell_mmvq_nvfp4<<<(unsigned)nrows, 32, 0, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}
extern "C" void ink_launch_dequant_nvfp4(const void* vx, float* y, long long nblocks, void* stream) {
    inkwell_dequant_nvfp4<<<(unsigned)nblocks, 32, 0, (cudaStream_t)stream>>>(vx, y, nblocks);
}

// ---------------- batched expert GEMV (8 experts per launch) ------------------
// gate_up: w layout rows [ff=2*ne] per expert, y[2*ne] per expert.
// gridDim.x = rows, gridDim.y = expert slot; w += slot*row_bytes via ids.

__device__ __forceinline__ float unit_iq2xxs(const block_iq2_xxs* w, const float* __restrict__ xs, int g, int NU) {
    const int b = g / 32, gb = g % 32, ib = gb / 4, il = gb % 4;
    const uint16_t* q2 = w[b].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*)q2;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | ((uint32_t)q2[3] << 16);
    const float d = fp16_to_f32(w[b].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        a += d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f) * xs[j * NU + g];
    return a;
}

// Two warps per row: halves each warp's dependent-load chain (latency-bound).
// Partials merge via atomicAdd — caller must zero y first.
__global__ void inkwell_mmvq_iq2_xxs_moe(const void* const* __restrict__ Wp,
                                         const float* __restrict__ x,
                                         float* __restrict__ y,       // [nslots][nrows]
                                         int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    const int NU = (int)(ncols / 8) + 1;             // padded stride, kills bank conflicts
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[(i & 7) * NU + (i >> 3)] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_iq2_xxs* w = (const block_iq2_xxs*)((const uint8_t*)Wp[slot])
                             + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
#pragma unroll 2
    for (int g = lane + half * 32; g < ngroups; g += 64) acc += unit_iq2xxs(w, xs, g, NU);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}

__global__ void inkwell_mmvq_q2_0_moe(const void* const* __restrict__ Wp,
                                      const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_q2_0* w = (const block_q2_0*)((const uint8_t*)Wp[slot])
                          + row * (ncols / 64);
    const int nunits = (int)(ncols / 16);          // 16 vals = 4 qs bytes
    float acc = 0.f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u >> 2, ub = u & 3;           // 4 units per 64-val block
        // NOTE: qs sits at byte offset 2 in the 18-byte block — a u32 load here
        // is misaligned and faults on Pascal. Keep two aligned u16 reads.
        const uint32_t q = (uint32_t)*(const uint16_t*)&w[b].qs[ub * 4]
                         | ((uint32_t)*(const uint16_t*)&w[b].qs[ub * 4 + 2] << 16);
        const float d = fp16_to_f32(w[b].d);
        const float* xv = xs + u * 16;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            acc += d * ((int)((q >> (8 * (j >> 2) + 2 * (j & 3))) & 3) - 1) * xv[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}

// gate_up fusion: rows ncols..2*ncols hold "up", swiglu(gate)*up -> [ncols]
// wait — gate_up tensor rows are [gate;up] halves of ncols each: out = silu(gate)*up
__global__ void inkwell_swiglu_pair(float* __restrict__ y, const float* __restrict__ gu,
                                    int64_t nslots, int64_t ncols) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nslots * ncols) return;
    const int64_t s = i / ncols, c = i % ncols;
    const float g = gu[s * 2 * ncols + c];
    const float u = gu[s * 2 * ncols + ncols + c];
    y[i] = (g / (1.f + expf(-g))) * u;
}

extern "C" void ink_launch_mmvq_iq2_xxs_moe(const void* const* Wp,
                                            const float* x, float* y,
                                            long long nrows, long long ncols, long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_iq2_xxs_moe<<<g, 512, ((size_t)ncols + 8) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}
extern "C" void ink_launch_mmvq_q2_0_moe(const void* const* Wp,
                                         const float* x, float* y,
                                         long long nrows, long long ncols, long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16), (unsigned)nslots);
    inkwell_mmvq_q2_0_moe<<<g, 512, (size_t)ncols * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}
extern "C" void ink_launch_swiglu_pair(float* y, const float* gu, long long nslots, long long ncols, void* stream) {
    int64_t n = nslots * ncols;
    inkwell_swiglu_pair<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(y, gu, nslots, ncols);
}

// iq1_s batched: {u16 d; u8 qs[32]; u16 qh[8]}, 50B/256
__global__ void inkwell_mmvq_iq1_s_moe(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                       float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_iq1_s* w = (const block_iq1_s*)((const uint8_t*)Wp[slot])
                           + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane + half * 32; g < ngroups; g += 64) {
        const int b = g / 32, gb = g % 32;
        const int ib = gb / 4, il = gb % 4;
        const float delta = (w[b].qh[ib] & 0x8000) ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
        const float d = fp16_to_f32(w[b].d) * (2 * ((w[b].qh[ib] >> 12) & 7) + 1);
        uint32_t grid32[2];
        const int8_t* q = (const int8_t*)grid32;
        grid32[0] = iq1s_grid_gpu[w[b].qs[4 * ib + il] | (((w[b].qh[ib] >> 3 * il) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        const float4 xv0 = *(const float4*)(xs + g * 8);
        const float4 xv1 = *(const float4*)(xs + g * 8 + 4);
        acc += d * (q[0] + delta) * xv0.x + d * (q[1] + delta) * xv0.y
             + d * (q[2] + delta) * xv0.z + d * (q[3] + delta) * xv0.w
             + d * (q[4] + delta) * xv1.x + d * (q[5] + delta) * xv1.y
             + d * (q[6] + delta) * xv1.z + d * (q[7] + delta) * xv1.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}

// nvfp4 batched (36B/64): {u8 d[4] UE4M3; u8 qs[32] E2M1}
__global__ void inkwell_mmvq_nvfp4_moe(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                       float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_nvfp4* w = (const block_nvfp4*)((const uint8_t*)Wp[slot])
                           + row * (ncols / 64);
    const int ngroups = (int)(ncols / 16);
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g >> 2, sub = g & 3;
        const float d = ue4m3_f(w[b].d[sub]);
        const uint32_t qlo = *(const uint32_t*)&w[b].qs[sub * 8];
        const uint32_t qhi = *(const uint32_t*)&w[b].qs[sub * 8 + 4];
        const float4 xv0 = *(const float4*)(xs + g * 16);
        const float4 xv1 = *(const float4*)(xs + g * 16 + 4);
        const float4 xv2 = *(const float4*)(xs + g * 16 + 8);
        const float4 xv3 = *(const float4*)(xs + g * 16 + 12);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t q = (qlo >> (8 * j)) & 0xFF;
            acc += d * kvalues_fp4[q & 0x0F] * (&xv0.x)[j]
                 + d * kvalues_fp4[q >> 4]   * (&xv2.x)[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t q = (qhi >> (8 * j)) & 0xFF;
            acc += d * kvalues_fp4[q & 0x0F] * (&xv1.x)[j]
                 + d * kvalues_fp4[q >> 4]   * (&xv3.x)[j];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}

// SITU activation for gate_up pairs with per-expert scale:
// gu[s] rows are [gate(0..ff) ; up(ff..2ff)], scaled by sgu[s].
// hid = (beta*tanh(g/beta))*sigmoid(g) * (lin_beta*tanh(u/lin_beta))
__global__ void inkwell_situ_pair(const float* __restrict__ gu, const float* __restrict__ sgu,
                                  float* __restrict__ hid,
                                  float situ_beta, float situ_lin_beta, int64_t ncols, int64_t nslots) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nslots * ncols) return;
    const int64_t s = i / ncols, c = i % ncols;
    const float sc = sgu ? sgu[s] : 1.f;
    const float g = gu[s * 2 * ncols + c] * sc;
    const float u = gu[s * 2 * ncols + ncols + c] * sc;
    const float a = (situ_beta * tanhf(g / situ_beta)) / (1.f + expf(-g));
    const float b = situ_lin_beta * tanhf(u / situ_lin_beta);
    hid[i] = a * b;
}

// weighted sum of expert outputs: out[c] = sum_s w[s] * sd[s] * dn[s*ncols+c]
__global__ void inkwell_expert_wsum(const float* __restrict__ dn, const float* __restrict__ w,
                                    const float* __restrict__ sd,
                                    float* __restrict__ out, int64_t ncols, int64_t nslots) {
    const int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncols) return;
    float acc = 0.f;
    for (int64_t s = 0; s < nslots; ++s)
        acc += w[s] * (sd ? sd[s] : 1.f) * dn[s * ncols + c];
    out[c] = acc;
}

extern "C" void ink_launch_mmvq_iq1_s_moe(const void* const* Wp,
                                          const float* x, float* y, long long nrows, long long ncols,
                                          long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_iq1_s_moe<<<g, 512, (size_t)ncols * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}
extern "C" void ink_launch_mmvq_nvfp4_moe(const void* const* Wp,
                                          const float* x, float* y, long long nrows, long long ncols,
                                          long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16), (unsigned)nslots);
    inkwell_mmvq_nvfp4_moe<<<g, 512, (size_t)ncols * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// NVFP4 via DP4A: kvalues_fp4 is already int8 (E2M1 x2, /2 folded into the ue4m3
// scale). x is quantized to int8 per 64-val block; each 16-val group becomes
// 4 int8x4 packs -> __dp4a, acc in int32, scaled once per group.
__global__ void inkwell_mmvq_nvfp4_moe_i8(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                        float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    x += slot * x_stride;
    const int nblk = (int)(ncols / 64);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 64; i += 16) m = fmaxf(m, fabsf(x[b * 64 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 6];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_nvfp4* w = (const block_nvfp4*)((const uint8_t*)Wp[slot])
                           + row * (ncols / 64);
    const int ngroups = (int)(ncols / 16);
    float acc = 0.f;
    for (int g = lane + half * 32; g < ngroups; g += 64) {
        const int b = g >> 2, sub = g & 3;
        const float dsx = ue4m3_f(w[b].d[sub]) * sx[b];
        const uint32_t qlo = *(const uint32_t*)&w[b].qs[sub * 8];
        const uint32_t qhi = *(const uint32_t*)&w[b].qs[sub * 8 + 4];
        const uint32_t lo_l = qlo & 0x0F0F0F0Fu, hi_l = qhi & 0x0F0F0F0Fu;
        const uint32_t lo_h = (qlo >> 4) & 0x0F0F0F0Fu, hi_h = (qhi >> 4) & 0x0F0F0F0Fu;
        const uint32_t* xp = (const uint32_t*)(qx + g * 16);
        int isum = 0;
        const uint32_t masks[4] = {lo_l, hi_l, lo_h, hi_h}; // vals 0-3, 4-7, 8-11, 12-15
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t q = masks[p];
            uint32_t wp = (uint8_t)kvalues_fp4[q & 0xFF]
                        | ((uint32_t)(uint8_t)kvalues_fp4[(q >> 8) & 0xFF] << 8)
                        | ((uint32_t)(uint8_t)kvalues_fp4[(q >> 16) & 0xFF] << 16)
                        | ((uint32_t)(uint8_t)kvalues_fp4[q >> 24] << 24);
            isum = __dp4a((int)xp[p], (int)wp, isum);
        }
        acc += dsx * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}
extern "C" void ink_launch_mmvq_nvfp4_moe_i8(const void* const* Wp,
                                             const float* x, float* y, long long nrows, long long ncols,
                                             long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_nvfp4_moe_i8<<<g, 512, (size_t)ncols + (size_t)(ncols / 64) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// Q2_0 via DP4A: code -> {-1,0,1,2} fits int8; byte j packs vals 4j..4j+3 in
// natural order. Decode each qs byte into 4 int8 lanes with shifts (4 ops)
// then __dp4a. x quantized to int8 per 64-val block like the nvfp4 path.
__device__ __forceinline__ uint32_t q20_byte_to_i8x4(uint32_t b) {
    uint32_t v = 0;
    v |= (uint32_t)(uint8_t)((int)(b & 3) - 1);
    v |= (uint32_t)(uint8_t)((int)((b >> 2) & 3) - 1) << 8;
    v |= (uint32_t)(uint8_t)((int)((b >> 4) & 3) - 1) << 16;
    v |= (uint32_t)(uint8_t)((int)((b >> 6) & 3) - 1) << 24;
    return v;
}
__global__ void inkwell_mmvq_q2_0_moe_i8(const void* const* __restrict__ Wp,
                                         const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    x += slot * x_stride;
    const int nblk = (int)(ncols / 64);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 64; i += 16) m = fmaxf(m, fabsf(x[b * 64 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 6];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_q2_0* w = (const block_q2_0*)((const uint8_t*)Wp[slot])
                          + row * (ncols / 64);
    const int nunits = (int)(ncols / 16);
    float acc = 0.f;
#pragma unroll 1
    for (int u = lane + half * 32; u < nunits; u += 64) {
        const int b = u >> 2, ub = u & 3;
        const uint32_t q = (uint32_t)*(const uint16_t*)&w[b].qs[ub * 4]
                         | ((uint32_t)*(const uint16_t*)&w[b].qs[ub * 4 + 2] << 16);
        const float dsx = fp16_to_f32(w[b].d) * sx[b];
        const uint32_t* xp = (const uint32_t*)(qx + u * 16);
        int isum = __dp4a((int)xp[0], (int)q20_byte_to_i8x4(q & 0xFF),
                   __dp4a((int)xp[1], (int)q20_byte_to_i8x4((q >> 8) & 0xFF),
                   __dp4a((int)xp[2], (int)q20_byte_to_i8x4((q >> 16) & 0xFF),
                   __dp4a((int)xp[3], (int)q20_byte_to_i8x4(q >> 24), 0))));
        acc += dsx * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}
extern "C" void ink_launch_mmvq_q2_0_moe_i8(const void* const* Wp,
                                            const float* x, float* y, long long nrows, long long ncols,
                                            long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_q2_0_moe_i8<<<g, 512, (size_t)ncols + (size_t)(ncols / 64) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// IQ1_S via DP4A: the grid lookup already yields the 8 vals as int8 bytes in
// two u32s -> dp4a weight packs directly. The delta offset is not integer,
// so it is applied as delta * sum(x_int8) via dp4a(x, 0x01010101).
__global__ void inkwell_mmvq_iq1_s_moe_i8(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                        float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    x += slot * x_stride;
    const int nblk = (int)(ncols / 64);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 64; i += 16) m = fmaxf(m, fabsf(x[b * 64 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 6];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_iq1_s* w = (const block_iq1_s*)((const uint8_t*)Wp[slot])
                           + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g / 32, gb = g % 32;
        const int ib = gb / 4, il = gb % 4;
        const float delta = (w[b].qh[ib] & 0x8000) ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
        const float dsx = fp16_to_f32(w[b].d) * (2 * ((w[b].qh[ib] >> 12) & 7) + 1) * sx[g >> 3];
        const uint32_t grid = iq1s_grid_gpu[w[b].qs[4 * ib + il] | (((w[b].qh[ib] >> 3 * il) & 7) << 8)];
        const uint32_t w0 = grid & 0x0f0f0f0fu;
        const uint32_t w1 = (grid >> 4) & 0x0f0f0f0fu;
        const uint32_t* xp = (const uint32_t*)(qx + g * 8);
        int isum = __dp4a((int)xp[0], (int)w0, __dp4a((int)xp[1], (int)w1, 0));
        const int sumx = __dp4a((int)xp[0], 0x01010101, __dp4a((int)xp[1], 0x01010101, 0));
        acc += dsx * ((float)isum + delta * (float)sumx);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}
extern "C" void ink_launch_mmvq_iq1_s_moe_i8(const void* const* Wp,
                                             const float* x, float* y, long long nrows, long long ncols,
                                             long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16), (unsigned)nslots);
    inkwell_mmvq_iq1_s_moe_i8<<<g, 512, (size_t)ncols + (size_t)(ncols / 64) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// IQ2_XXS via DP4A: grid gives magnitudes (u8), signs is a separate byte mask.
// Expand the sign byte to per-byte 0xFF/0x00 masks, negate lanes via
// __vsubss4 and blend -> signed int8 packs for __dp4a.
__global__ void inkwell_mmvq_iq2_xxs_moe_i8(const void* const* __restrict__ Wp,
                                          const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    __shared__ uint64_t smask[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        uint64_t m = 0;
        for (int b = 0; b < 8; ++b) if (i & (1 << b)) m |= 0xFFull << (8 * b);
        smask[i] = m;
    }
    x += slot * x_stride;
    const int nblk = (int)(ncols / 32);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 32; i += 16) m = fmaxf(m, fabsf(x[b * 32 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 5];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_iq2_xxs* w = (const block_iq2_xxs*)((const uint8_t*)Wp[slot])
                             + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g / 32, gb = g % 32, ib = gb / 4, il = gb % 4;
        const uint16_t* q2 = w[b].qs + 4 * ib;
        const uint8_t* aux8 = (const uint8_t*)q2;
        const uint64_t grid = iq2xxs_grid[aux8[il]];
        const uint32_t aux32 = q2[2] | ((uint32_t)q2[3] << 16);
        const float dsx = fp16_to_f32(w[b].d) * (0.5f + (aux32 >> 28)) * 0.25f * sx[g >> 2];
        const uint64_t msk = smask[ksigns_iq2xs[(aux32 >> 7 * il) & 127]];
        const uint32_t g0 = (uint32_t)grid, g1 = (uint32_t)(grid >> 32);
        const uint32_t m0 = (uint32_t)msk, m1 = (uint32_t)(msk >> 32);
        const uint32_t w0 = (g0 & ~m0) | (__vsubss4(0u, g0) & m0);
        const uint32_t w1 = (g1 & ~m1) | (__vsubss4(0u, g1) & m1);
        const uint32_t* xp = (const uint32_t*)(qx + g * 8);
        int isum = __dp4a((int)xp[0], (int)w0, __dp4a((int)xp[1], (int)w1, 0));
        acc += dsx * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}
extern "C" void ink_launch_mmvq_iq2_xxs_moe_i8(const void* const* Wp,
                                               const float* x, float* y, long long nrows, long long ncols,
                                               long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16), (unsigned)nslots);
    inkwell_mmvq_iq2_xxs_moe_i8<<<g, 1024, (size_t)ncols + (size_t)(ncols / 32) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}
extern "C" void ink_launch_situ_pair(const float* gu, const float* sgu, float* hid,
                                     float beta, float lin_beta, long long ncols, long long nslots, void* stream) {
    int64_t n = nslots * ncols;
    inkwell_situ_pair<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(gu, sgu, hid, beta, lin_beta, ncols, nslots);
}
extern "C" void ink_launch_expert_wsum(const float* dn, const float* w, const float* sd,
                                       float* out, long long ncols, long long nslots, void* stream) {
    inkwell_expert_wsum<<<(unsigned)((ncols + 255) / 256), 256, 0, (cudaStream_t)stream>>>(dn, w, sd, out, ncols, nslots);
}


// ---------------- IQ2_XS (74B/256): {u16 d; u16 qs[32]; u8 scales[8]} ---------
struct block_iq2_xs { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; };

__global__ void inkwell_mmvq_iq2_xs(const void* __restrict__ W, const float* __restrict__ x,
                                    float* __restrict__ y, int64_t nrows, int64_t ncols) {
    const int64_t row = blockIdx.x;
    const block_iq2_xs* w = (const block_iq2_xs*)W + row * (ncols / 256);
    const int lane = threadIdx.x;    const int nvals = (int)ncols;
    float acc = 0.f;
    for (int p = lane; p < nvals; p += 32) {
        const int b = p >> 8, g8 = (p >> 3) & 31, ib32 = g8 >> 2, l = g8 & 3; // 8-val unit
        const block_iq2_xs* wb = w + b;
        const uint16_t qv = wb->qs[g8];
        const int nib = l >> 1;
        const float d = fp16_to_f32(wb->d) * (0.5f + ((wb->scales[ib32] >> (4 * nib)) & 0xF)) * 0.25f;
        const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (qv & 511));
        const uint8_t signs = ksigns_iq2xs[qv >> 9];
        acc += d * grid[p & 7] * ((signs & kmask_iq2xs[p & 7]) ? -1.f : 1.f) * x[p];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}

__global__ void inkwell_mmvq_iq2_xs_moe(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                        float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_iq2_xs* w0 = (const block_iq2_xs*)((const uint8_t*)Wp[slot]);
    const block_iq2_xs* w = w0 + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane + half * 32; g < ngroups; g += 64) {
        const int b = g >> 5, g8 = g & 31, ib32 = g8 >> 2, l = g8 & 3;
        const uint16_t qv = w[b].qs[g8];
        const float d = fp16_to_f32(w[b].d) * (0.5f + ((w[b].scales[ib32] >> (4 * (l >> 1))) & 0xF)) * 0.25f;
        const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (qv & 511));
        const uint8_t signs = ksigns_iq2xs[qv >> 9];
        const float* xv = xs + g * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc += d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f) * xv[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}

// ---------------- IQ4_XS (136B/256): {u16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]}
struct block_iq4_xs { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };

__global__ void inkwell_mmvq_iq4_xs_moe(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                        float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_iq4_xs* w = (const block_iq4_xs*)((const uint8_t*)Wp[slot])
                            + row * (ncols / 256);
    const int nunits = (int)(ncols / 16);          // 16 vals = 16 qs bytes
    float acc = 0.f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u >> 4, ub = u & 15;          // 16 units per 256-val block
        const int ib = ub >> 1, half = ub & 1;      // unit covers half of a 32-val ib
        const uint2 qa = *(const uint2*)&w[b].qs[ib * 16];
        const uint2 qb = *(const uint2*)&w[b].qs[ib * 16 + 8];
        const int ls = ((w[b].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((w[b].scales_h >> (2 * ib)) & 3) << 4);
        const float dl = fp16_to_f32(w[b].d) * (ls - 32);
        const float4 xv[4] = {
            *(const float4*)(xs + u * 16),      *(const float4*)(xs + u * 16 + 4),
            *(const float4*)(xs + u * 16 + 8),  *(const float4*)(xs + u * 16 + 12)};
        const unsigned long long qs64[2] = {
            (unsigned long long)qa.x | ((unsigned long long)qa.y << 32),
            (unsigned long long)qb.x | ((unsigned long long)qb.y << 32)};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int byte = (int)((qs64[j >> 3] >> (8 * (j & 7))) & 0xFF);
            acc += dl * (float)kvalues_iq4nl[half ? (byte >> 4) : (byte & 0xF)] * (&xv[j >> 2].x)[j & 3];
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}

extern "C" void ink_launch_mmvq_iq2_xs(const void* W, const float* x, float* y,
                                       long long nrows, long long ncols, void* stream) {
    inkwell_mmvq_iq2_xs<<<(unsigned)nrows, 32, 0, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}
extern "C" void ink_launch_mmvq_iq2_xs_moe(const void* const* Wp,
                                           const float* x, float* y, long long nrows, long long ncols,
                                           long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_iq2_xs_moe<<<g, 512, (size_t)ncols * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}
extern "C" void ink_launch_mmvq_iq4_xs_moe(const void* const* Wp,
                                           const float* x, float* y, long long nrows, long long ncols,
                                           long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16), (unsigned)nslots);
    inkwell_mmvq_iq4_xs_moe<<<g, 1024, (size_t)ncols * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// ---------------- IQ2_S (82B/256): {u16 d; u8 qs[64]; u8 qh[8]; u8 scales[8]} --
// qs[0..32]=grid idx, qs[32..64]=sign bytes; grid idx |= qh[ib32]<<(8-2l)&0x300
struct block_iq2_s { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; };

__global__ void inkwell_mmvq_iq2_s_moe(const void* const* __restrict__ Wp, const float* __restrict__ x,
                                       float* __restrict__ y, int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ float xs[];
    x += slot * x_stride;
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_iq2_s* w = (const block_iq2_s*)((const uint8_t*)Wp[slot])
                           + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane + half * 32; g < ngroups; g += 64) {
        const int b = g >> 5, ib32 = (g & 31) >> 2, l = g & 3;
        const float d = fp16_to_f32(w[b].d)
            * (0.5f + ((w[b].scales[ib32] >> (4 * (l >> 1))) & 0xF)) * 0.25f;
        const int gidx = w[b].qs[4 * ib32 + l] | ((w[b].qh[ib32] << (8 - 2 * l)) & 0x300);
        const uint8_t* grid = (const uint8_t*)(iq2s_grid + gidx);
        const uint8_t signs = w[b].qs[32 + 4 * ib32 + l];
        const float4 xv0 = *(const float4*)(xs + g * 8);
        const float4 xv1 = *(const float4*)(xs + g * 8 + 4);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc += d * grid[j]     * ((signs & kmask_iq2xs[j])     ? -1.f : 1.f) * (&xv0.x)[j]
               + d * grid[j + 4] * ((signs & kmask_iq2xs[j + 4]) ? -1.f : 1.f) * (&xv1.x)[j];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}

extern "C" void ink_launch_mmvq_iq2_s_moe(const void* const* Wp,
                                          const float* x, float* y, long long nrows, long long ncols,
                                          long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_iq2_s_moe<<<g, 512, (size_t)ncols * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// ===== device-side MoE dispatch: resolve + alloc + gather =====
// resolve: ids[s] -> ptrs[s] via device map; misses queued.
// alloc: 1 warp, serial victim pick (LRU), updates map/rev/lru + ptrs + work.
// gather: one block per miss, zero-copy u32 read from host -> vram slot.
// All stream-ordered: resolve -> alloc -> gather -> GEMVs read VRAM.

extern "C" __global__ void inkwell_moe_resolve(const int* ids, const int* map,
    unsigned long long* lru, const unsigned long long* stamp,
    unsigned long long arena, unsigned long long host, long long esz,
    unsigned long long* ptrs, int* missq, int nslots) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    const int e = ids[s];
    const int m = map ? map[e] : -1;
    ptrs[s] = m >= 0 ? arena + (unsigned long long)m * esz
                     : host + (unsigned long long)e * esz;
    if (m >= 0 && lru && stamp) lru[m] = *stamp;  // touch: selected experts are hot
    if (m < 0 && map) {
        const int q = atomicAdd(missq, 1);
        if (q < 8) { missq[2 + 2*q] = s; missq[2 + 2*q + 1] = e; }
    }
}

extern "C" __global__ void inkwell_moe_alloc(const int* missq, int* map,
    unsigned long long* lru, unsigned long long* stamp, int* rev,
    unsigned long long arena, unsigned long long host, long long esz,
    unsigned long long* ptrs, unsigned long long* work, int k) {
    int n = missq[0];
    if (n > 8) n = 8;
    const int lane = threadIdx.x & 31;
    for (int i = 0; i < n; ++i) {
        const int s = missq[2 + 2*i];
        const int e = missq[2 + 2*i + 1];
        unsigned long long best = ~0ull;
        int victim = 0;
        for (int j = lane; j < k; j += 32) {
            if (lru[j] < best) { best = lru[j]; victim = j; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const unsigned long long ob = __shfl_down_sync(0xffffffffu, best, o);
            const int ov = __shfl_down_sync(0xffffffffu, victim, o);
            if (ob < best) { best = ob; victim = ov; }
        }
        victim = __shfl_sync(0xffffffffu, victim, 0);
        if (lane == 0) {
            const int old = rev[victim];
            if (old >= 0) map[old] = -1;
            map[e] = victim;
            rev[victim] = e;
            *stamp += 1;
            lru[victim] = *stamp;
            ptrs[s] = arena + (unsigned long long)victim * esz;
            work[2*i]   = host + (unsigned long long)e * esz;
            work[2*i+1] = arena + (unsigned long long)victim * esz;
        }
        __syncwarp();
    }
}

extern "C" __global__ void inkwell_moe_gather(const int* missq,
    const unsigned long long* work, long long esz, unsigned long long* stats) {
    const int i = blockIdx.x;
    if (i >= missq[0] || i >= 8) return;
    const unsigned char* src = (const unsigned char*)(uintptr_t)work[2*i];
    unsigned char* dst = (unsigned char*)(uintptr_t)work[2*i+1];
    // 16B fast path when both ends are aligned (uv4), u32 mid, byte tail.
    if ((((uintptr_t)src | (uintptr_t)dst) & 15) == 0) {
        const long long nv = esz >> 4;
        const uint4* s4 = (const uint4*)src;
        uint4* d4 = (uint4*)dst;
        for (long long j = threadIdx.x; j < nv; j += blockDim.x) d4[j] = s4[j];
        const long long rem = esz & 15;
        if (rem) {
            const unsigned int* s32 = (const unsigned int*)(src + (nv << 4));
            unsigned int* d32 = (unsigned int*)(dst + (nv << 4));
            for (long long j = threadIdx.x; j < (rem >> 2); j += blockDim.x) d32[j] = s32[j];
            if (threadIdx.x == 0) for (long long b = rem & ~3ll; b < rem; ++b) dst[(nv << 4) + b] = src[(nv << 4) + b];
        }
    } else {
        const long long nu = esz >> 2;
        const unsigned int* s32 = (const unsigned int*)src;
        unsigned int* d32 = (unsigned int*)dst;
        for (long long j = threadIdx.x; j < nu; j += blockDim.x) d32[j] = s32[j];
        if (threadIdx.x == 0) for (long long b = nu << 2; b < esz; ++b) dst[b] = src[b];
    }
}

extern "C" void ink_launch_moe_resolve(const int* ids, const int* map,
    unsigned long long* lru, const unsigned long long* stamp,
    unsigned long long arena, unsigned long long host, long long esz,
    unsigned long long* ptrs, int* missq, int nslots, void* stream) {
    inkwell_moe_resolve<<<(nslots + 127)/128, 128, 0, (cudaStream_t)stream>>>(
        ids, map, lru, stamp, arena, host, esz, ptrs, missq, nslots);
}
extern "C" void ink_launch_moe_alloc(const int* missq, int* map,
    unsigned long long* lru, unsigned long long* stamp, int* rev,
    unsigned long long arena, unsigned long long host, long long esz,
    unsigned long long* ptrs, unsigned long long* work, int k, void* stream) {
    inkwell_moe_alloc<<<1, 32, 0, (cudaStream_t)stream>>>(
        missq, map, lru, stamp, rev, arena, host, esz, ptrs, work, k);
}
extern "C" void ink_launch_moe_gather(const int* missq,
    const unsigned long long* work, long long esz, unsigned long long* stats, void* stream) {
    inkwell_moe_gather<<<8, 512, 0, (cudaStream_t)stream>>>(missq, work, esz, stats);
}

struct block_q8_0 { uint16_t d; int8_t qs[32]; };

// Dense GEMV for q8_0: x quantized to int8 per 32-val block (same class as
// ggml's q8_1 x-quant), then 8 dp4a per weight block. 2 warps per row.
__global__ void inkwell_mmvq_q8_0_moe_i8(const void* const* __restrict__ Wp,
                                         const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    x += slot * x_stride;
    const int nblk = (int)(ncols / 32);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 32; i += 16) m = fmaxf(m, fabsf(x[b * 32 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 5];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_q8_0* w = (const block_q8_0*)((const uint8_t*)Wp[slot]) + row * nblk;
    float acc = 0.f;
    for (int b = lane + half * 32; b < nblk; b += 64) {
        // qs sits at offset 2 in the 34B block — not 4B-aligned: compose u32s
        const uint32_t q[8] = {
            (uint32_t)*(const uint16_t*)&w[b].qs[0]  | ((uint32_t)*(const uint16_t*)&w[b].qs[2]  << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[4]  | ((uint32_t)*(const uint16_t*)&w[b].qs[6]  << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[8]  | ((uint32_t)*(const uint16_t*)&w[b].qs[10] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[12] | ((uint32_t)*(const uint16_t*)&w[b].qs[14] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[16] | ((uint32_t)*(const uint16_t*)&w[b].qs[18] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[20] | ((uint32_t)*(const uint16_t*)&w[b].qs[22] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[24] | ((uint32_t)*(const uint16_t*)&w[b].qs[26] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[28] | ((uint32_t)*(const uint16_t*)&w[b].qs[30] << 16),
        };
        const uint32_t* xp = (const uint32_t*)(qx + b * 32);
        int isum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) isum = __dp4a((int)xp[j], (int)q[j], isum);
        acc += fp16_to_f32(w[b].d) * sx[b] * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[slot * nrows + row], acc);
}


// Dense variant: single weight tensor, no slot indirection.
__global__ void inkwell_mmvq_q8_0_i8(const void* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int64_t nrows, int64_t ncols) {
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    const int nblk = (int)(ncols / 32);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 32; i += 16) m = fmaxf(m, fabsf(x[b * 32 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 5];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_q8_0* w = (const block_q8_0*)W + row * nblk;
    float acc = 0.f;
#pragma unroll 2
    for (int b = lane + half * 32; b < nblk; b += 64) {
        const uint32_t q[8] = {
            (uint32_t)*(const uint16_t*)&w[b].qs[0]  | ((uint32_t)*(const uint16_t*)&w[b].qs[2]  << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[4]  | ((uint32_t)*(const uint16_t*)&w[b].qs[6]  << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[8]  | ((uint32_t)*(const uint16_t*)&w[b].qs[10] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[12] | ((uint32_t)*(const uint16_t*)&w[b].qs[14] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[16] | ((uint32_t)*(const uint16_t*)&w[b].qs[18] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[20] | ((uint32_t)*(const uint16_t*)&w[b].qs[22] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[24] | ((uint32_t)*(const uint16_t*)&w[b].qs[26] << 16),
            (uint32_t)*(const uint16_t*)&w[b].qs[28] | ((uint32_t)*(const uint16_t*)&w[b].qs[30] << 16),
        };
        const uint32_t* xp = (const uint32_t*)(qx + b * 32);
        int isum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) isum = __dp4a((int)xp[j], (int)q[j], isum);
        acc += fp16_to_f32(w[b].d) * sx[b] * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[row], acc);
}
extern "C" void ink_launch_mmvq_q8_0_i8(const void* W,
                                        const float* x, float* y,
                                        long long nrows, long long ncols, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16));
    cudaMemsetAsync(y, 0, (size_t)nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_q8_0_i8<<<g, 1024, ((size_t)ncols + 8) * 4, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}

extern "C" void ink_launch_mmvq_q8_0_moe_i8(const void* const* Wp,
                                            const float* x, float* y,
                                            long long nrows, long long ncols, long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 7) / 8), (unsigned)nslots);
    cudaMemsetAsync(y, 0, (size_t)nslots * nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_q8_0_moe_i8<<<g, 512, ((size_t)ncols + 8) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// Q6_K via DP4A: val n = ((ql_nib | qh_bits<<4) - 32) in int8; x int8 per 16-vals.
// block_q6_K { u8 ql[128] (lo/hi nibbles for n<128/>=128), u8 qh[64] (2b each for
// quarters n/64), i8 scales[16], u16 d } = 210B / 256 vals.
struct block_q6_k { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
__global__ void inkwell_mmvq_q6_k_i8(const void* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int64_t nrows, int64_t ncols) {
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    const int nxb = (int)(ncols / 16);
    for (int b = (int)threadIdx.x >> 4; b < nxb; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = fabsf(x[b * 16 + t]);
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 4];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_q6_k* w = (const block_q6_k*)W + row * (ncols / 256);
    const int ngroups = (int)(ncols / 16);   // 16-val groups
    float acc = 0.f;
#pragma unroll 1
    for (int G = lane + half * 32; G < ngroups; G += 64) {
        const int b = G >> 4;                // 256-val superblock
        const int g = G & 15;                // group inside
        // real Q6_K interleave per 128-val half (dequantize_q6_K):
        const int ql_off = 64 * (g >> 3) + 16 * (g & 1) + 32 * ((g >> 1) & 1);
        const int qh_off = 32 * (g >> 3) + 16 * (g & 1);
        const int hish = ((g >> 2) & 1) * 4;   // nibble: lo for s<2, hi for s>=2
        const int qsh  = 2 * ((g & 7) >> 1);   // qh bit shift per segment
        const uint16_t* qlp = (const uint16_t*)(w[b].ql + ql_off);
        const uint16_t* qhp = (const uint16_t*)(w[b].qh + qh_off);
        const uint32_t* xp = (const uint32_t*)(qx + G * 16);
        int isum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = (uint32_t)qlp[2 * j] | ((uint32_t)qlp[2 * j + 1] << 16);
            const uint32_t h = (uint32_t)qhp[2 * j] | ((uint32_t)qhp[2 * j + 1] << 16);
            uint32_t wpack = 0;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int nib = (int)((q >> (8 * v + hish)) & 0xF);
                const int hb  = (int)((h >> (8 * v + qsh)) & 3);
                wpack |= (uint32_t)(uint8_t)((nib | (hb << 4)) - 32) << (8 * v);
            }
            isum = __dp4a((int)xp[j], (int)wpack, isum);
        }
        const int sc = w[b].scales[8 * (g >> 3) + (g & 1) + 2 * ((g & 7) >> 1)];
        acc += fp16_to_f32(w[b].d) * (float)sc * sx[G] * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[row], acc);
}
extern "C" void ink_launch_mmvq_q6_k_i8(const void* W,
                                        const float* x, float* y,
                                        long long nrows, long long ncols, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16));
    cudaMemsetAsync(y, 0, (size_t)nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_q6_k_i8<<<g, 1024, ((size_t)ncols + 16) * 4, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}

// Q5_K via DP4A: y = d*sc*(nib|hb<<4)*x - dmin*m*x. q fits int8; the min term
// folds in as -dmin*m*sx*sum(x_i8) via dp4a with 0x01010101.
// block_q5_K { u16 dm[2] (d, dmin), u8 scales[12] (8 sc + 8 min, 6-bit), u8 qh[32], u8 qs[128] } = 176B/256.
struct block_q5_k { uint16_t dm[2]; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
__device__ __forceinline__ void q5k_scale_min(int k, const uint8_t* sc, int& s, int& m) {
    if (k < 4) { s = sc[k] & 63; m = sc[k + 4] & 63; }
    else { s = (sc[k + 4] & 0xF) | ((sc[k - 4] >> 6) << 4);
           m = (sc[k + 4] >> 4) | ((sc[k] >> 6) << 4); }
}
__global__ void inkwell_mmvq_q5_k_i8(const void* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int64_t nrows, int64_t ncols) {
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    const int nxb = (int)(ncols / 16);
    for (int b = (int)threadIdx.x >> 4; b < nxb; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = fabsf(x[b * 16 + t]);
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 4];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (row >= nrows) return;
    const block_q5_k* w = (const block_q5_k*)W + row * (ncols / 256);
    const int ngroups = (int)(ncols / 16);
    float acc = 0.f;
#pragma unroll 1
    for (int G = lane + half * 32; G < ngroups; G += 64) {
        const int b = G >> 4;
        const int g = G & 15;                     // group inside 256-block
        const int ql_off = 32 * (g >> 2) + 16 * (g & 1);
        const int hish = ((g >> 1) & 1) * 4;      // nibble: lo for first half of 32-sector
        const int qhb  = 2 * (g >> 2) + ((g >> 1) & 1); // uniform qh bit per group
        const uint16_t* qlp = (const uint16_t*)(w[b].qs + ql_off);
        const uint16_t* qhp = (const uint16_t*)(w[b].qh + 16 * (g & 1));
        int sc6, m6;
        q5k_scale_min(2 * (g >> 2) + ((g >> 1) & 1), w[b].scales, sc6, m6);
        const uint32_t* xp = (const uint32_t*)(qx + G * 16);
        int isum = 0, xsum = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = (uint32_t)qlp[2 * j] | ((uint32_t)qlp[2 * j + 1] << 16);
            const uint32_t h = (uint32_t)qhp[2 * j] | ((uint32_t)qhp[2 * j + 1] << 16);
            uint32_t wpack = 0;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int nib = (int)((q >> (8 * v + hish)) & 0xF);
                const int hb  = (h >> (8 * v + qhb)) & 1;
                wpack |= (uint32_t)(uint8_t)(nib | (hb << 4)) << (8 * v);
            }
            isum = __dp4a((int)xp[j], (int)wpack, isum);
            xsum = __dp4a((int)xp[j], 0x01010101, xsum);
        }
        const float d = fp16_to_f32(((const uint16_t*)&w[b])[0]);
        const float dmin = fp16_to_f32(((const uint16_t*)&w[b])[1]);
        acc += (d * (float)sc6 * (float)isum - dmin * (float)m6 * (float)xsum) * sx[G];
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) atomicAdd(&y[row], acc);
}
extern "C" void ink_launch_mmvq_q5_k_i8(const void* W,
                                        const float* x, float* y,
                                        long long nrows, long long ncols, void* stream) {
    dim3 g((unsigned)((nrows + 15) / 16));
    cudaMemsetAsync(y, 0, (size_t)nrows * 4, (cudaStream_t)stream);
    inkwell_mmvq_q5_k_i8<<<g, 1024, ((size_t)ncols + 16) * 4, (cudaStream_t)stream>>>(W, x, y, nrows, ncols);
}

// IQ2_XS via DP4A: same sign-mask trick as iq2xxs_i8 but grid index comes from
// qs[g8] (u16: low 9 bits grid row, top 7 bits sign-pack index) and the scale
// nibble lives in scales[ib32] at position (l>>1).
__global__ void inkwell_mmvq_iq2_xs_moe_i8(const void* const* __restrict__ Wp,
                                         const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    __shared__ uint64_t smask[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        uint64_t m = 0;
        for (int b = 0; b < 8; ++b) if (i & (1 << b)) m |= 0xFFull << (8 * b);
        smask[i] = m;
    }
    x += slot * x_stride;
    const int nblk = (int)(ncols / 32);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 32; i += 16) m = fmaxf(m, fabsf(x[b * 32 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 5];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_iq2_xs* w = (const block_iq2_xs*)((const uint8_t*)Wp[slot])
                            + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g >> 5, g8 = g & 31, ib32 = g8 >> 2, l = g8 & 3;
        const uint16_t qv = w[b].qs[g8];
        const uint64_t grid = iq2xs_grid[qv & 511];
        const float dsx = fp16_to_f32(w[b].d)
            * (0.5f + ((w[b].scales[ib32] >> (4 * (l >> 1))) & 0xF)) * 0.25f
            * sx[g >> 2];
        const uint64_t msk = smask[ksigns_iq2xs[qv >> 9]];
        const uint32_t g0 = (uint32_t)grid, g1 = (uint32_t)(grid >> 32);
        const uint32_t m0 = (uint32_t)msk, m1 = (uint32_t)(msk >> 32);
        const uint32_t w0 = (g0 & ~m0) | (__vsubss4(0u, g0) & m0);
        const uint32_t w1 = (g1 & ~m1) | (__vsubss4(0u, g1) & m1);
        const uint32_t* xp = (const uint32_t*)(qx + g * 8);
        int isum = __dp4a((int)xp[0], (int)w0, __dp4a((int)xp[1], (int)w1, 0));
        acc += dsx * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}
extern "C" void ink_launch_mmvq_iq2_xs_moe_i8(const void* const* Wp,
                                              const float* x, float* y, long long nrows, long long ncols,
                                              long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 31) / 32), (unsigned)nslots);
    inkwell_mmvq_iq2_xs_moe_i8<<<g, 1024, (size_t)ncols + (size_t)(ncols / 32) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// IQ2_S via DP4A: gidx = qs[4*ib32+l] | qh top bits -> iq2s_grid row;
// signs byte at qs[32+4*ib32+l]; same smask/vsubss4 trick.
__global__ void inkwell_mmvq_iq2_s_moe_i8(const void* const* __restrict__ Wp,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int64_t nrows, int64_t ncols, int64_t x_stride) {
    const int slot = blockIdx.y;
    extern __shared__ int8_t qx[];
    float* sx = (float*)(qx + ncols);
    __shared__ uint64_t smask[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        uint64_t m = 0;
        for (int b = 0; b < 8; ++b) if (i & (1 << b)) m |= 0xFFull << (8 * b);
        smask[i] = m;
    }
    x += slot * x_stride;
    const int nblk = (int)(ncols / 32);
    for (int b = (int)threadIdx.x >> 4; b < nblk; b += (int)(blockDim.x >> 4)) {
        const int t = threadIdx.x & 15;
        float m = 0.f;
        for (int i = t; i < 32; i += 16) m = fmaxf(m, fabsf(x[b * 32 + i]));
        for (int o = 8; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (t == 0) sx[b] = m / 127.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < (int)ncols; i += blockDim.x) {
        const float sc = sx[i >> 5];
        qx[i] = (int8_t)(sc > 0.f ? (int)rintf(x[i] / sc) : 0);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= nrows) return;
    const block_iq2_s* w = (const block_iq2_s*)((const uint8_t*)Wp[slot])
                           + row * (ncols / 256);
    const int ngroups = (int)(ncols / 8);
    float acc = 0.f;
    for (int g = lane; g < ngroups; g += 32) {
        const int b = g >> 5, ib32 = (g & 31) >> 2, l = g & 3;
        const int gidx = w[b].qs[4 * ib32 + l] | ((w[b].qh[ib32] << (8 - 2 * l)) & 0x300);
        const uint64_t grid = iq2s_grid[gidx];
        const float dsx = fp16_to_f32(w[b].d)
            * (0.5f + ((w[b].scales[ib32] >> (4 * (l >> 1))) & 0xF)) * 0.25f
            * sx[g >> 2];
        const uint64_t msk = smask[w[b].qs[32 + 4 * ib32 + l]];
        const uint32_t g0 = (uint32_t)grid, g1 = (uint32_t)(grid >> 32);
        const uint32_t m0 = (uint32_t)msk, m1 = (uint32_t)(msk >> 32);
        const uint32_t w0 = (g0 & ~m0) | (__vsubss4(0u, g0) & m0);
        const uint32_t w1 = (g1 & ~m1) | (__vsubss4(0u, g1) & m1);
        const uint32_t* xp = (const uint32_t*)(qx + g * 8);
        int isum = __dp4a((int)xp[0], (int)w0, __dp4a((int)xp[1], (int)w1, 0));
        acc += dsx * (float)isum;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[slot * nrows + row] = acc;
}
extern "C" void ink_launch_mmvq_iq2_s_moe_i8(const void* const* Wp,
                                             const float* x, float* y, long long nrows, long long ncols,
                                             long long x_stride, int nslots, void* stream) {
    dim3 g((unsigned)((nrows + 31) / 32), (unsigned)nslots);
    inkwell_mmvq_iq2_s_moe_i8<<<g, 1024, (size_t)ncols + (size_t)(ncols / 32) * 4, (cudaStream_t)stream>>>(Wp, x, y, nrows, ncols, x_stride);
}

// ===== fused decode attention (Pascal-safe replacement for flash_attn_ext) =====
// One block per query head. score[j] = scale * <q_h, K[hkv(h), j]> + bias[j + h*n_kv] + mask[j]
// softmax over n_kv real positions plus an implicit zero-logit sink whose mask
// entry is mask[n_kv]. out[h*hd + d] = sum_j attn[j] * V[hkv(h), j, d].
template<bool KV16>
__global__ void inkwell_attn_decode(const void* __restrict__ K, const void* __restrict__ V,
                                    const float* __restrict__ Q, const float* __restrict__ bias,
                                    const float* __restrict__ mask, float* __restrict__ out,
                                    int nh, int hkv, int n_kv, int hd,
                                    int k_s1, int k_s2, int v_s1, int v_s2,
                                    int q_s2, int b_s2, float scale) {
    extern __shared__ float sm[];
    float* scores = sm;
    float* qs = sm + n_kv;
    __shared__ float red[32];
    const int h = blockIdx.x;
    const int hk = h / (nh / hkv);
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;
    const int w = threadIdx.x >> 5;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = Q[d + h * q_s2];
    __syncthreads();
    const char* Kh = (const char*)K + (size_t)hk * k_s2;
    for (int j = w; j < n_kv; j += nw) {
        const char* kj = Kh + (size_t)j * k_s1;
        float a = 0.f;
        if (KV16) {
            const __half* kjh = (const __half*)kj;
            for (int d = lane; d < hd; d += 32) a += qs[d] * __half2float(kjh[d]);
        } else {
            const float* kjf = (const float*)kj;
            for (int d = lane; d < hd; d += 32) a += qs[d] * kjf[d];
        }
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
        if (lane == 0) scores[j] = a;
    }
    __syncthreads();
    float m = -1e30f;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float s = scores[j] * scale + bias[j + h * b_s2] + mask[j];
        scores[j] = s;
        m = fmaxf(m, s);
    }
    m = fmaxf(m, mask[n_kv]); // zero-logit sink with its mask entry
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[w] = m;
    __syncthreads();
    if (w == 0) {
        float mm = -1e30f;
        for (int i = lane; i < nw; i += 32) mm = fmaxf(mm, red[i]);
        for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, o));
        if (lane == 0) red[0] = mm;
    }
    __syncthreads();
    const float mx = red[0];
    float den = 0.f;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        const float e = __expf(scores[j] - mx);
        scores[j] = e;
        den += e;
    }
    if (threadIdx.x == 0) den += __expf(mask[n_kv] - mx);
    for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (lane == 0) red[w] = den;
    __syncthreads();
    if (w == 0) {
        float dd = 0.f;
        for (int i = lane; i < nw; i += 32) dd += red[i];
        for (int o = 16; o > 0; o >>= 1) dd += __shfl_xor_sync(0xffffffffu, dd, o);
        if (lane == 0) red[0] = dd;
    }
    __syncthreads();
    const float inv = 1.f / red[0];
    const char* Vh = (const char*)V + (size_t)hk * v_s2;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        if (KV16) {
            const __half* vh = (const __half*)Vh;
            for (int j = 0; j < n_kv; ++j) o += scores[j] * __half2float(vh[(size_t)j * v_s1 / 2 + d]);
        } else {
            const float* vf = (const float*)Vh;
            for (int j = 0; j < n_kv; ++j) o += scores[j] * vf[(size_t)j * v_s1 / 4 + d];
        }
        out[h * hd + d] = o * inv;
    }
}
// Split-K decode attention. Each block owns one KV chunk and up to three query
// heads sharing one KV head. Partials are [m, l, acc(hd)] per (chunk, q head);
// the reduce kernel adds the same zero-logit sink as the single-block kernel.
#define INKWELL_ATTN_MAX_SPLIT 512
#define INKWELL_ATTN_MAX_QG 3

// Warp-split variant: each warp owns a strided slice of the chunk and keeps
// one online-softmax state per grouped query head. A warp lane owns four V
// columns, so K and V are each read once with no tile staging or block-wide
// synchronization in the token loop.
template<bool KV16>
__global__ void inkwell_attn_decode_splitw(const void* __restrict__ K, const void* __restrict__ V,
                                           const float* __restrict__ Q, const float* __restrict__ bias,
                                           const float* __restrict__ mask, float* __restrict__ part,
                                           int nh, int hkv, int n_kv, int hd,
                                           int k_s1, int k_s2, int v_s1, int v_s2,
                                           int q_s2, int b_s2, float scale,
                                           int chunk_len, int qper, int nsub) {
    extern __shared__ float sw[];
    float* sacc = sw;
    float* smm = sacc + INKWELL_ATTN_MAX_QG * 128;
    float* sml = smm + INKWELL_ATTN_MAX_QG;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int w = tid >> 5;
    const int nw = blockDim.x >> 5;
    const int chunk = blockIdx.x;
    const int hk = blockIdx.y / nsub;
    const int sg = blockIdx.y - hk * nsub;
    const int ratio = nh / hkv;
    const int qrel0 = sg * qper;
    const int gc = min(qper, ratio - qrel0);
    const int qbase = hk * ratio + qrel0;
    const int j0 = chunk * chunk_len;
    const int j1 = min(n_kv, j0 + chunk_len);
    const int d0 = lane * 4;
    const char* Kh = (const char*)K + (size_t)hk * k_s2;
    const char* Vh = (const char*)V + (size_t)hk * v_s2;

    float qr[INKWELL_ATTN_MAX_QG][4] = {};
    float acc[INKWELL_ATTN_MAX_QG][4] = {};
    float mst[INKWELL_ATTN_MAX_QG], lst[INKWELL_ATTN_MAX_QG];
    #pragma unroll
    for (int g = 0; g < INKWELL_ATTN_MAX_QG; ++g) {
        mst[g] = -1e30f;
        lst[g] = 0.f;
    }
    for (int g = 0; g < gc; ++g) {
        const float* qp = Q + (size_t)(qbase + g) * q_s2 + d0;
        #pragma unroll
        for (int f = 0; f < 4; ++f) qr[g][f] = qp[f];
    }

    uint2 k16 = make_uint2(0, 0), v16 = make_uint2(0, 0);
    uint2 kn16 = make_uint2(0, 0), vn16 = make_uint2(0, 0);
    float4 k32 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 v32 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 kn32 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 vn32 = make_float4(0.f, 0.f, 0.f, 0.f);
    if (j0 + w < j1) {
        const char* kj = Kh + (size_t)(j0 + w) * k_s1;
        const char* vj = Vh + (size_t)(j0 + w) * v_s1;
        if (KV16) {
            k16 = __ldcs((const uint2*)(kj + lane * 8));
            v16 = __ldcs((const uint2*)(vj + lane * 8));
        } else {
            k32 = __ldcs((const float4*)(kj + lane * 16));
            v32 = __ldcs((const float4*)(vj + lane * 16));
        }
    }
    for (int j = j0 + w; j < j1; j += nw) {
        const int jn = j + nw;
        if (jn < j1) {
            const char* kj = Kh + (size_t)jn * k_s1;
            const char* vj = Vh + (size_t)jn * v_s1;
            if (KV16) {
                kn16 = __ldcs((const uint2*)(kj + lane * 8));
                vn16 = __ldcs((const uint2*)(vj + lane * 8));
            } else {
                kn32 = __ldcs((const float4*)(kj + lane * 16));
                vn32 = __ldcs((const float4*)(vj + lane * 16));
            }
        }

        float kval[4], vval[4];
        if (KV16) {
            const __half2 k01 = *(const __half2*)&k16.x;
            const __half2 k23 = *(const __half2*)&k16.y;
            const __half2 v01 = *(const __half2*)&v16.x;
            const __half2 v23 = *(const __half2*)&v16.y;
            kval[0] = __low2float(k01); kval[1] = __high2float(k01);
            kval[2] = __low2float(k23); kval[3] = __high2float(k23);
            vval[0] = __low2float(v01); vval[1] = __high2float(v01);
            vval[2] = __low2float(v23); vval[3] = __high2float(v23);
        } else {
            kval[0] = k32.x; kval[1] = k32.y; kval[2] = k32.z; kval[3] = k32.w;
            vval[0] = v32.x; vval[1] = v32.y; vval[2] = v32.z; vval[3] = v32.w;
        }

        float a[INKWELL_ATTN_MAX_QG] = {};
        #pragma unroll
        for (int g = 0; g < INKWELL_ATTN_MAX_QG; ++g) {
            if (g < gc) a[g] = qr[g][0] * kval[0] + qr[g][1] * kval[1] + qr[g][2] * kval[2] + qr[g][3] * kval[3];
        }
        for (int g = 0; g < gc; ++g) {
            for (int o = 16; o > 0; o >>= 1) a[g] += __shfl_xor_sync(0xffffffffu, a[g], o);
        }
        const float mk = mask[j];
        #pragma unroll
        for (int g = 0; g < INKWELL_ATTN_MAX_QG; ++g) {
            if (g >= gc) continue;
            const int h = qbase + g;
            const float sc = a[g] * scale + bias[j + h * b_s2] + mk;
            const float mold = mst[g];
            if (sc > mold) {
                const float c = __expf(mold - sc);
                mst[g] = sc;
                lst[g] = lst[g] * c + 1.f;
                #pragma unroll
                for (int f = 0; f < 4; ++f) acc[g][f] = acc[g][f] * c + vval[f];
            } else {
                const float p = __expf(sc - mold);
                lst[g] += p;
                #pragma unroll
                for (int f = 0; f < 4; ++f) acc[g][f] += p * vval[f];
            }
        }
        k16 = kn16;
        v16 = vn16;
        k32 = kn32;
        v32 = vn32;
    }

    if (w == 0) {
        #pragma unroll
        for (int g = 0; g < INKWELL_ATTN_MAX_QG; ++g) {
            if (g < gc) {
                if (lane == 0) {
                    smm[g] = mst[g];
                    sml[g] = lst[g];
                }
                #pragma unroll
                for (int f = 0; f < 4; ++f) sacc[g * 128 + d0 + f] = acc[g][f];
            }
        }
    }
    __syncthreads();

    if (w == 1) {
        #pragma unroll
        for (int g = 0; g < INKWELL_ATTN_MAX_QG; ++g) {
            if (g >= gc) continue;
            float* pb = part + ((size_t)chunk * nh + qbase + g) * (hd + 2);
            const float mx = fmaxf(smm[g], mst[g]);
            const float w0 = __expf(smm[g] - mx);
            const float w1 = __expf(mst[g] - mx);
            if (lane == 0) {
                pb[0] = mx;
                pb[1] = sml[g] * w0 + lst[g] * w1;
            }
            #pragma unroll
            for (int f = 0; f < 4; ++f) {
                pb[2 + d0 + f] = sacc[g * 128 + d0 + f] * w0 + acc[g][f] * w1;
            }
        }
    }
}


__global__ void inkwell_attn_decode_reduce(const float* __restrict__ part, const float* __restrict__ mask,
                                           float* __restrict__ out, int nh, int n_kv, int hd, int nchunks) {
    __shared__ float ws[INKWELL_ATTN_MAX_SPLIT];
    __shared__ float red[128];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    float mc = -1e30f;
    for (int c = tid; c < nchunks; c += 128) {
        mc = fmaxf(mc, part[((size_t)c * nh + h) * (hd + 2)]);
    }
    red[tid] = mc;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float mx = fmaxf(red[0], mask[n_kv]);
    __syncthreads();

    float den = 0.f;
    for (int c = tid; c < nchunks; c += 128) {
        const float* pb = part + ((size_t)c * nh + h) * (hd + 2);
        const float wc = __expf(pb[0] - mx);
        ws[c] = wc;
        den += pb[1] * wc;
    }
    red[tid] = den;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float inv = 1.f / (red[0] + __expf(mask[n_kv] - mx));
    for (int d = tid; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int c = 0; c < nchunks; ++c) {
            const float* pb = part + ((size_t)c * nh + h) * (hd + 2);
            o += pb[2 + d] * ws[c];
        }
        out[h * hd + d] = o * inv;
    }
}

extern "C" void ink_launch_attn_decode_old(const void* K, const void* V, const void* Q,
                                           const void* bias, const void* mask, void* out,
                                           long long nh, long long hkv, long long n_kv, long long hd,
                                           long long k_s1, long long k_s2, long long v_s1, long long v_s2,
                                           long long q_s2, long long b_s2, float scale, int kv16, void* stream) {
    const dim3 g((unsigned)nh), b(256);
    const size_t sm = (size_t)(n_kv + hd) * 4;
    const cudaStream_t st = (cudaStream_t)stream;
    if (kv16) {
        inkwell_attn_decode<true><<<g, b, sm, st>>>(
            K, V, (const float*)Q, (const float*)bias, (const float*)mask, (float*)out,
            (int)nh, (int)hkv, (int)n_kv, (int)hd,
            (int)k_s1, (int)k_s2, (int)v_s1, (int)v_s2, (int)q_s2, (int)b_s2, scale);
    } else {
        inkwell_attn_decode<false><<<g, b, sm, st>>>(
            K, V, (const float*)Q, (const float*)bias, (const float*)mask, (float*)out,
            (int)nh, (int)hkv, (int)n_kv, (int)hd,
            (int)k_s1, (int)k_s2, (int)v_s1, (int)v_s2, (int)q_s2, (int)b_s2, scale);
    }
}

static float* inkwell_attn_partbuf(size_t floats) {
    static float* p = nullptr;
    static size_t cap = 0;
    if (p && cap >= floats) return p;
    if (!p && cudaMalloc((void**)&p, floats * sizeof(float)) == cudaSuccess) {
        cap = floats;
        return p;
    }
    return nullptr;
}

extern "C" int ink_launch_attn_decode_split(const void* K, const void* V, const void* Q,
                                            const void* bias, const void* mask, void* out,
                                            long long nh, long long hkv, long long n_kv, long long hd,
                                            long long k_s1, long long k_s2, long long v_s1, long long v_s2,
                                            long long q_s2, long long b_s2, float scale, int kv16,
                                            int chunk_len, void* stream) {
    const cudaStream_t st = (cudaStream_t)stream;
    const int ratio = (int)(nh / hkv);
    const int nchunks = (int)((n_kv + chunk_len - 1) / chunk_len);
    if (nchunks > INKWELL_ATTN_MAX_SPLIT) return 0;
    float* part = inkwell_attn_partbuf((size_t)INKWELL_ATTN_MAX_SPLIT * nh * (hd + 2));
    if (!part) return 0;
    const int qper = min(INKWELL_ATTN_MAX_QG, ratio);
    const int nsub = (ratio + qper - 1) / qper;
    const dim3 gp((unsigned)nchunks, (unsigned)(hkv * nsub));
    const size_t smw = (INKWELL_ATTN_MAX_QG * 128 + 2 * INKWELL_ATTN_MAX_QG) * 4;
    if (kv16) {
        inkwell_attn_decode_splitw<true><<<gp, dim3(64), smw, st>>>(
            K, V, (const float*)Q, (const float*)bias, (const float*)mask, part,
            (int)nh, (int)hkv, (int)n_kv, (int)hd,
            (int)k_s1, (int)k_s2, (int)v_s1, (int)v_s2, (int)q_s2, (int)b_s2, scale,
            chunk_len, qper, nsub);
    } else {
        inkwell_attn_decode_splitw<false><<<gp, dim3(64), smw, st>>>(
            K, V, (const float*)Q, (const float*)bias, (const float*)mask, part,
            (int)nh, (int)hkv, (int)n_kv, (int)hd,
            (int)k_s1, (int)k_s2, (int)v_s1, (int)v_s2, (int)q_s2, (int)b_s2, scale,
            chunk_len, qper, nsub);
    }
    inkwell_attn_decode_reduce<<<dim3((unsigned)nh), 128, 0, st>>>(
        part, (const float*)mask, (float*)out, (int)nh, (int)n_kv, (int)hd, nchunks);
    return 1;
}

extern "C" void ink_launch_attn_decode(const void* K, const void* V, const void* Q,
                                       const void* bias, const void* mask, void* out,
                                       long long nh, long long hkv, long long n_kv, long long hd,
                                       long long k_s1, long long k_s2, long long v_s1, long long v_s2,
                                       long long q_s2, long long b_s2, float scale, int kv16, void* stream) {
    // k/v strides arrive in BYTES (dtype-agnostic); q_s2/b_s2 stay f32 elements
    static int no_split = -1;
    static int split_min = -1;
    static int chunk_len = -1;
    if (no_split < 0) {
        no_split = getenv("INKWELL_NO_ATTNSPLIT") ? 1 : 0;
        const char* e = getenv("INKWELL_ATTNSPLIT_MIN");
        split_min = e ? atoi(e) : 320;
        e = getenv("INKWELL_ATTNSPLIT_CHUNK");
        chunk_len = e ? atoi(e) : 0;
        if (e && chunk_len < 32) chunk_len = 32;
    }

    int use_chunk = chunk_len;
    if (use_chunk == 0) use_chunk = n_kv <= 16384 ? 32 : 256;
    const int ratio = hkv > 0 ? (int)(nh / hkv) : 0;
    const bool shape_ok = hkv > 0 && nh % hkv == 0 && ratio > 0 && hd == 128 &&
        (kv16 ? (k_s1 % 8 == 0 && v_s1 % 16 == 0) : (k_s1 % 16 == 0 && v_s1 % 16 == 0));
    const int nchunks = shape_ok ? (int)((n_kv + use_chunk - 1) / use_chunk) : 0;
    if (!no_split && shape_ok && n_kv >= split_min && nchunks <= INKWELL_ATTN_MAX_SPLIT &&
        ink_launch_attn_decode_split(K, V, Q, bias, mask, out, nh, hkv, n_kv, hd,
            k_s1, k_s2, v_s1, v_s2, q_s2, b_s2, scale, kv16, use_chunk, stream)) {
        return;
    }
    ink_launch_attn_decode_old(K, V, Q, bias, mask, out, nh, hkv, n_kv, hd,
        k_s1, k_s2, v_s1, v_s2, q_s2, b_s2, scale, kv16, stream);
}

// ===================== EXL3 trellis GEMV (sm_61, mul1 codebook) =====================
// W = diag(suh) H What H diag(svh); y = svh . H (What^T-ish GEMV over decoded tiles).
// Decomposition: xh = H128*(suh*x); t[n] = sum_k xh[k]*What[k,n]; y = svh*H128*t.
// No tensor cores on Pascal: decode + scalar FMA. mul1 codebook only (all tensors
// in the SAGE pack are mul1-marked). K = trellis.shape[2]/16; half-integer K uses
// step widths KA + bit(i) of 0xAAAA (odd positions get +1 bit).
// tile_perm: ring position i=8t+j -> tile element r=(t%4)*2+(j&1)+(j&2?8:0),
// c=t/4+(j&4?8:0).

__device__ __forceinline__ uint32_t exl3_window(const uint32_t* __restrict__ ring,
                                                int i, int ka, int halfk) {
    const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
    const int s_i = halfk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka; // inclusive prefix
    const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
    const int nw = r_bits >> 5;
    const int i0 = b0 >> 5;
    const int s0 = 48 - (b0 & 31);
    const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[i1];
    return (uint32_t)(v >> s0) & 0xffffu;
}

__device__ __forceinline__ float exl3_mul1(uint32_t w) {
    w *= 0x83DCD12Du;
    const uint32_t s = __dp4a(w, 0x01010101u, 0x6400u); // bytesum + fp16 bits 0x6400
    const __half h = __ushort_as_half((uint16_t)s);
    // fp32 fma + single fp16 rounding == __hfma(h, k_inv, k_bias) exactly
    const float hf = __half2float(h);
    return __half2float(__float2half(
        __fmaf_rn(hf, __half2float(__ushort_as_half(0x1eee)),
                  __half2float(__ushort_as_half(0xc931)))));
}

// xh = H128 * (suh .* x): block per 128-block, butterfly in smem, /sqrt(128).
__global__ void inkwell_exl3_had_in(const float* __restrict__ x,
                                    const uint16_t* __restrict__ suh,
                                    float* __restrict__ xh, int k) {
    extern __shared__ float sm[];
    const int blk = blockIdx.x;
    const int t = threadIdx.x;
    const int base = blk * 128;
    float v = (base + t < k) ? x[base + t] * __half2float(__ushort_as_half(suh[base + t])) : 0.f;
    sm[t] = v;
    __syncthreads();
    for (int s = 1; s < 128; s <<= 1) {
        const float u = (t & s) ? sm[t - s] - sm[t] : sm[t] + sm[t + s];
        __syncthreads();
        sm[t] = u;
        __syncthreads();
    }
    if (base + t < k) xh[base + t] = sm[t] * 0.08838834764831845f; // 1/sqrt(128)
}

// merged had_in, default on (INKWELL_NO_HADMERGE=1 restores per-matrix had_in;
// nsys graph-node trace: had_in 1.16 -> 0.62 ms/token).
// Member z applies the identical 128-point butterfly to its own x with its
// own suh into its own xh; only the launch count changes.
struct HadNArgs { const float* x[8]; const uint16_t* suh[8]; float* xh[8]; int n; int k; };
__global__ void inkwell_exl3_had_in_n(const HadNArgs a) {
    extern __shared__ float sm[];
    const int blk = blockIdx.x;
    const int t = threadIdx.x;
    const int base = blk * 128;
    const int z = blockIdx.y;
    float v = (base + t < a.k) ? a.x[z][base + t] * __half2float(__ushort_as_half(a.suh[z][base + t])) : 0.f;
    sm[t] = v;
    __syncthreads();
    for (int s = 1; s < 128; s <<= 1) {
        const float u = (t & s) ? sm[t - s] - sm[t] : sm[t] + sm[t + s];
        __syncthreads();
        sm[t] = u;
        __syncthreads();
    }
    if (base + t < a.k) a.xh[z][base + t] = sm[t] * 0.08838834764831845f; // 1/sqrt(128)
}

// t[n] = sum_k xh[k] * What[k,n]. grid.x = tn tiles; 256 threads, thread i = ring pos i.
__device__ __forceinline__ int r_bits_words(int ka, int halfk) {
    return (halfk ? 256 * ka + 128 : 256 * ka) / 32;
}

__global__ void inkwell_exl3_gemv(const uint32_t* __restrict__ trel,
                                  const float* __restrict__ xh,
                                  float* __restrict__ t,
                                  int tk, int tn, int ka, int halfk) {
    __shared__ float xhs[16];
    __shared__ float acc[16];
    const int tn_i = blockIdx.x;
    const int i = threadIdx.x;
    const int tt = i >> 3, j = i & 7;
    const int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
    const int c = (tt >> 2) + ((j & 4) ? 8 : 0);
    if (i < 16) acc[i] = 0.f;
    __syncthreads();
    const uint32_t* tile_base = trel;
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        if (i < 16) xhs[i] = xh[tk_i * 16 + i];
        __syncthreads();
        const uint32_t* ring = tile_base + (size_t)(tk_i * tn + tn_i) * (size_t)(r_bits_words(ka, halfk));
        const float wv = exl3_mul1(exl3_window(ring, i, ka, halfk));
        atomicAdd(&acc[c], xhs[r] * wv);
        __syncthreads();
    }
    if (i < 16) t[tn_i * 16 + i] = acc[i];
}

// y = svh .* (H128 * t): block per 128-block.
__global__ void inkwell_exl3_had_out(const float* __restrict__ t,
                                     const uint16_t* __restrict__ svh,
                                     float* __restrict__ y, int n, int nky, int tstride) {
    extern __shared__ float sm[];
    const int blk = blockIdx.x;
    const int th = threadIdx.x;
    const int base = blk * 128;
    float v = 0.f;
    if (base + th < n) { for (int z = 0; z < nky; ++z) v += t[z * tstride + base + th]; }
    sm[th] = v;
    __syncthreads();
    for (int s = 1; s < 128; s <<= 1) {
        const float u = (th & s) ? sm[th - s] - sm[th] : sm[th] + sm[th + s];
        __syncthreads();
        sm[th] = u;
        __syncthreads();
    }
    if (base + th < n) y[base + th] = sm[th] * 0.08838834764831845f
        * __half2float(__ushort_as_half(svh[base + th]));
}

// DP4A GEMV: dec(i) = fp16(1024 + bytesum(w*M)) * k_inv + k_bias is affine in
// bytesum → t[c] = k_inv * sum_r xh[r] * bytesum(w_rc*M) + k_bias * sum_r xh[r].
// bytesum = dp4a(w*M, 0x01010101). One thread per output column c loops all r:
// no atomics, no fp16 decode in the inner loop.
// x-int8-per-16-group + weight plane-byte dp4a variant.
// Per weight: xh[r]*bs[r] == dp4a(w16*MAGIC_bytes, broadcast(x8[r])) exactly,
// where x8 = per-window int8 quant of xh (16 elems share a scale).
// All windows' x8+scale are quantized once in the prologue (smem, no per-iter
// barrier). Requires tk <= 160 (launcher falls back otherwise).
__global__ void inkwell_exl3_gemv_dp4a_x8(const uint32_t* __restrict__ trel,
                                        const float* __restrict__ xh,
                                        float* __restrict__ t,
                                        int tk, int tn, int ka, int halfk) {
    const int g = threadIdx.x >> 6;
    const int u = threadIdx.x & 63;
    const int c = u >> 2, rp = u & 3;
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk);
    __shared__ uint32_t rings[4 * 8 * 80];
    __shared__ int8_t xq8[160][16];
    __shared__ float xsc[160];
    __shared__ float xhb[4][16];
    const bool live = tn_i < tn;
    const int tkc = (tk + gridDim.y - 1) / gridDim.y;
    const int tk0 = blockIdx.y * tkc;
    const int tk1 = min(tk0 + tkc, tk);
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    float s = 0.f, sx = 0.f;
    uint32_t widx[4];
    {
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int nw = r_bits >> 5;
        int q = 0;
        for (int r = rp; r < 16; r += 4, ++q) {
            const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                        | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
            const int s_i = halfk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
            const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
            const int i0 = b0 >> 5;
            const int s0 = 48 - (b0 & 31);
            const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
            widx[q] = (uint32_t)(i0 | (i1 << 7) | (s0 << 14));
        }
    }
    // prologue 1: quantize ALL of this block's windows (16-lane groups, shfl amax)
    {
        const int lane = threadIdx.x & 15;
        const int wgrp = threadIdx.x >> 4;  // 16 groups x 16 lanes
        for (int win = wgrp; win < tk1 - tk0; win += 16) {
            const float xv = xh[(tk0 + win) * 16 + lane];
            float amax = fabsf(xv);
#pragma unroll
            for (int m = 8; m > 0; m >>= 1)
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
            const float sc = fmaxf(amax / 127.f, 1e-12f);
            const float inv = amax > 0.f ? 127.f / amax : 0.f;
            const int v = max(-127, min(127, (int)roundf(xv * inv)));
            xq8[win][lane] = (int8_t)v;
            if (lane == 0) xsc[win] = sc;
        }
    }
    // prologue 2: fill ring + xhb for batch tk0
    if (threadIdx.x < 64) {
        const int j = threadIdx.x;
        const int tkn0h = min(4, tk1 - tk0);
        if (j < 16 * tkn0h) xhb[j >> 4][j & 15] = xh[(tk0 + (j >> 4)) * 16 + (j & 15)];
    }
    if (live) {
        const int tkn0 = min(4, tk1 - tk0);
        uint32_t* rg = rings + g * 4 * 80;
        for (int bn = 0; bn < tkn0; ++bn) {
            const uint32_t* ring = trel + (size_t)((tk0 + bn) * tn + tn_i) * (size_t)w;
            if (u < w) rg[bn * 80 + u] = ring[u];
            if (u + 64 < w) rg[bn * 80 + u + 64] = ring[u + 64];
        }
    }
    __syncthreads();
    for (int tkb0 = tk0; tkb0 < tk1; tkb0 += 4) {
        const int tkn = min(4, tk1 - tkb0);
        const int nb0 = tkb0 + 4;
        const int tkn2 = min(4, tk1 - nb0);
        uint32_t pfb[4][2];
        float pfx = 0.f;
        if (live && nb0 < tk1) {
            for (int b = 0; b < tkn2; ++b) {
                const uint32_t* ring = trel + (size_t)((nb0 + b) * tn + tn_i) * (size_t)w;
                if (u < w) pfb[b][0] = ring[u];
                if (u + 64 < w) pfb[b][1] = ring[u + 64];
            }
        }
        if (threadIdx.x < 64 && nb0 < tk1) {
            const int j = threadIdx.x;
            if (j < 16 * tkn2) pfx = xh[(nb0 + (j >> 4)) * 16 + (j & 15)];
        }
        if (live) {
            const uint32_t* rg = rings + g * 4 * 80;
            for (int b = 0; b < tkn; ++b) {
                const int win = tkb0 + b - tk0;
                const float scb = xsc[win];
                const int8_t* xw = xq8[win];
                uint32_t x8b[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    x8b[q] = (uint32_t)(uint8_t)xw[rp + 4 * q] * 0x01010101u;
                int qint = 0;
                float sxl = 0.f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t wi = widx[q];
                    const uint64_t v = ((uint64_t)rg[b * 80 + (wi & 127)] << 32)
                                     | (uint64_t)rg[b * 80 + ((wi >> 7) & 127)];
                    const uint32_t w16 = (uint32_t)(v >> (wi >> 14)) & 0xffffu;
                    asm volatile("dp4a.u32.s32 %0, %1, %2, %0;" : "+r"(qint)
                                 : "r"(w16 * 0x83DCD12Du), "r"(x8b[q]));
                    sxl += (float)(int8_t)x8b[q];
                }
                s += scb * (float)qint;
                sx += xhb[b][rp] + xhb[b][rp + 4] + xhb[b][rp + 8] + xhb[b][rp + 12];
                (void)sxl;
            }
        }
        __syncthreads();
        if (live && nb0 < tk1) {
            uint32_t* rg = rings + g * 4 * 80;
            for (int b = 0; b < tkn2; ++b) {
                if (u < w) rg[b * 80 + u] = pfb[b][0];
                if (u + 64 < w) rg[b * 80 + u + 64] = pfb[b][1];
            }
        }
        if (threadIdx.x < 64 && nb0 < tk1) {
            const int j = threadIdx.x;
            if (j < 16 * tkn2) xhb[j >> 4][j & 15] = pfx;
        }
        __syncthreads();
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    sx += __shfl_xor_sync(0xffffffffu, sx, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    sx += __shfl_xor_sync(0xffffffffu, sx, 2);
    if (live && rp == 0) t[blockIdx.y * (tn * 16) + tn_i * 16 + c] = fmaf(KINV, s, KOFF * sx);
}

__global__ void inkwell_exl3_gemv_dp4a(const uint32_t* __restrict__ trel,
                                       const float* __restrict__ xh,
                                       float* __restrict__ t,
                                       int tk, int tn, int ka, int halfk) {
    const int g = threadIdx.x >> 6;      // 4 tiles per block, 64 threads/tile
    const int u = threadIdx.x & 63;      // 4 threads per column (r split 4-way)
    const int c = u >> 2, rp = u & 3;
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk); // ≤80 u32 per tile
    __shared__ float xhb[8][16];
    __shared__ uint32_t rings[4 * 4 * 80]; // 4 tiles × 4 tk-batch × ≤80 words
    const bool live = tn_i < tn;
    // split-K: gridDim.y chunks over the tk dimension
    const int tkc = (tk + gridDim.y - 1) / gridDim.y;
    const int tk0 = blockIdx.y * tkc;
    const int tk1 = min(tk0 + tkc, tk);
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    float s = 0.f, sx = 0.f;
    // hoisted window indices per r-lane (i0|i1<<7|s0<<14) — (c,r) fixed for all batches
    uint32_t widx[4];
    {
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int nw = r_bits >> 5;
        int q = 0;
        for (int r = rp; r < 16; r += 4, ++q) {
            const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                        | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
            const int s_i = halfk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
            const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
            const int i0 = b0 >> 5;
            const int s0 = 48 - (b0 & 31);
            const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
            widx[q] = (uint32_t)(i0 | (i1 << 7) | (s0 << 14));
        }
    }
    // prologue fill of batch tk0 (global-load stall happens once)
    {
        const int tkn0 = min(4, tk1 - tk0);
        if (threadIdx.x < 64) {
            const int j = threadIdx.x;
            if (j < 16 * tkn0) xhb[j >> 4][j & 15] = xh[(tk0 + (j >> 4)) * 16 + (j & 15)];
        }
        if (live) {
            uint32_t* rg = rings + g * 320;
            for (int b = 0; b < tkn0; ++b) {
                const uint32_t* ring = trel + (size_t)((tk0 + b) * tn + tn_i) * (size_t)w;
                if (u < w) rg[b * 80 + u] = ring[u];
                if (u + 64 < w) rg[b * 80 + u + 64] = ring[u + 64];
            }
        }
    }
    __syncthreads();
    for (int tkb0 = tk0; tkb0 < tk1; tkb0 += 4) {
        const int tkn = min(4, tk1 - tkb0);
        const int nb0 = tkb0 + 4;
        const int tkn2 = min(4, tk1 - nb0);
        // prefetch next batch to registers; hide global latency behind math
        uint32_t pfb[4][2];
        float pfx = 0.f;
        if (live && nb0 < tk1) {
            for (int b = 0; b < tkn2; ++b) {
                const uint32_t* ring = trel + (size_t)((nb0 + b) * tn + tn_i) * (size_t)w;
                if (u < w) pfb[b][0] = ring[u];
                if (u + 64 < w) pfb[b][1] = ring[u + 64];
            }
        }
        if (threadIdx.x < 64 && nb0 < tk1) {
            const int j = threadIdx.x;
            if (j < 16 * tkn2) pfx = xh[(nb0 + (j >> 4)) * 16 + (j & 15)];
        }
        if (live) {
            const uint32_t* rg = rings + g * 320;
            for (int b = 0; b < tkn; ++b) {
                int q = 0;
#pragma unroll
                for (int r = rp; r < 16; r += 4, ++q) {
                    const uint32_t wi = widx[q];
                    const uint64_t v = ((uint64_t)rg[b * 80 + (wi & 127)] << 32)
                                     | (uint64_t)rg[b * 80 + ((wi >> 7) & 127)];
                    const uint32_t w16 = (uint32_t)(v >> (wi >> 14)) & 0xffffu;
                    const int bs = (int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
                    s += xhb[b][r] * (float)bs;
                    sx += xhb[b][r];
                }
            }
        }
        __syncthreads();
        if (live && nb0 < tk1) {
            uint32_t* rg = rings + g * 320;
            for (int b = 0; b < tkn2; ++b) {
                if (u < w) rg[b * 80 + u] = pfb[b][0];
                if (u + 64 < w) rg[b * 80 + u + 64] = pfb[b][1];
            }
        }
        if (threadIdx.x < 64 && nb0 < tk1) {
            const int j = threadIdx.x;
            if (j < 16 * tkn2) xhb[j >> 4][j & 15] = pfx;
        }
        __syncthreads();
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    sx += __shfl_xor_sync(0xffffffffu, sx, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    sx += __shfl_xor_sync(0xffffffffu, sx, 2);
    if (live && rp == 0) t[blockIdx.y * (tn * 16) + tn_i * 16 + c] = fmaf(KINV, s, KOFF * sx);
}

// stream-decode helpers shared with the MoE kernel (defined near inkwell_exl3_gemv_moe_s)
__device__ __forceinline__ uint32_t sgemv_mul16(uint32_t w);
template <int D> __device__ __forceinline__ uint32_t sgemv_win(uint32_t hi, uint32_t lo);
template <int KA, int H>
__device__ __forceinline__ void sgemv_run(const uint32_t* __restrict__ tb, int tn, int tn_i,
                                          int tk_lo, int tk_hi, const float* __restrict__ xs,
                                          float& a0, float& a1);

// ---- M=16-paired "C0c" core (port of opt/kbench/exl3_math2 r2::gemv_m16 MODE 0) ----
// 16 lanes per tile (sub = threadIdx.x & 15), 2-way in-block tk split
// (part = threadIdx.x >> 7), exact FP32 I2F: acc += x * (bytesum(h*C) - 510),
// out = KINV*acc + (KOFF + 510*KINV)*sx.
// The round-2 L1 ring rotation (rot bit q == raw bit (q + KA - 16) mod R) is
// folded into the lane bit offset, so trellis words on disk/device are NOT
// transformed and every other reader (ntok, dp4a, v2, dbg) stays valid.
// Killswitch: INKWELL_NO_PW16 -> launchers pass pw16=0 (shipped core).
#define PW16_CENT 0xFFFFFE02u   // -510: zero-centered bytesum bias
#define PW16_KOFFC -0.0018157958984375f  // KOFF + 510*KINV
// CM must reach the MAC asm from an opaque (device-memory) source or ptxas
// refolds h*0x83DCD12D into the slower 4-XMAD form (r2 "CM trick").
__device__ uint32_t ink_pw16_cm = 0x83DCD12Du;
__device__ __forceinline__ uint32_t pw16_h0(uint32_t U, uint32_t cm) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, cl;\n\t}"
        : "=r"(t) : "r"(U), "r"(cm));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, lo, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(U), "r"(cm), "r"(t));
    return p;
}
__device__ __forceinline__ uint32_t pw16_h1(uint32_t U, uint32_t cm) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, cl;\n\t}"
        : "=r"(t) : "r"(U), "r"(cm));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, hi, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(U), "r"(cm), "r"(t));
    return p;
}
__device__ __forceinline__ void pw16_pair(uint32_t U, float x1, float x0,
                                          float& a1, float& a0, uint32_t cm) {
    a1 = fmaf(x1, (float)(int)__dp4a(pw16_h1(U, cm), 0x01010101u, PW16_CENT), a1);
    a0 = fmaf(x0, (float)(int)__dp4a(pw16_h0(U, cm), 0x01010101u, PW16_CENT), a0);
}
template <int KA, int H>
__device__ __forceinline__ void pw16_run(const uint32_t* __restrict__ tb, int tn, int tn_i,
                                         int a_lo, int a_hi, const float* __restrict__ xs,
                                         float& a0, float& a1) {
    constexpr int NW = 8 * KA + 4 * H;
    const int sub = threadIdx.x & 15;
    const int hx_f = (sub & 1) ? 8 : 0;
    const uint32_t cm = ink_pw16_cm;
    if constexpr (KA == 2 && H == 0) {
        // rot word w = fsl(src[w], src[w-1], 18); lane sub needs rot sub & sub+1.
        // src[sub-1] is lane sub-1's S0 -> SHFL instead of a third LDG (LSU-bound).
        const size_t stride = (size_t)tn * 16;
        const uint32_t* p0 = tb + ((size_t)a_lo * tn + tn_i) * 16 + sub;
        const uint32_t* p1 = tb + ((size_t)a_lo * tn + tn_i) * 16 + ((sub + 1) & 15);
        const int prv = (threadIdx.x & 16) | ((sub + 15) & 15);   // lane sub-1 in own tile group
        const uint32_t tmask = 0xffffu << (threadIdx.x & 16);
#pragma unroll 2
        for (int a = a_lo; a < a_hi; ++a, p0 += stride, p1 += stride) {
            const uint32_t S0 = __ldg(p0), S1 = __ldg(p1);
            const uint32_t Sm = __shfl_sync(tmask, S0, prv);
            const uint32_t W0 = __funnelshift_l(S0, Sm, 18);
            const uint32_t W1 = __funnelshift_l(S1, S0, 18);
            const float4 xA = *(const float4*)&xs[a * 16 + hx_f + 0];
            const float4 xB = *(const float4*)&xs[a * 16 + hx_f + 4];
            pw16_pair(W0,                          xA.x, xA.z, a0, a0, cm);
            pw16_pair(__funnelshift_l(W1, W0, 2), xA.y, xA.w, a0, a0, cm);
            pw16_pair(__funnelshift_l(W1, W0, 4), xB.x, xB.z, a0, a0, cm);
            pw16_pair(__funnelshift_l(W1, W0, 6), xB.y, xB.w, a0, a0, cm);
            pw16_pair(__funnelshift_l(W1, W0, 8), xA.x, xA.z, a1, a1, cm);
            pw16_pair(__funnelshift_l(W1, W0,10), xA.y, xA.w, a1, a1, cm);
            pw16_pair(__funnelshift_l(W1, W0,12), xB.x, xB.z, a1, a1, cm);
            pw16_pair(__funnelshift_l(W1, W0,14), xB.y, xB.w, a1, a1, cm);
        }
        return;
    }
    if constexpr (KA == 4 && H == 0) {
        // rot word w = fsl(src[w], src[w-1], 20); lane sub needs rot 2sub..2sub+2.
        // src[2sub-1] is lane sub-1's S1 -> SHFL; 3 LDG like the rotated-ring version.
        const size_t stride = (size_t)tn * 32;
        const uint32_t* p0 = tb + ((size_t)a_lo * tn + tn_i) * 32 + ((2 * sub) & 31);
        const uint32_t* p1 = tb + ((size_t)a_lo * tn + tn_i) * 32 + ((2 * sub + 1) & 31);
        const uint32_t* p2 = tb + ((size_t)a_lo * tn + tn_i) * 32 + ((2 * sub + 2) & 31);
        const int prv = (threadIdx.x & 16) | ((sub + 15) & 15);
        const uint32_t tmask = 0xffffu << (threadIdx.x & 16);
        auto do_mac = [&](int j, uint32_t U, bool h1, const float* xv) {
            const int s = j >> 3, k = j & 7;
            const int r_loc = ((k & 2) ? 4 : 0) + 2 * s + (k & 1);
            float& acc = (k & 4) ? a1 : a0;
            const uint32_t p = h1 ? pw16_h1(U, cm) : pw16_h0(U, cm);
            acc = fmaf(xv[r_loc], (float)(int)__dp4a(p, 0x01010101u, PW16_CENT), acc);
        };
#pragma unroll 1
        for (int a = a_lo; a < a_hi; ++a, p0 += stride, p1 += stride, p2 += stride) {
            const uint32_t S0 = __ldg(p0), S1 = __ldg(p1), S2 = __ldg(p2);
            const uint32_t Sm = __shfl_sync(tmask, S1, prv);
            const uint32_t W[3] = {
                __funnelshift_l(S0, Sm, 20), __funnelshift_l(S1, S0, 20), __funnelshift_l(S2, S1, 20)};
            const float4 xA = *(const float4*)&xs[a * 16 + hx_f + 0];
            const float4 xB = *(const float4*)&xs[a * 16 + hx_f + 4];
            const float xv[8] = {xA.x, xA.y, xA.z, xA.w, xB.x, xB.y, xB.z, xB.w};
            do_mac(0,  W[0], true, xv); do_mac(4,  W[0], false, xv);
            do_mac(8,  W[1], true, xv); do_mac(12, W[1], false, xv);
            { uint32_t U = __funnelshift_l(W[1], W[0],  4); do_mac(1, U, true, xv); do_mac(5, U, false, xv); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  8); do_mac(2, U, true, xv); do_mac(6, U, false, xv); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 12); do_mac(3, U, true, xv); do_mac(7, U, false, xv); }
            { uint32_t U = __funnelshift_l(W[2], W[1],  4); do_mac(9, U, true, xv); do_mac(13,U, false, xv); }
            { uint32_t U = __funnelshift_l(W[2], W[1],  8); do_mac(10,U, true, xv); do_mac(14,U, false, xv); }
            { uint32_t U = __funnelshift_l(W[2], W[1], 12); do_mac(11,U, true, xv); do_mac(15,U, false, xv); }
        }
        return;
    }
    // generic K: lane base in raw bits = rot base + KA - 16 (mod R)
    int base_bit = sub * (16 * KA + 8 * H) + KA - 16;
    if (base_bit < 0) base_bit += 32 * NW;
    const int w0 = base_bit >> 5, o = base_bit & 31;
    constexpr int MAX_SPAN = 15 * KA + 8 * H + 16;
    constexpr int NA = (MAX_SPAN + 31) / 32;
    const uint32_t stride_b = (uint32_t)(tn * NW * 4);
    const char* qb = (const char*)(tb + ((size_t)a_lo * tn + tn_i) * NW);
    int wi[NA + 1];
#pragma unroll
    for (int m = 0; m <= NA; ++m) { int t = w0 + m; wi[m] = (t >= NW) ? (t - NW) : t; }
#pragma unroll 1
    for (int a = a_lo; a < a_hi; ++a, qb += stride_b) {
        const uint32_t* ptr = (const uint32_t*)qb;
        uint32_t W[NA];
        {
            uint32_t rprev = __ldg(ptr + wi[0]);
#pragma unroll
            for (int m = 0; m < NA; ++m) {
                const uint32_t rcur = __ldg(ptr + wi[m + 1]);
                W[m] = __funnelshift_l(rcur, rprev, o);
                rprev = rcur;
            }
        }
        const float4 xA = *(const float4*)&xs[a * 16 + hx_f + 0];
        const float4 xB = *(const float4*)&xs[a * 16 + hx_f + 4];
        const float xv[8] = {xA.x, xA.y, xA.z, xA.w, xB.x, xB.y, xB.z, xB.w};
        auto get_u = [&](int d) -> uint32_t {
            const int i = d >> 5, bi = d & 31;
            if (bi == 0) return W[i];
            if (bi <= 16) return W[i] << bi;
            return __funnelshift_l(W[i + 1], W[i], bi);
        };
        auto do_mac_j = [&](int j, uint32_t U, bool h1, float& t0, float& t1) {
            const int s = j >> 3, k = j & 7;
            const int r_loc = ((k & 2) ? 4 : 0) + 2 * s + (k & 1);
            float& acc = (k & 4) ? t1 : t0;
            const uint32_t p = h1 ? pw16_h1(U, cm) : pw16_h0(U, cm);
            acc = fmaf(xv[r_loc], (float)(int)__dp4a(p, 0x01010101u, PW16_CENT), acc);
        };
        if constexpr (KA == 4 && H == 0) {
            do_mac_j(0,  W[0], true, a0, a1); do_mac_j(4,  W[0], false, a0, a1);
            do_mac_j(8,  W[1], true, a0, a1); do_mac_j(12, W[1], false, a0, a1);
            { uint32_t U = __funnelshift_l(W[1], W[0],  4); do_mac_j(1, U, true, a0, a1); do_mac_j(5, U, false, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  8); do_mac_j(2, U, true, a0, a1); do_mac_j(6, U, false, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 12); do_mac_j(3, U, true, a0, a1); do_mac_j(7, U, false, a0, a1); }
            { uint32_t U = __funnelshift_l(W[2], W[1],  4); do_mac_j(9, U, true, a0, a1); do_mac_j(13,U, false, a0, a1); }
            { uint32_t U = __funnelshift_l(W[2], W[1],  8); do_mac_j(10,U, true, a0, a1); do_mac_j(14,U, false, a0, a1); }
            { uint32_t U = __funnelshift_l(W[2], W[1], 12); do_mac_j(11,U, true, a0, a1); do_mac_j(15,U, false, a0, a1); }
        } else if constexpr (KA == 1 && H == 1) {
            do_mac_j(0, W[0], true, a0, a1);
            { uint32_t U = __funnelshift_l(W[1], W[0], 2); do_mac_j(1, U, true, a0, a1); do_mac_j(12, U, false, a0, a1); do_mac_j(2, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 5); do_mac_j(3, U, true, a0, a1); do_mac_j(14, U, false, a0, a1); do_mac_j(4, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 8); do_mac_j(5, U, true, a0, a1); do_mac_j(6, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],11); do_mac_j(7, U, true, a0, a1); do_mac_j(8, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],14); do_mac_j(9, U, true, a0, a1); do_mac_j(10,U + U, true, a0, a1); }
            do_mac_j(11, __funnelshift_l(W[1], W[0], 17), true, a0, a1);
            do_mac_j(13, __funnelshift_l(W[1], W[0], 20), true, a0, a1);
            do_mac_j(15, __funnelshift_l(W[1], W[0], 23), true, a0, a1);
        } else if constexpr (KA == 2 && H == 1) {
            do_mac_j(0, W[0], true, a0, a1);
            { uint32_t U = __funnelshift_l(W[1], W[0],  2); do_mac_j(7,  U, false, a0, a1); do_mac_j(1, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  4); do_mac_j(8,  U, false, a0, a1); do_mac_j(2, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  7); do_mac_j(9,  U, false, a0, a1); do_mac_j(3, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  9); do_mac_j(10, U, false, a0, a1); do_mac_j(4, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 12); do_mac_j(11, U, false, a0, a1); do_mac_j(5, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 14); do_mac_j(12, U, false, a0, a1); do_mac_j(6, U + U, true, a0, a1); }
            do_mac_j(13, W[1] + W[1], true, a0, a1);
            do_mac_j(14, W[1] << 3, true, a0, a1);
            do_mac_j(15, W[1] << 6, true, a0, a1);
        } else if constexpr (KA == 3 && H == 0) {
            do_mac_j(0, W[0], true, a0, a1);
            { uint32_t U = __funnelshift_l(W[1], W[0],  2); do_mac_j(6,  U, false, a0, a1); do_mac_j(1, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  5); do_mac_j(7,  U, false, a0, a1); do_mac_j(2, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0],  8); do_mac_j(8,  U, false, a0, a1); do_mac_j(3, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 11); do_mac_j(9,  U, false, a0, a1); do_mac_j(4, U + U, true, a0, a1); }
            { uint32_t U = __funnelshift_l(W[1], W[0], 14); do_mac_j(10, U, false, a0, a1); do_mac_j(5, U + U, true, a0, a1); }
            do_mac_j(11, W[1] + W[1], true, a0, a1);
            do_mac_j(12, W[1] << 4, true, a0, a1);
            do_mac_j(13, W[1] << 7, true, a0, a1);
            do_mac_j(14, W[1] << 10, true, a0, a1);
            do_mac_j(15, W[1] << 13, true, a0, a1);
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int D = j * KA + H * ((j + 1) >> 1);
                if ((D & 15) == 0) do_mac_j(j, W[D >> 5], (D & 16) == 0, a0, a1);
                else do_mac_j(j, get_u(D), true, a0, a1);
            }
        }
    }
}

// pw16 per-block epilogue: xor-merge even/odd lane pairs, cross-part reduce
// through `red` (must point at scratch smem >= 128 floats; caller guarantees all
// reads of it are done), then `writer` lanes (part 0, even sub, live tile) hold
// final a0/a1 for output columns tn_i*16 + (sub>>1) [+8].
__device__ __forceinline__ void pw16_reduce(int part, int t_in_blk, int sub, bool live,
                                            float* red, float& a0, float& a1) {
    a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
    a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    __syncthreads();
    if (part == 1 && (sub & 1) == 0 && live) {
        red[t_in_blk * 16 + (sub >> 1)] = a0;
        red[t_in_blk * 16 + (sub >> 1) + 8] = a1;
    }
    __syncthreads();
    if (part == 0 && (sub & 1) == 0 && live) {
        a0 += red[t_in_blk * 16 + (sub >> 1)];
        a1 += red[t_in_blk * 16 + (sub >> 1) + 8];
    }
}

// compile-time enable mask: bit (ka*2+hk) -> use pw16 for that K
#ifndef PW16_KMASK
#define PW16_KMASK ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 8))   // K=2.0/2.5/3.0/4.0 (per-K kbench)
#endif
// opt-in INKWELL_PW16=1 (default off: 16-pair A/B 11/16, below the 12/16 bar)
static inline int ink_pw16_mask() {
    static int m = -1;
    if (m < 0) m = (getenv("INKWELL_PW16") && !getenv("INKWELL_NO_PW16")) ? PW16_KMASK : 0;   // default OFF until E2E A/B passes
    return m;
}
// "stream-decode" dense single-token gemv (same design as inkwell_exl3_gemv_moe_s):
// one warp per 16-wide tile column, 256 threads = 8 warps = 8 tile columns;
// lane t owns 8 consecutive ring windows at compile-time offsets, 3 __ldg words
// per tile, no smem ring, no __syncthreads in the loop; Σx over the block's
// k-range precomputed in the prologue; out = KINV*acc + KOFF*Σx.
// grid (ceil(tn/8), nky): block y owns tk chunk [y*tkc, min(y*tkc+tkc, tk)) and
// writes its PARTIAL to t[y*(tn*16) + col] with a plain store (empty chunks
// write 0); inkwell_exl3_had_out sums the nky partials.
#define SGEMV_DENSE_MAXK 3072
__global__ void __launch_bounds__(256) inkwell_exl3_gemv_s(
                                       const uint32_t* __restrict__ trel,
                                       const float* __restrict__ xh,
                                       float* __restrict__ t,
                                       int tk, int tn, int ka, int halfk, int pw16) {
    const int tkc = (tk + gridDim.y - 1) / gridDim.y;
    const int tk_lo = blockIdx.y * tkc;
    const int tk_hi = min(tk_lo + tkc, tk);
    const int kk = ka * 2 + halfk;
    const bool pw = (pw16 >> kk) & 1;
    __shared__ __align__(16) float xs[SGEMV_DENSE_MAXK];
    __shared__ float red[8];
    // prologue: stage this block's k-range of x and its Σx (pw16 permutes lanes)
    float sxl = 0.f;
    for (int e = tk_lo * 16 + threadIdx.x; e < tk_hi * 16; e += 256) {
        const float v = xh[e];
        sxl += v;
        if (pw) { const int r = e & 15; xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = v; }
        else xs[e] = v;
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) sxl += __shfl_xor_sync(0xffffffffu, sxl, s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sxl;
    __syncthreads();
    float sx = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx += red[i];
    if (pw) {
        const int part = threadIdx.x >> 7;
        const int t_in_blk = (threadIdx.x & 127) >> 4;
        const int sub = threadIdx.x & 15;
        const int tn_i = blockIdx.x * 8 + t_in_blk;
        const int span = tk_hi - tk_lo;
        const int a_lo = tk_lo + ((part * span) >> 1);
        const int a_hi = tk_lo + (((part + 1) * span) >> 1);
        float a0 = 0.f, a1 = 0.f;
        const bool live = tn_i < tn;
        if (live) {
            switch (kk) {
                case 3:  pw16_run<1, 1>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 4:  pw16_run<2, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 5:  pw16_run<2, 1>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 6:  pw16_run<3, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 7:  pw16_run<3, 1>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 8:  pw16_run<4, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 10: pw16_run<5, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                default: __trap();
            }
        }
        pw16_reduce(part, t_in_blk, sub, live, xs, a0, a1);
        if (part == 0 && (sub & 1) == 0 && live) {
            const float KINV = __half2float(__ushort_as_half(0x1eee));
            const float off = PW16_KOFFC * sx;
            float* dst = &t[(size_t)blockIdx.y * (tn * 16) + tn_i * 16 + (sub >> 1)];
            dst[0] = fmaf(KINV, a0, off); dst[8] = fmaf(KINV, a1, off);
        }
        return;
    }
    const int tn_i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tn_i >= tn) return;
    float a0 = 0.f, a1 = 0.f;
    switch (kk) {
        case 3:  sgemv_run<1, 1>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 4:  sgemv_run<2, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 5:  sgemv_run<2, 1>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 6:  sgemv_run<3, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 7:  sgemv_run<3, 1>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 8:  sgemv_run<4, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 10: sgemv_run<5, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        default: __trap();
    }
    a0 += __shfl_xor_sync(0xffffffffu, a0, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
    a0 += __shfl_xor_sync(0xffffffffu, a0, 2); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    const int lane = threadIdx.x & 31;
    if ((lane & 3) == 0) {
        const float KINV = __half2float(__ushort_as_half(0x1eee));
        const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
        const float off = KOFF * sx;   // chunk-local Σx belongs to every column once
        float* dst = &t[(size_t)blockIdx.y * (tn * 16) + tn_i * 16 + (lane >> 2)];
        dst[0] = fmaf(KINV, a0, off); dst[8] = fmaf(KINV, a1, off);
    }
}

// Fused variant of the inkwell_exl3_gemv_s chain: had_in stays a separate
// launch (xh is shared across the mz set), but the gemv absorbs memset +
// had_out: partials go to t[y*n + col] as before and the LAST block for
// strip bx (atomic ticket on ctr[bx], pre-zeroed, self-resetting) sums the
// nky partials in the same ascending-z order as inkwell_exl3_had_out, then
// runs the same 128-butterfly + svh multiply into y. Bit-identical output.
// (Folding had_in too was tried first: ~24x redundant chunk transforms on
// the nky==1 unembed made it a net loss.)
__global__ void __launch_bounds__(256) inkwell_exl3_gemv_sf(
                                       const uint32_t* __restrict__ trel,
                                       const float* __restrict__ xh,
                                       const uint16_t* __restrict__ svh,
                                       float* __restrict__ y,
                                       float* __restrict__ t,
                                       int* __restrict__ ctr,
                                       int tk, int tn, int ka, int halfk, int pw16) {
    const int n = tn * 16;
    const int tkc = (tk + gridDim.y - 1) / gridDim.y;
    const int tk_lo = blockIdx.y * tkc;
    const int tk_hi = min(tk_lo + tkc, tk);
    const int kk = ka * 2 + halfk;
    const bool pw = (pw16 >> kk) & 1;
    __shared__ __align__(16) float xs[SGEMV_DENSE_MAXK];
    __shared__ float red[8];
    __shared__ unsigned int ticket;
    __shared__ float strip[128];
    const int e_lo = tk_lo * 16, e_hi = tk_hi * 16;
    // prologue: stage this block's k-range of x and its Sigma_x (pw16 permutes lanes)
    float sxl = 0.f;
    for (int e = e_lo + threadIdx.x; e < e_hi; e += 256) {
        const float v = xh[e];
        sxl += v;
        if (pw) { const int r = e & 15; xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = v; }
        else xs[e] = v;
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) sxl += __shfl_xor_sync(0xffffffffu, sxl, s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sxl;
    __syncthreads();
    float sx = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx += red[i];
    const int part = threadIdx.x >> 7;
    const int t_in_blk = (threadIdx.x & 127) >> 4;
    const int sub = threadIdx.x & 15;
    const int tn_i = blockIdx.x * 8 + (pw ? t_in_blk : (threadIdx.x >> 5));
    const bool live = tn_i < tn;
    float a0 = 0.f, a1 = 0.f;
    if (pw) {
        const int span = tk_hi - tk_lo;
        const int a_lo = tk_lo + ((part * span) >> 1);
        const int a_hi = tk_lo + (((part + 1) * span) >> 1);
        if (live && span > 0) {
            switch (kk) {
                case 3:  pw16_run<1, 1>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 4:  pw16_run<2, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 5:  pw16_run<2, 1>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 6:  pw16_run<3, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 7:  pw16_run<3, 1>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 8:  pw16_run<4, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 10: pw16_run<5, 0>(trel, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                default: __trap();
            }
        }
        pw16_reduce(part, t_in_blk, sub, live, xs, a0, a1);
    } else if (tk_hi > tk_lo && tn_i < tn) {
        switch (kk) {
            case 3:  sgemv_run<1, 1>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 4:  sgemv_run<2, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 5:  sgemv_run<2, 1>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 6:  sgemv_run<3, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 7:  sgemv_run<3, 1>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 8:  sgemv_run<4, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 10: sgemv_run<5, 0>(trel, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            default: __trap();
        }
    }
    if (!pw) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, 1); a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
        a0 += __shfl_xor_sync(0xffffffffu, a0, 2); a1 += __shfl_xor_sync(0xffffffffu, a1, 2);
    }
    const int lane = threadIdx.x & 31;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = pw ? PW16_KOFFC
                          : 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    const float off = KOFF * sx;
    const bool wr = live && (pw ? (part == 0 && (sub & 1) == 0) : ((lane & 3) == 0));
    const int scol = (pw ? t_in_blk : (threadIdx.x >> 5)) * 16 + (pw ? (sub >> 1) : (lane >> 2));
    if (gridDim.y == 1) {
        // single-k-block: strip lives in smem only; zero first so columns
        // >= n contribute 0 to the butterfly exactly like had_out's v=0.
        if (threadIdx.x < 128) strip[threadIdx.x] = 0.f;
        __syncthreads();
        if (wr) {
            strip[scol]     = fmaf(KINV, a0, off);
            strip[scol + 8] = fmaf(KINV, a1, off);
        }
        __syncthreads();
    } else {
        if (wr) {
            float* dst = &t[(size_t)blockIdx.y * n + blockIdx.x * 128 + scol];
            dst[0] = fmaf(KINV, a0, off); dst[8] = fmaf(KINV, a1, off);
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) ticket = atomicAdd(&ctr[blockIdx.x], 1u);
        __syncthreads();
        if (ticket != gridDim.y - 1) return;   // not last: done
        __threadfence();
        if (threadIdx.x == 0) ctr[blockIdx.x] = 0;   // self-reset for next launch
        const int c = blockIdx.x * 128 + threadIdx.x;
        if (threadIdx.x < 128) {
            float v = 0.f;
            if (c < n) for (int z = 0; z < (int)gridDim.y; ++z) v += __ldcg(&t[(size_t)z * n + c]);   // L2: written by other blocks
            strip[threadIdx.x] = v;
        }
        __syncthreads();
    }
    // shared epilogue: strip[128] = partial sums -> H128 * svh -> y (had_out
    // clone). Only the last block per strip (or every block when nky==1)
    // reaches this; barriers stay uniform across all 256 threads.
    {
        const int c = blockIdx.x * 128 + threadIdx.x;
        const bool live = threadIdx.x < 128;
        for (int s = 1; s < 128; s <<= 1) {
            const float u = !live ? 0.f
                          : (threadIdx.x & s) ? strip[threadIdx.x - s] - strip[threadIdx.x]
                                              : strip[threadIdx.x] + strip[threadIdx.x + s];
            __syncthreads();
            if (live) strip[threadIdx.x] = u;
            __syncthreads();
        }
        if (live && c < n) y[c] = strip[threadIdx.x] * 0.08838834764831845f
                                * __half2float(__ushort_as_half(svh[c]));
    }
}

// Batched-token GEMV (spec-decode verify): one weight stream feeds NTOK=8
// activation sets. t layout: [nky][ntok][n]; xh: [ntok][tk*16].
__global__ void inkwell_exl3_gemv_dp4a_ntok(const uint32_t* __restrict__ trel,
                                          const float* __restrict__ xh,
                                          float* __restrict__ t,
                                          int tk, int tn, int ka, int halfk) {
    const int NTOK = 8;
    const int g = threadIdx.x >> 6;
    const int u = threadIdx.x & 63;
    const int c = u >> 2, rp = u & 3;
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk);
    __shared__ float xhb[8][8][16];   // [tk-batch][token][elem]
    __shared__ uint32_t rings[4 * 4 * 80];
    const bool live = tn_i < tn;
    const int tkc = (tk + gridDim.y - 1) / gridDim.y;
    const int tk0 = blockIdx.y * tkc;
    const int tk1 = min(tk0 + tkc, tk);
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    float s[8], sx[8];
#pragma unroll
    for (int z = 0; z < 8; ++z) { s[z] = 0.f; sx[z] = 0.f; }
    uint32_t widx[4];
    {
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int nw = r_bits >> 5;
        int q = 0;
        for (int r = rp; r < 16; r += 4, ++q) {
            const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                        | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
            const int s_i = halfk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
            const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
            const int i0 = b0 >> 5;
            const int s0 = 48 - (b0 & 31);
            const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
            widx[q] = (uint32_t)(i0 | (i1 << 7) | (s0 << 14));
        }
    }
    // xhb fill helper: all 256 threads cover [tkn][8][16]
    auto fill_xh = [&](int base, int tkn) {
        for (int j = threadIdx.x; j < 16 * tkn * NTOK; j += 256) {
            const int tt = j >> 6;
            const int rem = j & 63;
            xhb[rem >> 4][tt][rem & 15] =
                xh[(size_t)tt * tk * 16 + (size_t)(base + (rem >> 4)) * 16 + (rem & 15)];
        }
    };
    {
        const int tkn0 = min(4, tk1 - tk0);
        fill_xh(tk0, tkn0);
        if (live) {
            uint32_t* rg = rings + g * 320;
            for (int b = 0; b < tkn0; ++b) {
                const uint32_t* ring = trel + (size_t)((tk0 + b) * tn + tn_i) * (size_t)w;
                if (u < w) rg[b * 80 + u] = ring[u];
                if (u + 64 < w) rg[b * 80 + u + 64] = ring[u + 64];
            }
        }
    }
    __syncthreads();
    for (int tkb0 = tk0; tkb0 < tk1; tkb0 += 4) {
        const int tkn = min(4, tk1 - tkb0);
        const int nb0 = tkb0 + 4;
        const int tkn2 = min(4, tk1 - nb0);
        uint32_t pfb[4][2];
        if (live && nb0 < tk1) {
            for (int b = 0; b < tkn2; ++b) {
                const uint32_t* ring = trel + (size_t)((nb0 + b) * tn + tn_i) * (size_t)w;
                if (u < w) pfb[b][0] = ring[u];
                if (u + 64 < w) pfb[b][1] = ring[u + 64];
            }
        }
        float pfx[8];
        if (nb0 < tk1) {
            const int b = u >> 4, e = u & 15;
#pragma unroll
            for (int tt = 0; tt < 8; ++tt)
                pfx[tt] = (b < tkn2) ? xh[(size_t)tt * tk * 16 + (size_t)(nb0 + b) * 16 + e] : 0.f;
        }
        if (live) {
            const uint32_t* rg = rings + g * 320;
            for (int b = 0; b < tkn; ++b) {
                int q = 0;
#pragma unroll
                for (int r = rp; r < 16; r += 4, ++q) {
                    const uint32_t wi = widx[q];
                    const uint64_t v = ((uint64_t)rg[b * 80 + (wi & 127)] << 32)
                                     | (uint64_t)rg[b * 80 + ((wi >> 7) & 127)];
                    const uint32_t w16 = (uint32_t)(v >> (wi >> 14)) & 0xffffu;
                    const int bs = (int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
                    const float fbs = (float)bs;
#pragma unroll
                    for (int tt = 0; tt < 8; ++tt) {
                        const float xr = xhb[b][tt][r];
                        s[tt] += xr * fbs;
                        sx[tt] += xr;
                    }
                }
            }
        }
        __syncthreads();
        if (live && nb0 < tk1) {
            uint32_t* rg = rings + g * 320;
            for (int b = 0; b < tkn2; ++b) {
                if (u < w) rg[b * 80 + u] = pfb[b][0];
                if (u + 64 < w) rg[b * 80 + u + 64] = pfb[b][1];
            }
        }
        if (nb0 < tk1) {
            const int b = u >> 4, e = u & 15;
#pragma unroll
            for (int tt = 0; tt < 8; ++tt)
                if (b < tkn2) xhb[b][tt][e] = pfx[tt];
        }
        __syncthreads();
    }
#pragma unroll
    for (int tt = 0; tt < 8; ++tt) {
        s[tt] += __shfl_xor_sync(0xffffffffu, s[tt], 1);
        sx[tt] += __shfl_xor_sync(0xffffffffu, sx[tt], 1);
        s[tt] += __shfl_xor_sync(0xffffffffu, s[tt], 2);
        sx[tt] += __shfl_xor_sync(0xffffffffu, sx[tt], 2);
        if (live && rp == 0)
            t[((size_t)blockIdx.y * 8 + tt) * (size_t)(tn * 16) + tn_i * 16 + c] =
                fmaf(KINV, s[tt], KOFF * sx[tt]);
    }
}

extern "C" void ink_launch_exl3_gemv_ntok(const void* trel, const void* suh, const void* svh,
                                          const float* x, float* y,
                                          long long k, long long n, long long ka, long long halfk,
                                          float* scratch, void* stream) {
    const int NTOK = 8;
    const int tk = (int)(k / 16), tn = (int)(n / 16);
    { static int hm = -1; if (hm < 0) { hm = getenv("INKWELL_NO_HADMERGE") ? 0 : 1; }
      if (hm) {
          HadNArgs a; a.n = NTOK; a.k = (int)k;
          for (int tt = 0; tt < NTOK; ++tt) {
              a.x[tt] = x + (size_t)tt * k;
              a.suh[tt] = (const uint16_t*)suh;
              a.xh[tt] = scratch + (size_t)tt * k;
          }
          inkwell_exl3_had_in_n<<<dim3((unsigned)(k / 128), NTOK), 128, 512, (cudaStream_t)stream>>>(a);
      } else
          for (int tt = 0; tt < NTOK; ++tt)
              inkwell_exl3_had_in<<<(unsigned)(k / 128), 128, 512, (cudaStream_t)stream>>>(
                  x + (size_t)tt * k, (const uint16_t*)suh, scratch + (size_t)tt * k, (int)k); }
    const int nb = (tn + 3) / 4;
    int nky = 1;
    { static int tgt = -1; if (tgt < 0) { const char* e = getenv("INKWELL_GEMV_NKY_TGT"); tgt = e ? atoi(e) : 512; }
      while (nb * nky < tgt && nky < 8 && nky * 16 <= tk) nky *= 2; }
    if (nky > 1) cudaMemsetAsync(scratch + (size_t)NTOK * k, 0, (size_t)nky * NTOK * n * 4, (cudaStream_t)stream);
    dim3 gsz((unsigned)nb, (unsigned)nky, 1);
    inkwell_exl3_gemv_dp4a_ntok<<<gsz, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)trel, scratch, scratch + (size_t)NTOK * k, tk, tn, (int)ka, (int)halfk);
    for (int tt = 0; tt < NTOK; ++tt)
        inkwell_exl3_had_out<<<(unsigned)(n / 128), 128, 512, (cudaStream_t)stream>>>(
            scratch + (size_t)NTOK * k + (size_t)tt * n, (const uint16_t*)svh,
            y + (size_t)tt * n, (int)n, nky, (int)(NTOK * n));
}

__global__ void inkwell_exl3_gemv_dp4a_dbg(const uint32_t* __restrict__ trel,
                                          const float* __restrict__ xh,
                                          float* __restrict__ out_s, float* __restrict__ out_sx,
                                          int tk, int tn, int ka, int halfk) {
    __shared__ float xhs[16];
    const int tn_i = blockIdx.x;
    const int c = threadIdx.x;
    const int w = r_bits_words(ka, halfk);
    float s = 0.f, sx = 0.f;
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        xhs[c] = xh[tk_i * 16 + c];
        __syncthreads();
        const uint32_t* ring = trel + (size_t)(tk_i * tn + tn_i) * (size_t)w;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                        | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
            const uint32_t win = exl3_window(ring, i, ka, halfk);
            const int bs = (int)__dp4a(win * 0x83DCD12Du, 0x01010101u, 0u);
            s += xhs[r] * (float)bs;
            sx += xhs[r];
        }
        __syncthreads();
    }
    out_s[tn_i * 16 + c] = s;
    out_sx[tn_i * 16 + c] = sx;
}
extern "C" void ink_launch_exl3_dp4a_dbg(const void* trel, const float* xh,
    float* s, float* sx, long long tk, long long tn, long long ka, long long halfk, void* stream) {
    inkwell_exl3_gemv_dp4a_dbg<<<(unsigned)tn, 16, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)trel, xh, s, sx, (int)tk, (int)tn, (int)ka, (int)halfk);
}

// everything after had_in (xh already in scratch[0..k)) — factored out so the
// mz path can merge the per-projection had_ins upstream of the stream fork.
static void exl3_gemv_tail(const void* trel, const void* svh, float* y,
                           long long k, long long n, long long ka, long long halfk,
                           float* scratch, void* stream) {
    const int tk = (int)(k / 16), tn = (int)(n / 16);
    { static int no_s = -1; if (no_s < 0) { const char* e = getenv("INKWELL_NO_SGEMV_DENSE"); no_s = e ? atoi(e) : 0; }
      if (!no_s && k <= SGEMV_DENSE_MAXK) {
        const int nb = (tn + 7) / 8;
        int nky = 1;
        { static int tgt = -1; if (tgt < 0) { const char* e = getenv("INKWELL_SGEMV_DENSE_TGT"); tgt = e ? atoi(e) : 32; }
          while (nb * nky < tgt && nky < 8 && nky * 2 <= tk) nky *= 2; }
        // fused path: gemv_sf absorbs memset + had_out.
        // killswitch INKWELL_NO_SGEMV_FUSE -> the 3-launch chain below.
        // ctr[nb] lives at scratch[k + 8n .. +512) (alloc extended +512,
        // pre-zeroed at load); nky==1 uses no counter -> unbounded nb ok.
        { static int no_f = -1; if (no_f < 0) { const char* e = getenv("INKWELL_NO_SGEMV_FUSE"); no_f = e ? atoi(e) : 0; }
          if (!no_f && (n % 16) == 0 && (nky == 1 || nb <= 512)) {
            dim3 gsz((unsigned)nb, (unsigned)nky, 1);
            inkwell_exl3_gemv_sf<<<gsz, 256, 0, (cudaStream_t)stream>>>(
                (const uint32_t*)trel, scratch /* xh */, (const uint16_t*)svh,
                y, scratch + k, (int*)(scratch + k + 8 * n), tk, tn, (int)ka, (int)halfk, ink_pw16_mask());
            return;
          } }
        if (nky > 1) cudaMemsetAsync(scratch + k, 0, (size_t)nky * n * 4, (cudaStream_t)stream);
        dim3 gsz((unsigned)nb, (unsigned)nky, 1);
        inkwell_exl3_gemv_s<<<gsz, 256, 0, (cudaStream_t)stream>>>(
            (const uint32_t*)trel, scratch, scratch + k, tk, tn, (int)ka, (int)halfk, ink_pw16_mask());
        inkwell_exl3_had_out<<<(unsigned)(n / 128), 128, 512, (cudaStream_t)stream>>>(
            scratch + k, (const uint16_t*)svh, y, (int)n, nky, (int)n);
        return;
      } }
    const int nb = (tn + 3) / 4;
    // aim for ~256 blocks: split K when tn is small
    int nky = 1;
    { static int tgt = -1; if (tgt < 0) { const char* e = getenv("INKWELL_GEMV_NKY_TGT"); tgt = e ? atoi(e) : 512; }
      while (nb * nky < tgt && nky < 8 && nky * 16 <= tk) nky *= 2; }
    if (nky > 1) cudaMemsetAsync(scratch + k, 0, nky * n * 4, (cudaStream_t)stream);
    dim3 gsz((unsigned)nb, (unsigned)nky, 1);
    { static int x8 = -1; if (x8 < 0) { const char* e = getenv("INKWELL_EXL3_X8"); x8 = e ? atoi(e) : 0; }
      if (x8 && tk <= 160) inkwell_exl3_gemv_dp4a_x8<<<gsz, 256, 0, (cudaStream_t)stream>>>(
              (const uint32_t*)trel, scratch, scratch + k, tk, tn, (int)ka, (int)halfk);
      else inkwell_exl3_gemv_dp4a<<<gsz, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)trel, scratch, scratch + k, tk, tn, (int)ka, (int)halfk); }
    inkwell_exl3_had_out<<<(unsigned)(n / 128), 128, 512, (cudaStream_t)stream>>>(
        scratch + k, (const uint16_t*)svh, y, (int)n, nky, (int)n);
}

extern "C" void ink_launch_exl3_gemv(const void* trel, const void* suh, const void* svh,
                                     const float* x, float* y,
                                     long long k, long long n, long long ka, long long halfk,
                                     float* scratch, void* stream) {
    inkwell_exl3_had_in<<<(unsigned)(k / 128), 128, 512, (cudaStream_t)stream>>>(
        x, (const uint16_t*)suh, scratch /* xh */, (int)k);
    exl3_gemv_tail(trel, svh, y, k, n, ka, halfk, scratch, stream);
}

__global__ void inkwell_exl3_dbg_dec(const uint32_t* __restrict__ trel,
                                     float* __restrict__ out, int ka, int halfk) {
    const int i = threadIdx.x;
    out[i] = exl3_mul1(exl3_window(trel, i, ka, halfk));
}
extern "C" void ink_launch_exl3_dbg(const void* trel, float* out, long long ka, long long halfk, void* stream) {
    inkwell_exl3_dbg_dec<<<1, 256, 0, (cudaStream_t)stream>>>((const uint32_t*)trel, out, (int)ka, (int)halfk);
}
extern "C" void ink_launch_exl3_dbg_had(const float* x, const void* suh, float* out, long long k, void* stream) {
    inkwell_exl3_had_in<<<(unsigned)(k / 128), 128, 512, (cudaStream_t)stream>>>(
        x, (const uint16_t*)suh, out, (int)k);
}

// ===================== EXL3 MoE (per-slot expert descriptors) =====================
// desc[expert] = 4 u64: {trellis ptr, suh ptr, svh ptr, ka | halfk<<8}
// All pointers are pinned host-mapped (zero-copy readable on sm_61).

// 128-dim Hadamard via 4 in-register elems/lane + 5 warp-shuffle levels.
// Element e = lane*4+i inside the warp's 128-block; same butterfly order as
// the old smem version but with zero __syncthreads and no smem traffic.
__device__ __forceinline__ void had128_warp(float* __restrict__ v, int ebase) {
#pragma unroll
    for (int s = 1; s < 4; s <<= 1) {
        float u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            u[i] = (i & s) ? v[i ^ s] - v[i] : v[i] + v[i ^ s];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = u[i];
    }
#pragma unroll
    for (int s = 4; s < 128; s <<= 1) {
        const int mask = s >> 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], mask);
            v[i] = ((ebase + i) & s) ? o - v[i] : v[i] + o;
        }
    }
}

__global__ void inkwell_exl3_had_in_moe(const uint64_t* __restrict__ descs,
                                        const uint64_t* __restrict__ descs2,
                                        const int* __restrict__ ids,
                                        const float* __restrict__ x,
                                        float* __restrict__ xh, float* __restrict__ xh2,
                                        int xh2s, int k,
                                        const int* __restrict__ missf, int skipmode,
                                        const int* __restrict__ euni, const int* __restrict__ etok) {
    const int slot = blockIdx.y;
    const int sfl = euni ? euni[slot] : slot;
    if (skipmode && ((missf[sfl] != 0) == (skipmode == 1))) return;
    const uint64_t* dd = blockIdx.z ? descs2 : descs;
    float* xho = blockIdx.z ? xh2 + slot * xh2s : xh + slot * k;
    const uint16_t* suh = (const uint16_t*)(uintptr_t)dd[ids[slot] * 4 + 1];
    const size_t xoff = etok ? (size_t)etok[slot] * k : (descs2 ? 0 : (size_t)slot * k);
    const int lane = threadIdx.x & 31;
    const int base = blockIdx.x * 512 + (threadIdx.x >> 5) * 128;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = base + lane * 4 + i;
        v[i] = (e < k) ? x[xoff + e] * __half2float(__ushort_as_half(suh[e])) : 0.f;
    }
    had128_warp(v, lane * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = base + lane * 4 + i;
        if (e < k) xho[e] = v[i] * 0.08838834764831845f;
    }
}

// grid (tn, nslots): block decodes tile tn_i of expert ids[slot]; acc -> t[slot][n]
__global__ void inkwell_exl3_gemv_moe(const uint64_t* __restrict__ descs,
                                      const uint64_t* __restrict__ descs2,
                                      const int* __restrict__ ids,
                                      const float* __restrict__ xh,
                                      const float* __restrict__ xh2, int xh2s,
                                      float* __restrict__ t, int tslot, int tzoff,
                                      int tk, int tn, int k, int n, int nky,
                                      const int* __restrict__ missf, int skipmode) {
    const int slot = blockIdx.y;
    // skipmode 1: hits only; 2: misses only; 0: all slots
    if (skipmode && ((missf[slot] != 0) == (skipmode == 1))) return;
    const int nsel = descs2 ? 2 : 1;             // fused gate+up pairs vs single
    const int ky = blockIdx.z / nsel;            // k-range index
    const int zsel = blockIdx.z % nsel;
    const uint64_t* d = (zsel ? descs2 : descs) + ids[slot] * 4;
    const float* xhb_g = zsel ? xh2 + slot * xh2s : xh + slot * k;
    // this block's batch-row range within [0, tk)
    const int tk_lo = (ky * tk) / nky;
    const int tk_hi = ((ky + 1) * tk) / nky;
    const uint32_t* tile_base = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int g = threadIdx.x >> 6;      // 4 tiles per block, 64 threads/tile
    const int u = threadIdx.x & 63;      // 4 threads per column (r split 4-way)
    const int c = u >> 2, rp = u & 3;
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk); // ≤32 u32 per tile
    __shared__ uint32_t rings[4 * 8 * 32]; // 4 tiles × 8 tk-batch × ≤32 words
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    float s = 0.f, sx = 0.f;
    const bool live = tn_i < tn;
    __shared__ float xhb[8][16];
    // hoisted window indices: (i0, i1, s0) per q — (c,r) fixed across all batches
    uint32_t widx[4];
    {
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int nw = r_bits >> 5;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = rp + 4 * q;
            const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                        | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
            const int s_i = halfk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
            const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
            const int i0 = b0 >> 5;
            const int s0 = 48 - (b0 & 31);
            const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
            widx[q] = (uint32_t)(i0 | (i1 << 6) | (s0 << 12));
        }
    }
    // prologue: fill batch 0 (global loads stall, unavoidable once)
    {
        const int tkn0 = min(8, tk_hi - tk_lo);
        if (threadIdx.x < 64) {
            const int j = threadIdx.x;
            if (j < 16 * tkn0) xhb[j >> 4][j & 15] = xhb_g[(tk_lo + (j >> 4)) * 16 + (j & 15)];
            const int j2 = j + 64;
            if (j2 < 16 * tkn0) xhb[j2 >> 4][j2 & 15] = xhb_g[(tk_lo + (j2 >> 4)) * 16 + (j2 & 15)];
        }
        if (live && u < (w + 3) / 4) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn0; ++b) {
                const uint4 v4 = *(const uint4*)&tile_base[((size_t)((tk_lo + b) * tn + tn_i)) * (size_t)w + u * 4];
                rg[b * 32 + u * 4] = v4.x; rg[b * 32 + u * 4 + 1] = v4.y;
                rg[b * 32 + u * 4 + 2] = v4.z; rg[b * 32 + u * 4 + 3] = v4.w;
            }
        }
    }
    __syncthreads();
    for (int tk0 = tk_lo; tk0 < tk_hi; tk0 += 8) {
        const int tkn = min(8, tk_hi - tk0);
        const int nk0 = tk0 + 8;
        const int tkn2 = min(8, tk_hi - nk0);
        // prefetch next batch into registers while current smem is consumed —
        // hides the global-load latency behind this batch's math.
        uint32_t pfb[8];
        float pfx0 = 0.f, pfx1 = 0.f;
        if (live && nk0 < tk_hi && u < w) {
            for (int b = 0; b < tkn2; ++b)
                pfb[b] = tile_base[((size_t)((nk0 + b) * tn + tn_i)) * (size_t)w + u];
        }
        if (nk0 < tk_hi) {
            const int j = threadIdx.x;
            if (j < 64) {
                pfx0 = (j < 16 * tkn2) ? xhb_g[(nk0 + (j >> 4)) * 16 + (j & 15)] : 0.f;
                const int j2 = j + 64;
                pfx1 = (j2 < 16 * tkn2) ? xhb_g[(nk0 + (j2 >> 4)) * 16 + (j2 & 15)] : 0.f;
            }
        }
        if (live) {
            const uint32_t* rg = rings + g * 256;
            float xr[8][4];
#pragma unroll
            for (int b = 0; b < 8; ++b)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    xr[b][q] = xhb[b][rp + 4 * q];
            for (int b = 0; b < tkn; ++b) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t wi = widx[q];
                    const uint64_t v = ((uint64_t)rg[b * 32 + (wi & 63)] << 32)
                                     | (uint64_t)rg[b * 32 + ((wi >> 6) & 63)];
                    const uint32_t w16 = (uint32_t)(v >> (wi >> 12)) & 0xffffu;
                    const int bs = (int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
                    s += xr[b][q] * (float)bs;
                    sx += xr[b][q];
                }
            }
        }
        __syncthreads();
        // store next batch from regs (load latency already hidden)
        if (live && nk0 < tk_hi && u < w) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn2; ++b) rg[b * 32 + u] = pfb[b];
        }
        if (nk0 < tk_hi) {
            const int j = threadIdx.x;
            if (j < 64) {
                if (j < 16 * tkn2) xhb[j >> 4][j & 15] = pfx0;
                const int j2 = j + 64;
                if (j2 < 16 * tkn2) xhb[j2 >> 4][j2 & 15] = pfx1;
            }
        }
        __syncthreads();
    }
    // 4-way lane reduce within each 4-thread column group
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    sx += __shfl_xor_sync(0xffffffffu, sx, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    sx += __shfl_xor_sync(0xffffffffu, sx, 2);
    if (live && rp == 0) {
        float* dstp = &t[slot * n + zsel * tzoff + tn_i * 16 + c];
        const float v = fmaf(KINV, s, KOFF * sx);
        if (nky > 1) atomicAdd(dstp, v); else *dstp = v;
    }
}

// ---- "stream-decode" MoE gemv (port of kbench sg::gemv_moe_s, MAC=0 f32 exact) ----
// One warp owns one 16x16 tile column (tn) and walks tk. Lane t owns ring
// positions 8t..8t+7: 8 CONSECUTIVE windows -> compile-time bit offsets
// (j*KA + H*((j+1)>>1)). Per tile: 3 __ldg words straight from global, 2 SHF
// align into a 64-bit register stream, then 1 SHF + XMAD-mul16 + IDP/I2F/FFMA
// per window. No smem ring, no __syncthreads in the loop; Σx over the block's
// k-range is precomputed in the prologue. Same affine out as inkwell_exl3_gemv_moe.
__device__ __forceinline__ uint32_t sgemv_mul16(uint32_t w) {   // (w & 0xffff) * 0x83DCD12D mod 2^32
    const uint32_t h = w & 0xffffu;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}
// window j (16 bits, MSB-first) out of the 64-bit stream hi:lo; offset compile-time
template <int D> __device__ __forceinline__ uint32_t sgemv_win(uint32_t hi, uint32_t lo) {
    if constexpr (D <= 16) return hi >> (16 - D);
    else return __funnelshift_r(lo, hi, 48 - D);
}
// Same stream position aligned to the TOP 16 bits. The low half is don't-care:
// the patched CH2 product consumes only XMAD's selected high half.
template <int D> __device__ __forceinline__ uint32_t ch2_view(uint32_t hi, uint32_t lo) {
    if constexpr (D == 0) return hi;
    else if constexpr (D < 32) return __funnelshift_l(lo, hi, D);
    else if constexpr (D == 32) return lo;
    else return lo << (D - 32);
}
// ptxas cannot select the high-half XMAD.PSL form directly. It emits the full
// low+high decomposition of U*0x83dc; the build-time SASS patcher retires the
// low-half XMAD, leaving exactly ((U.H1*0x83dc)<<16) + U.H1*0xd12d.
__device__ __forceinline__ uint32_t ch2_mul16_hi(uint32_t U) {
    uint32_t t;
    asm("{\n\t.reg .u16 lo, hi;\n\tmov.b32 {lo, hi}, %1;\n\tmul.wide.u16 %0, hi, 0xd12d;\n\t}"
        : "=r"(t) : "r"(U));
    return U * 0x83dcu + t;
}
__device__ __forceinline__ float ch2_bs_f32(uint32_t p) {
#ifdef CH2_USE_MAGIC_FADD
    const uint32_t bits = __dp4a(p, 0x01010101u, 0x4B000000u);
    return __uint_as_float(bits) - 8388608.f;
#else
    // Same exact integer s, but uses the otherwise idle conversion pipe.
    // On sm_61 the magic-FADD form competes for the saturated main ALU pipe.
    return (float)(int)__dp4a(p, 0x01010101u, 0u);
#endif
}
template <int KA, int H>
__device__ __forceinline__ void ch2_run(const uint32_t* __restrict__ tb, int tn, int tn_i,
                                        int tk_lo, int tk_hi, const float* __restrict__ xs,
                                        float& a0, float& a1) {
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
#pragma unroll 4
    for (int a = tk_lo; a < tk_hi; ++a) {
        const uint32_t* q = p + (size_t)a * stride;
        const int kb = a * 16 + r0;
        uint32_t pp[8];
#define SG_D(j) ((j) * KA + H * (((j) + 1) >> 1))
        constexpr bool TOP32 = (7 * KA + 4 * H) < 16;
        if constexpr (TOP32) {
            const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1);
            const uint32_t hi = __funnelshift_l(W1, W0, o);
            pp[0] = ch2_mul16_hi(hi); pp[1] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(1)));
            pp[2] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(2))); pp[3] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(3)));
            pp[4] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(4))); pp[5] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(5)));
            pp[6] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(6))); pp[7] = ch2_mul16_hi(__funnelshift_l(hi, hi, SG_D(7)));
        } else {
            const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
            const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
            pp[0] = ch2_mul16_hi(ch2_view<SG_D(0)>(hi, lo)); pp[1] = ch2_mul16_hi(ch2_view<SG_D(1)>(hi, lo));
            pp[2] = ch2_mul16_hi(ch2_view<SG_D(2)>(hi, lo)); pp[3] = ch2_mul16_hi(ch2_view<SG_D(3)>(hi, lo));
            pp[4] = ch2_mul16_hi(ch2_view<SG_D(4)>(hi, lo)); pp[5] = ch2_mul16_hi(ch2_view<SG_D(5)>(hi, lo));
            pp[6] = ch2_mul16_hi(ch2_view<SG_D(6)>(hi, lo)); pp[7] = ch2_mul16_hi(ch2_view<SG_D(7)>(hi, lo));
        }
#undef SG_D
        const float4 xv = *(const float4*)&xs[a * 16 + (t & 3) * 4];
        float f[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = ch2_bs_f32(pp[j]);
        a0 = fmaf(xv.x, f[0], a0); a0 = fmaf(xv.y, f[1], a0); a0 = fmaf(xv.z, f[2], a0); a0 = fmaf(xv.w, f[3], a0);
        a1 = fmaf(xv.x, f[4], a1); a1 = fmaf(xv.y, f[5], a1); a1 = fmaf(xv.z, f[6], a1); a1 = fmaf(xv.w, f[7], a1);
    }
}

template <int KA, int H>
__device__ __forceinline__ void sgemv_run(const uint32_t* __restrict__ tb, int tn, int tn_i,
                                          int tk_lo, int tk_hi, const float* __restrict__ xs,
                                          float& a0, float& a1) {
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
#pragma unroll 4
    for (int a = tk_lo; a < tk_hi; ++a) {
        const uint32_t* q = p + (size_t)a * stride;
        const int kb = a * 16 + r0;
        const uint32_t W0 = __ldg(q + w0), W1 = __ldg(q + w1), W2 = __ldg(q + w2);
        const uint32_t hi = __funnelshift_l(W1, W0, o), lo = __funnelshift_l(W2, W1, o);
        uint32_t pp[8];
#define SG_D(j) ((j) * KA + H * (((j) + 1) >> 1))
        pp[0] = sgemv_mul16(sgemv_win<SG_D(0)>(hi, lo)); pp[1] = sgemv_mul16(sgemv_win<SG_D(1)>(hi, lo));
        pp[2] = sgemv_mul16(sgemv_win<SG_D(2)>(hi, lo)); pp[3] = sgemv_mul16(sgemv_win<SG_D(3)>(hi, lo));
        pp[4] = sgemv_mul16(sgemv_win<SG_D(4)>(hi, lo)); pp[5] = sgemv_mul16(sgemv_win<SG_D(5)>(hi, lo));
        pp[6] = sgemv_mul16(sgemv_win<SG_D(6)>(hi, lo)); pp[7] = sgemv_mul16(sgemv_win<SG_D(7)>(hi, lo));
#undef SG_D
        const float2 xa = *(const float2*)&xs[kb], xc = *(const float2*)&xs[kb + 8];
        float f[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = (float)(int)__dp4a(pp[j], 0x01010101u, 0u);
        a0 = fmaf(xa.x, f[0], a0); a0 = fmaf(xa.y, f[1], a0); a0 = fmaf(xc.x, f[2], a0); a0 = fmaf(xc.y, f[3], a0);
        a1 = fmaf(xa.x, f[4], a1); a1 = fmaf(xa.y, f[5], a1); a1 = fmaf(xc.x, f[6], a1); a1 = fmaf(xc.y, f[7], a1);
    }
}

#define SGEMV_MAXK 2304
// Drop-in for inkwell_exl3_gemv_moe: same params/semantics.
// grid ((tn+7)/8, nslots, nsel*nky), 256 threads = 8 warps = 8 tn tiles of one slot.
__global__ void __launch_bounds__(256) inkwell_exl3_gemv_moe_s(
                                      const uint64_t* __restrict__ descs,
                                      const uint64_t* __restrict__ descs2,
                                      const int* __restrict__ ids,
                                      const float* __restrict__ xh,
                                      const float* __restrict__ xh2, int xh2s,
                                      float* __restrict__ t, int tslot, int tzoff,
                                      int tk, int tn, int k, int n, int nky,
                                      const int* __restrict__ missf, int skipmode, int pw16) {
    const int slot = blockIdx.y;
    // skipmode 1: hits only; 2: misses only; 0: all slots (uniform per block)
    if (skipmode && ((missf[slot] != 0) == (skipmode == 1))) return;
    const int nsel = descs2 ? 2 : 1;             // fused gate+up pairs vs single
    const int ky = blockIdx.z / nsel;            // k-range index
    const int zsel = blockIdx.z % nsel;
    const uint64_t* d = (zsel ? descs2 : descs) + ids[slot] * 4;
    const int tk_lo = (ky * tk) / nky;
    const int tk_hi = ((ky + 1) * tk) / nky;
    const float* xg = zsel ? xh2 + (size_t)slot * xh2s : xh + (size_t)slot * k;
    const uint32_t* tb = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff), hk = (int)((d[3] >> 8) & 1);
    const int kk = ka * 2 + hk;
    const bool ch2 = (pw16 & (1 << 30)) != 0 && kk != 10;
    const bool pw = (pw16 >> kk) & 1;
    __shared__ __align__(16) float xs[SGEMV_MAXK];
    __shared__ float red[8];
    // prologue: stage this block's k-range of x and its Σx (pw16 permutes lanes)
    float sxl = 0.f;
    for (int e = tk_lo * 16 + threadIdx.x; e < tk_hi * 16; e += 256) {
        const float v = xg[e];
        sxl += v;
        if (pw) { const int r = e & 15; xs[(e & ~15) + ((r & 4) ? 8 : 0) + ((r & 8) ? 4 : 0) + (r & 3)] = v; }
        else if (ch2) { const int r = e & 15; xs[(e & ~15) + 4 * ((r & 7) >> 1) + (r & 1) + ((r & 8) ? 2 : 0)] = v; }
        else xs[e] = v;
    }
#pragma unroll
    for (int s = 16; s; s >>= 1) sxl += __shfl_xor_sync(0xffffffffu, sxl, s);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sxl;
    __syncthreads();
    float sx = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sx += red[i];
    if (ch2) {
        const int tn_i = blockIdx.x * 8 + (threadIdx.x >> 5);
        if (tn_i >= tn) return;
        float a0 = 0.f, a1 = 0.f;
        switch (kk) {
            case 3:  ch2_run<1, 1>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 4:  ch2_run<2, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 5:  ch2_run<2, 1>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 6:  ch2_run<3, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 7:  ch2_run<3, 1>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 8:  ch2_run<4, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
            case 10: ch2_run<5, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
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
        return;
    }
    if (pw) {
        const int part = threadIdx.x >> 7;
        const int t_in_blk = (threadIdx.x & 127) >> 4;
        const int sub = threadIdx.x & 15;
        const int tn_i = blockIdx.x * 8 + t_in_blk;
        const int span = tk_hi - tk_lo;
        const int a_lo = tk_lo + ((part * span) >> 1);
        const int a_hi = tk_lo + (((part + 1) * span) >> 1);
        float a0 = 0.f, a1 = 0.f;
        const bool live = tn_i < tn;
        if (live) {
            switch (kk) {
                case 3:  pw16_run<1, 1>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 4:  pw16_run<2, 0>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 5:  pw16_run<2, 1>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 6:  pw16_run<3, 0>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 7:  pw16_run<3, 1>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 8:  pw16_run<4, 0>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                case 10: pw16_run<5, 0>(tb, tn, tn_i, a_lo, a_hi, xs, a0, a1); break;
                default: __trap();
            }
        }
        pw16_reduce(part, t_in_blk, sub, live, xs, a0, a1);
        if (part == 0 && (sub & 1) == 0 && live) {
            const float KINV = __half2float(__ushort_as_half(0x1eee));
            const float off = PW16_KOFFC * sx;   // chunk Σx belongs to every column once
            float* dst = &t[(size_t)slot * n + zsel * tzoff + tn_i * 16 + (sub >> 1)];
            const float v0 = fmaf(KINV, a0, off), v1 = fmaf(KINV, a1, off);
            if (nky > 1) { atomicAdd(dst, v0); atomicAdd(dst + 8, v1); } else { dst[0] = v0; dst[8] = v1; }
        }
        return;
    }
    const int tn_i = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (tn_i >= tn) return;
    float a0 = 0.f, a1 = 0.f;
    switch (kk) {
        case 3:  sgemv_run<1, 1>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 4:  sgemv_run<2, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 5:  sgemv_run<2, 1>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 6:  sgemv_run<3, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 7:  sgemv_run<3, 1>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 8:  sgemv_run<4, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
        case 10: sgemv_run<5, 0>(tb, tn, tn_i, tk_lo, tk_hi, xs, a0, a1); break;
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

// ---- optional build-generated CH2 NOP-deleted module (driver API) ----
extern "C" {
extern const unsigned char ink_ch2nopless_cubin_start[];
extern const unsigned char ink_ch2nopless_cubin_end[];
}

struct Ch2NpState {
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    int state = 0;   // 0 uninitialised, 1 enabled, 2 disabled/unavailable
    size_t bytes = 0;
    double selftest_ms = 0.0;
};
static Ch2NpState g_ch2np;

static int ch2np_requested() {
    if (getenv("INKWELL_NO_CH2NOPLESS")) return 0;
    const char* e = getenv("INKWELL_CH2NOPLESS");
    return e ? atoi(e) != 0 : 1;   // accepted; INKWELL_NO_CH2NOPLESS remains the killswitch
}

static CUresult ch2np_driver_launch(CUfunction f, dim3 grid,
                                    const uint64_t* descs, const uint64_t* descs2,
                                    const int* ids, const float* xh, const float* xh2, int xh2s,
                                    float* t, int tslot, int tzoff,
                                    int tk, int tn, int k, int n, int nky,
                                    const int* missf, int skipmode, int pw16,
                                    cudaStream_t stream) {
    void* args[] = {&descs, &descs2, &ids, &xh, &xh2, &xh2s, &t, &tslot, &tzoff,
                    &tk, &tn, &k, &n, &nky, &missf, &skipmode, &pw16};
    return cuLaunchKernel(f, grid.x, grid.y, grid.z, 256, 1, 1, 0, (CUstream)stream, args, nullptr);
}

static uint32_t ch2np_rs = 777001u;
static uint32_t ch2np_rnd() {
    ch2np_rs ^= ch2np_rs << 13; ch2np_rs ^= ch2np_rs >> 17; ch2np_rs ^= ch2np_rs << 5;
    return ch2np_rs;
}

// One small single-projection case per optimized K. The runtime-linked kernel
// (pw16=0, shipped scalar core) is the reference; the module uses bit 30 (CH2).
static bool ch2np_selftest(CUfunction f) {
    constexpr int TK = 16, TN = 16, K = TK * 16, N = TN * 16;
    constexpr size_t MAXW = (size_t)TK * TN * 40;
    uint32_t* trel = nullptr;
    uint64_t* desc = nullptr;
    int* ids = nullptr;
    float* x = nullptr;
    float* out_ref = nullptr;
    float* out_new = nullptr;
    bool ok = false;
    uint32_t hw[MAXW];
    float hx[K], ha[N], hb[N];
    const int cases[6][2] = {{1,1},{2,0},{2,1},{3,0},{3,1},{4,0}};

    if (cudaMalloc((void**)&trel, MAXW * 4) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&desc, 4 * 8) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&ids, 4) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&x, K * 4) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&out_ref, N * 4) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&out_new, N * 4) != cudaSuccess) goto done;

    for (int i = 0; i < K; ++i) hx[i] = ((int)(ch2np_rnd() % 2001) - 1000) / 512.f;
    if (cudaMemcpy(x, hx, K * 4, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    {
        int hi = 0;
        if (cudaMemcpy(ids, &hi, 4, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
    }

    for (int ci = 0; ci < 6; ++ci) {
        const int ka = cases[ci][0], hk = cases[ci][1];
        {
            const int nw = 8 * ka + 4 * hk;
            const size_t words = (size_t)TK * TN * nw;
            for (size_t i = 0; i < words; ++i) hw[i] = ch2np_rnd();
            if (cudaMemcpy(trel, hw, words * 4, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
            const uint64_t hd[4] = {(uint64_t)(uintptr_t)trel, 0, 0, (uint64_t)(ka | (hk << 8))};
            if (cudaMemcpy(desc, hd, sizeof hd, cudaMemcpyHostToDevice) != cudaSuccess) goto done;
            if (cudaMemset(out_ref, 0, N * 4) != cudaSuccess) goto done;
            if (cudaMemset(out_new, 0, N * 4) != cudaSuccess) goto done;

            dim3 grid((TN + 7) / 8, 1, 1);
            inkwell_exl3_gemv_moe_s<<<grid, 256, 0, nullptr>>>(desc, nullptr, ids, x, nullptr, 0,
                out_ref, N, 0, TK, TN, K, N, 1, nullptr, 0, 0);
            CUresult cr = ch2np_driver_launch(f, grid, desc, nullptr, ids, x, nullptr, 0,
                out_new, N, 0, TK, TN, K, N, 1, nullptr, 0, 1 << 30, nullptr);
            if (cr != CUDA_SUCCESS) goto done;
            if (cudaDeviceSynchronize() != cudaSuccess) goto done;
            if (cudaMemcpy(ha, out_ref, N * 4, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
            if (cudaMemcpy(hb, out_new, N * 4, cudaMemcpyDeviceToHost) != cudaSuccess) goto done;
            if (memcmp(ha, hb, N * 4) != 0) goto done;
        }
    }
    ok = cudaGetLastError() == cudaSuccess;
done:
    if (trel) cudaFree(trel);
    if (desc) cudaFree(desc);
    if (ids) cudaFree(ids);
    if (x) cudaFree(x);
    if (out_ref) cudaFree(out_ref);
    if (out_new) cudaFree(out_new);
    return ok;
}

extern "C" int ink_ch2nopless_startup() {
    if (g_ch2np.state) return g_ch2np.state == 1;
    g_ch2np.state = 2;
    if (!ch2np_requested()) return 0;
    const size_t bytes = (size_t)(ink_ch2nopless_cubin_end - ink_ch2nopless_cubin_start);
    if (bytes == 0) return 0;

    cudaFree(0);   // ensure the runtime primary context exists before module load
    cuInit(0);
    CUresult r = cuModuleLoadData(&g_ch2np.mod, (const void*)ink_ch2nopless_cubin_start);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "ch2nopless: disabled: cuModuleLoadData error %d\n", (int)r);
        return 0;
    }
    static const char* name = "_Z23inkwell_exl3_gemv_moe_sPKmS0_PKiPKfS4_iPfiiiiiiiS2_ii";
    r = cuModuleGetFunction(&g_ch2np.fn, g_ch2np.mod, name);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "ch2nopless: disabled: cuModuleGetFunction error %d\n", (int)r);
        cuModuleUnload(g_ch2np.mod);
        g_ch2np.mod = nullptr;
        return 0;
    }
    timespec ts0{}, ts1{};
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    const bool exact = ch2np_selftest(g_ch2np.fn);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    g_ch2np.selftest_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) * 1e-6;
    g_ch2np.bytes = bytes;
    if (!exact) {
        fprintf(stderr, "ch2nopless: disabled: bitwise self-test failed\n");
        cuModuleUnload(g_ch2np.mod);
        g_ch2np.mod = nullptr;
        g_ch2np.fn = nullptr;
        return 0;
    }
    g_ch2np.state = 1;
    fprintf(stderr, "ch2nopless: enabled cubin=%zu bytes selftest=%.2f ms\n", bytes, g_ch2np.selftest_ms);
    return 1;
}

extern "C" int ink_ch2nopless_enabled() { return g_ch2np.state == 1; }

static bool ch2np_try_launch(dim3 grid, cudaStream_t st,
                             const uint64_t* descs, const uint64_t* descs2,
                             const int* ids, const float* xh, const float* xh2, int xh2s,
                             float* t, int tslot, int tzoff,
                             int tk, int tn, int k, int n, int nky,
                             const int* missf, int skipmode, int pw16) {
    if (g_ch2np.state == 0) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(st, &cap) == cudaSuccess && cap == cudaStreamCaptureStatusNone)
            ink_ch2nopless_startup();
    }
    if (g_ch2np.state != 1) return false;
    const CUresult r = ch2np_driver_launch(g_ch2np.fn, grid, descs, descs2, ids, xh, xh2, xh2s,
        t, tslot, tzoff, tk, tn, k, n, nky, missf, skipmode, pw16 | (1 << 30), st);
    if (r == CUDA_SUCCESS) return true;
    fprintf(stderr, "ch2nopless: disabled: cuLaunchKernel error %d\n", (int)r);
    g_ch2np.state = 2;
    return false;
}

// Shared dispatcher: use the validated module when present, otherwise the
// runtime-linked kernel with the shipped scalar core (bit 30 clear).
static void launch_moe_s(dim3 grid, cudaStream_t st,
                         const uint64_t* descs, const uint64_t* descs2,
                         const int* ids, const float* xh, const float* xh2, int xh2s,
                         float* t, int tslot, int tzoff,
                         int tk, int tn, int k, int n, int nky,
                         const int* missf, int skipmode) {
    const int pw = ink_pw16_mask();
    if (ch2np_try_launch(grid, st, descs, descs2, ids, xh, xh2, xh2s, t, tslot, tzoff,
                         tk, tn, k, n, nky, missf, skipmode, pw)) return;
    inkwell_exl3_gemv_moe_s<<<grid, 256, 0, st>>>(descs, descs2, ids, xh, xh2, xh2s,
        t, tslot, tzoff, tk, tn, k, n, nky, missf, skipmode, pw);
}

__global__ void inkwell_exl3_had_out_moe
(const uint64_t* __restrict__ descs,
                                         const uint64_t* __restrict__ descs2,
                                         const int* __restrict__ ids,
                                         const float* __restrict__ t,
                                         float* __restrict__ y, int n, int stride, int off,
                                         int tslot, int tzoff) {
    const int slot = blockIdx.y;
    const uint64_t* dd = blockIdx.z ? descs2 : descs;
    const uint16_t* svh = (const uint16_t*)(uintptr_t)dd[ids[slot] * 4 + 2];
    off += blockIdx.z * n;
    const int lane = threadIdx.x & 31;
    const int base = blockIdx.x * 512 + (threadIdx.x >> 5) * 128;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = base + lane * 4 + i;
        v[i] = (e < n) ? t[slot * tslot + blockIdx.z * tzoff + e] : 0.f;
    }
    had128_warp(v, lane * 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = base + lane * 4 + i;
        if (e < n) y[slot * stride + off + e] = v[i] * 0.08838834764831845f
            * __half2float(__ushort_as_half(svh[e]));
    }
}

// Vectorized/fused MoE Hadamard chain (bit-identical operation order).
__global__ void inkwell_exl3_had_in_moe_vec(const uint64_t* __restrict__ descs,
        const uint64_t* __restrict__ descs2, const int* __restrict__ ids,
        const float* __restrict__ x, float* __restrict__ xh, float* __restrict__ xh2,
        int xh2s, int k, const int* __restrict__ missf, int skipmode) {
    const int slot = blockIdx.y;
    if (skipmode && ((missf[slot] != 0) == (skipmode == 1))) return;
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

__global__ void inkwell_exl3_had_out_moe_vec(const uint64_t* __restrict__ descs,
        const uint64_t* __restrict__ descs2, const int* __restrict__ ids,
        const float* __restrict__ t, float* __restrict__ y, int n, int stride, int off,
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

// Fused situ_pair + had_in(Down), ncols = 1152 and no per-expert scale.
__global__ void inkwell_situ_had_in_moe(const float* __restrict__ gu,
        const uint64_t* __restrict__ descs_d, const int* __restrict__ ids,
        float* __restrict__ xh, float situ_beta, float situ_lin_beta, int ncols) {
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

// Fused had_out(Down) + weighted sum over the fixed eight selected experts.
__global__ void inkwell_had_out_wsum_moe(const uint64_t* __restrict__ descs_d,
        const int* __restrict__ ids, const float* __restrict__ dn,
        const float* __restrict__ w, float* __restrict__ out, int n) {
    __shared__ float sm[8][128];
    const int slot = threadIdx.x >> 5;
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

static inline bool ink_no_hadvec() {
    static int v = -1;
    if (v < 0) v = (getenv("INKWELL_HADVEC") && !getenv("INKWELL_NO_HADVEC")) ? 0 : 1;
    return v != 0;
}

// Stage selected experts pinned-host → VRAM staging (coalesced uint4 copies,
// latency hidden by occupancy). Rewrites staged desc tables (indexed by
// ids[slot]) to point at staging; downstream kernels unchanged.
// Per (slot,proj) layout: [trellis][suh][svh]; trellis bytes depend on
// per-expert ka from the descriptor (10368 tiles × r_bits_words×4).
#define EXL3_STAGE_PROJ 1334016u  // w<=32: 10368*32*4 + 6912 suh/svh
__global__ void inkwell_exl3_moe_stage(const uint64_t* __restrict__ descs_g,
                                       const uint64_t* __restrict__ descs_u,
                                       const uint64_t* __restrict__ descs_d,
                                       const int* __restrict__ ids,
                                       uint8_t* __restrict__ staging,
                                       uint64_t* __restrict__ sdescs_g,
                                       uint64_t* __restrict__ sdescs_u,
                                       uint64_t* __restrict__ sdescs_d) {
    const int slot = blockIdx.x / 3;
    const int proj = blockIdx.x % 3;
    const uint64_t* src = (proj == 0 ? descs_g : (proj == 1 ? descs_u : descs_d));
    uint64_t* sd = (proj == 0 ? sdescs_g : (proj == 1 ? sdescs_u : sdescs_d));
    const uint64_t* d = src + ids[slot] * 4;
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int trel_b = 10368 * r_bits_words(ka, halfk) * 4;
    uint8_t* dst = staging + (size_t)(slot * 3 + proj) * EXL3_STAGE_PROJ;
    const int suh_b = (proj == 2 ? 2304 : 4608);  // down in=1152→2304B; gate/up in=2304→4608B
    const int svh_b = (proj == 2 ? 4608 : 2304);
    const uint4* s0 = (const uint4*)(uintptr_t)d[0];
    const uint4* s1 = (const uint4*)(uintptr_t)d[1];
    const uint4* s2 = (const uint4*)(uintptr_t)d[2];
    const int n0 = trel_b / 16, n1 = suh_b / 16, n2 = svh_b / 16;
    uint4* t0 = (uint4*)dst;
    uint4* t1 = (uint4*)(dst + trel_b);
    uint4* t2 = (uint4*)(dst + trel_b + suh_b);
    const int t = threadIdx.x;
    for (int i = t; i < n0; i += blockDim.x) t0[i] = s0[i];
    for (int i = t; i < n1; i += blockDim.x) t1[i] = s1[i];
    for (int i = t; i < n2; i += blockDim.x) t2[i] = s2[i];
    if (t == 0) {
        uint64_t* o = sd + ids[slot] * 4;
        o[0] = (uint64_t)(uintptr_t)dst;
        o[1] = (uint64_t)(uintptr_t)(dst + trel_b);
        o[2] = (uint64_t)(uintptr_t)(dst + trel_b + suh_b);
        o[3] = d[3];
    }
}

// Single-projection variant of moe_stage: one block per slot copies proj `proj`'s
// blob (trellis+suh+svh) from host-mapped descs into staging and fixes sdescs.
__global__ void inkwell_exl3_moe_stage_p(const uint64_t* __restrict__ descs,
                                       const int* __restrict__ ids,
                                       uint8_t* __restrict__ staging,
                                       uint64_t* __restrict__ sdescs,
                                       int proj) {
    const int slot = blockIdx.x;
    const uint64_t* d = descs + ids[slot] * 4;
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int trel_b = 10368 * r_bits_words(ka, halfk) * 4;
    uint8_t* dst = staging + (size_t)(slot * 3 + proj) * EXL3_STAGE_PROJ;
    const int suh_b = (proj == 2 ? 2304 : 4608);
    const int svh_b = (proj == 2 ? 4608 : 2304);
    const uint4* s0 = (const uint4*)(uintptr_t)d[0];
    const uint4* s1 = (const uint4*)(uintptr_t)d[1];
    const uint4* s2 = (const uint4*)(uintptr_t)d[2];
    const int n0 = trel_b / 16, n1 = suh_b / 16, n2 = svh_b / 16;
    uint4* t0 = (uint4*)dst;
    uint4* t1 = (uint4*)(dst + trel_b);
    uint4* t2 = (uint4*)(dst + trel_b + suh_b);
    const int t = threadIdx.x;
    for (int i = t; i < n0; i += blockDim.x) t0[i] = s0[i];
    for (int i = t; i < n1; i += blockDim.x) t1[i] = s1[i];
    for (int i = t; i < n2; i += blockDim.x) t2[i] = s2[i];
    if (t == 0) {
        uint64_t* o = sdescs + ids[slot] * 4;
        o[0] = (uint64_t)(uintptr_t)dst;
        o[1] = (uint64_t)(uintptr_t)(dst + trel_b);
        o[2] = (uint64_t)(uintptr_t)(dst + trel_b + suh_b);
        o[3] = d[3];
    }
}

// Per-layer VRAM expert cache: resolve ids -> K slots, copy only misses.
// tag[v] = expert id (-1 empty); map[s] = slot chosen for ids[s];
// miss[s] = 1 when a copy is required. rr = per-layer victim counter.
// PF_BASE+p encodes "slot p of the prefetch bank" in map[].
#define EXL3_PF_BASE 0x40000000
#define EXL3_DIRECT 0x20000000
__global__ void inkwell_exl3_cache_resolve(const int* __restrict__ ids,
        int* __restrict__ tag, int* __restrict__ map, int* __restrict__ miss,
        int* __restrict__ cnt, int nslots, int k,
        const int* __restrict__ ptag, const int* __restrict__ pready,
        int* __restrict__ pf_hit, const int* __restrict__ pnexp,
        int key_base, int force_direct, int* __restrict__ hm) {
    // LFU eviction: cnt[e] counts uses; the least-used resident is evicted.
    // pf_hit[0]=bank hits, pf_hit[1]=real misses (filled via serial ccopy)
    // hm[0]=selection hits (cache or bank), hm[1]=misses (debug counters)
    if (threadIdx.x != 0) return;
    if (pnexp) nslots = *pnexp;
    const int pf = pready ? *pready : 0;
    unsigned long long claimed = 0;
    // pass 1: hits (and prefetch-bank hits) — never evict slots needed now
    for (int s = 0; s < nslots; ++s) {
        const int e = ids[s] + key_base;
        ++cnt[e];
        int v = -1;
        for (int j = 0; j < k; ++j) if (tag[j] == e) { v = j; break; }
        int m = 0;
        if (v < 0 && pf) {
            for (int p2 = 0; p2 < nslots; ++p2)
                if (ptag[p2] == e) { v = EXL3_PF_BASE + p2; if (pf_hit) atomicAdd(pf_hit, 1); break; }
        }
        if (v >= 0) {
            miss[s] = 0;
            map[s] = v;
            if (v < k) claimed |= 1ULL << v;
            if (hm) ++hm[0];
        } else {
            miss[s] = 1;
            map[s] = -1;
            if (hm) ++hm[1];
        }
    }
    // pass 2: evict the least-used unclaimed resident for each miss
    for (int s = 0; s < nslots; ++s) {
        if (!miss[s]) continue;
        const int e = ids[s] + key_base;
        int best = -1, bestc = 0x7fffffff;
        for (int j = 0; j < k; ++j) {
            if (claimed & (1ULL << j)) continue;
            const int t2 = tag[j];
            const int c = (t2 < 0) ? 0 : cnt[t2];
            if (c < bestc) { bestc = c; best = j; }
        }
        if (best < 0 || force_direct) {
            map[s] = EXL3_DIRECT;
            continue;
        }
        const int v = best;
        claimed |= 1ULL << v;
        tag[v] = e;
        map[s] = v;
        if (pf_hit) atomicAdd(pf_hit + 1, 1);
    }
}

// kslot resolve (INKWELL_NO_KSLOT unset): same global-LFU eviction as the
// uniform cache, but slots are variable-size — the cache buffer is one arena
// of KSLOT_GRAN-byte granules and each expert's proj blob packs tightly
// (gexp[e*3+p] = 6*w_ep+1 granules, exact fit). A slot is 3 extents + a tag
// entry; soff[v*3+p] (bytes) is maintained here and read by ccopy.
// fl = free-extent list: fl[0]=n, then (base,len) i32 pairs sorted by base.
#define KSLOT_GRAN 6912
#define KSLOT_FLMAX 256  // > 3*64 max extents + 1
__global__ void inkwell_exl3_kslot_resolve(const int* __restrict__ ids,
        int* __restrict__ tag, uint64_t* __restrict__ soff,
        int* __restrict__ map, int* __restrict__ miss,
        int* __restrict__ cnt, int nslots, int kmax,
        const int* __restrict__ gexp, int* __restrict__ fl,
        const int* __restrict__ ptag, const int* __restrict__ pready,
        int* __restrict__ pf_hit, const int* __restrict__ pnexp,
        int key_base, int force_direct, int* __restrict__ hm) {
    if (threadIdx.x != 0) return;
    if (pnexp) nslots = *pnexp;
    const int pf = pready ? *pready : 0;
    unsigned long long claimed = 0;
    // pass 1: hits (and prefetch-bank hits)
    for (int s = 0; s < nslots; ++s) {
        const int e = ids[s] + key_base;
        ++cnt[e];
        int v = -1;
        for (int j = 0; j < kmax; ++j) if (tag[j] == e) { v = j; break; }
        if (v < 0 && pf) {
            for (int p2 = 0; p2 < nslots; ++p2)
                if (ptag[p2] == e) { v = EXL3_PF_BASE + p2; if (pf_hit) atomicAdd(pf_hit, 1); break; }
        }
        if (v >= 0) {
            miss[s] = 0;
            map[s] = v;
            if (v < kmax) claimed |= 1ULL << v;
            if (hm) ++hm[0];
        } else {
            miss[s] = 1;
            map[s] = -1;
            if (hm) ++hm[1];
        }
    }
    // helpers (single thread): first-fit alloc / sorted insert+merge free
    int nfl = fl[0];
    int* fb = fl + 1;      // bases
    int* fn = fl + 1 + KSLOT_FLMAX; // lens
#define KS_FIT(need, out) { out = -1; \
    for (int q = 0; q < nfl; ++q) if (fn[q] >= (need)) { out = q; break; } }
#define KS_ALLOC(idx, need, base) { base = fb[idx]; fb[idx] += (need); fn[idx] -= (need); \
    if (fn[idx] == 0) { for (int q = idx; q + 1 < nfl; ++q) { fb[q] = fb[q+1]; fn[q] = fn[q+1]; } --nfl; } }
#define KS_FREE(base, len) { \
    int q = 0; while (q < nfl && fb[q] < (base)) ++q; \
    for (int r = nfl; r > q; --r) { fb[r] = fb[r-1]; fn[r] = fn[r-1]; } \
    fb[q] = (base); fn[q] = (len); ++nfl; \
    if (q > 0 && fb[q-1] + fn[q-1] == fb[q]) { fn[q-1] += fn[q]; \
        for (int r = q; r + 1 < nfl; ++r) { fb[r] = fb[r+1]; fn[r] = fn[r+1]; } --nfl; --q; } \
    if (q + 1 < nfl && fb[q] + fn[q] == fb[q+1]) { fn[q] += fn[q+1]; \
        for (int r = q + 1; r + 1 < nfl; ++r) { fb[r] = fb[r+1]; fn[r] = fn[r+1]; } --nfl; } }
    // pass 2: per miss, find a free slot + 3 extents; evict global-least-used
    // unclaimed residents until they fit; DIRECT if nothing left to evict.
    for (int s = 0; s < nslots; ++s) {
        if (!miss[s]) continue;
        const int e = ids[s] + key_base;
        int v = -1;
        for (int j = 0; j < kmax; ++j) if (tag[j] < 0) { v = j; break; }
        int got[3];
        int ng = 0;
        for (;;) {
            while (v >= 0 && ng < 3) {
                const int need = gexp[ids[s] * 3 + ng];
                int fi; KS_FIT(need, fi);
                if (fi < 0) break;
                KS_ALLOC(fi, need, got[ng]);
                ++ng;
            }
            if (v >= 0 && ng == 3) break;
            // evict least-used unclaimed resident
            int best = -1, bestc = 0x7fffffff;
            for (int j = 0; j < kmax; ++j) {
                if (claimed & (1ULL << j)) continue;
                const int t2 = tag[j];
                if (t2 < 0) continue;
                const int c = cnt[t2];
                if (c < bestc) { bestc = c; best = j; }
            }
            if (best < 0) break;
            const int tv = tag[best];
            for (int p = 0; p < 3; ++p) {
                KS_FREE((int)(soff[best * 3 + p] / KSLOT_GRAN), gexp[tv * 3 + p]);
            }
            tag[best] = -1;
            if (v < 0) v = best;
        }
        if (v < 0 || ng < 3 || force_direct) {
            while (ng > 0) { --ng; KS_FREE(got[ng], gexp[ids[s] * 3 + ng]); }
            map[s] = EXL3_DIRECT;
            continue;
        }
        claimed |= 1ULL << v;
        tag[v] = e;
        for (int p = 0; p < 3; ++p) soff[v * 3 + p] = (uint64_t)got[p] * KSLOT_GRAN;
        map[s] = v;
        if (pf_hit) atomicAdd(pf_hit + 1, 1);
    }
    fl[0] = nfl;
#undef KS_FIT
#undef KS_ALLOC
#undef KS_FREE
}

// Copy missing experts into their cache slots and (always) point the staged
// desc tables at the slots. cache layout: (v*3+proj) * EXL3_STAGE_PROJ.
__global__ void inkwell_exl3_moe_ccopy(const uint64_t* __restrict__ descs_g,
                                       const uint64_t* __restrict__ descs_u,
                                       const uint64_t* __restrict__ descs_d,
                                       const int* __restrict__ ids,
                                       uint8_t* __restrict__ cache,
                                       const int* __restrict__ map,
                                       const int* __restrict__ miss,
                                       uint8_t* __restrict__ bank,
                                       const int64_t slot_proj,
                                       uint64_t* __restrict__ sdescs_g,
                                       uint64_t* __restrict__ sdescs_u,
                                       uint64_t* __restrict__ sdescs_d,
                                       int mode, const int* __restrict__ pnexp,
                                       const uint64_t* __restrict__ soff) {
    const int slot = blockIdx.x / 3;
    if (pnexp && slot >= *pnexp) return;
    const int proj = blockIdx.x % 3;
    const int part = blockIdx.y, nparts = gridDim.y;
    const uint64_t* src = (proj == 0 ? descs_g : (proj == 1 ? descs_u : descs_d));
    uint64_t* sd = (proj == 0 ? sdescs_g : (proj == 1 ? sdescs_u : sdescs_d));
    const uint64_t* d = src + ids[slot] * 4;
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int trel_b = 10368 * r_bits_words(ka, halfk) * 4;
    const int suh_b = (proj == 2 ? 2304 : 4608);
    const int svh_b = (proj == 2 ? 4608 : 2304);
    int mslot = map[slot];
    if (mslot >= EXL3_DIRECT && mslot < EXL3_PF_BASE) {
        // direct: point sdescs at the expert's host descriptors — the gemv
        // streams the blob over PCIe; no serial copy, cache stays stable.
        if (threadIdx.x == 0 && part == 0) {
            uint64_t* o = sd + ids[slot] * 4;
            o[0] = d[0];
            o[1] = d[1];
            o[2] = d[2];
            o[3] = d[3];
        }
        return;
    }
    uint8_t* cbase = cache;
    const bool bank_slot = mslot >= EXL3_PF_BASE;
    if (bank_slot) { cbase = bank; mslot -= EXL3_PF_BASE; }
    // soff (kslot): per-(slot,proj) byte offsets for size-class slots; the
    // prefetch bank keeps the uniform stride.
    uint8_t* dst = (!bank_slot && soff)
        ? cbase + soff[mslot * 3 + proj]
        : cbase + (size_t)(mslot * 3 + proj) * (size_t)slot_proj;
    if (threadIdx.x == 0 && part == 0) {
        uint64_t* o = sd + ids[slot] * 4;
        o[0] = (uint64_t)(uintptr_t)dst;
        o[1] = (uint64_t)(uintptr_t)(dst + trel_b);
        o[2] = (uint64_t)(uintptr_t)(dst + trel_b + suh_b);
        o[3] = d[3];
    }
    // mode 3 (INKWELL_SKIP_CCOPY, timing debug): meta only, no data copy.
    if (!miss[slot] || mode == 3) return;
    const uint4* s0 = (const uint4*)(uintptr_t)d[0];
    uint4* t0 = (uint4*)dst;
    // mode 1: meta + suh + svh tail (small, main stream — had_in needs suh);
    // mode 2: trel prefix only (bulk, side stream — gemv joins on it).
    const int lo_lim = (mode == 1) ? trel_b / 16 : 0;
    const int hi_lim = (mode == 2) ? trel_b / 16 : (trel_b + suh_b + svh_b) / 16;
    const int n0 = hi_lim - lo_lim;
    const int chunk = (n0 + nparts - 1) / nparts;
    const int lo = lo_lim + part * chunk, hi = min(hi_lim, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) t0[i] = s0[i];
}

__global__ void inkwell_exl3_copy8(const int* __restrict__ src, int* __restrict__ dst) {
    if (threadIdx.x < 8) dst[threadIdx.x] = src[threadIdx.x];
}
extern "C" void ink_launch_exl3_copy8(const void* src, void* dst, void* stream) {
    inkwell_exl3_copy8<<<1, 8, 0, (cudaStream_t)stream>>>((const int*)src, (int*)dst);
}

// Predictive prefetch, host-issued between tokens on the side stream.
// pf_resolve finds predicted misses among hist_ids and fills pf_map/pf_miss
// with bank slots; pf_ccopy copies those blobs into the bank; pf_tag writes
// ptags + pready LAST so a concurrent visit never reads a half-written bank.
// pready=0 first — a visit that runs while we copy just takes the miss path.
__global__ void inkwell_exl3_pf_resolve(const int* __restrict__ hist_ids,
        const int* __restrict__ tag, int* __restrict__ pf_map,
        int* __restrict__ pf_miss, int k, int nslots) {
    if (threadIdx.x != 0) return;
    int p = 0;
    for (int s = 0; s < nslots; ++s) {
        const int e = hist_ids[s];
        int hit = 0;
        for (int j = 0; j < k; ++j) if (tag[j] == e) { hit = 1; break; }
        pf_map[s] = hit ? -1 : p;
        pf_miss[s] = hit ? 0 : 1;
        if (!hit) ++p;
    }
}
__global__ void inkwell_exl3_moe_pcopy(const uint64_t* __restrict__ descs_g,
                                       const uint64_t* __restrict__ descs_u,
                                       const uint64_t* __restrict__ descs_d,
                                       const int* __restrict__ ids,
                                       uint8_t* __restrict__ bank,
                                       const int* __restrict__ pf_map,
                                       const int* __restrict__ pf_miss,
                                       const int64_t slot_proj,
                                       int* __restrict__ ptag) {
    const int slot = blockIdx.x / 3;
    const int proj = blockIdx.x % 3;
    const int part = blockIdx.y, nparts = gridDim.y;
    if (pf_map[slot] < 0 || !pf_miss[slot]) return;
    const uint64_t* srcp = (proj == 0 ? descs_g : (proj == 1 ? descs_u : descs_d));
    const uint64_t* d = srcp + ids[slot] * 4;
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int trel_b = 10368 * r_bits_words(ka, halfk) * 4;
    const int suh_b = (proj == 2 ? 2304 : 4608);
    const int svh_b = (proj == 2 ? 4608 : 2304);
    uint8_t* dst = bank + (size_t)(pf_map[slot] * 3 + proj) * (size_t)slot_proj;
    const uint4* s0 = (const uint4*)(uintptr_t)d[0];
    uint4* t0 = (uint4*)dst;
    const int n0 = (trel_b + suh_b + svh_b) / 16;
    const int chunk = (n0 + nparts - 1) / nparts;
    const int lo = part * chunk, hi = min(n0, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) t0[i] = s0[i];
}
__global__ void inkwell_exl3_pf_flag(int* __restrict__ pready, int v) {
    if (threadIdx.x == 0) *pready = v;
}
__global__ void inkwell_exl3_pf_tag(const int* __restrict__ hist_ids,
        const int* __restrict__ pf_map, int* __restrict__ ptag, int nslots) {
    __threadfence();
    if (threadIdx.x == 0)
        for (int s = 0; s < nslots; ++s)
            if (pf_map[s] >= 0) ptag[pf_map[s]] = hist_ids[s];
    __threadfence();
}
extern "C" void ink_launch_exl3_prefetch2(const void* descs_g, const void* descs_u, const void* descs_d,
        const int* hist_ids, const int* tag, void* bank, int* ptag, int* pready,
        int* pf_map, int* pf_miss, int nslots, int k, long long slot_proj, void* side) {
    const cudaStream_t ss = (cudaStream_t)side;
    inkwell_exl3_pf_flag<<<1, 1, 0, ss>>>(pready, 0);
    inkwell_exl3_pf_resolve<<<1, 32, 0, ss>>>(hist_ids, tag, pf_map, pf_miss, k, nslots);
    // copy each predicted miss into its bank slot — reuse moe_ccopy with the
    // private map (map[s] = bank slot or -1); write through `bank`.
    inkwell_exl3_moe_pcopy<<<dim3(nslots * 3, 4), 256, 0, ss>>>(
        (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
        hist_ids, (uint8_t*)bank, pf_map, pf_miss, slot_proj, ptag);
    inkwell_exl3_pf_tag<<<1, 1, 0, ss>>>(hist_ids, pf_map, ptag, nslots);
    inkwell_exl3_pf_flag<<<1, 1, 0, ss>>>(pready, 1);
}

// Full per-slot expert chain: stage -> gate -> situ -> down -> wsum.
// x fp32 -> int8 per-16-group, PRE-PACKED for tile_row_d4: out u32 at
// [slot*tk*4 + g*4 + m] holds int8x4 {x[2m], x[2m+1], x[2m+8], x[2m+9]} of group
// g; xqs[slot*tk + g] = amax/127 (1 when amax==0). grid (nslots, tk), 16 threads
// = lanes' 16 elements; each of lanes m=0..3 writes one pack via shfl gather.
__global__ void inkwell_exl3_quant_x_i8p(const float* __restrict__ x,
                                         uint32_t* __restrict__ xq,
                                         float* __restrict__ xqs, int n) {
    const int slot = blockIdx.x;
    const int g = blockIdx.y;
    const int lane = threadIdx.x;                 // 16 threads -> element index
    const float* xr = x + (size_t)slot * n + g * 16;
    const float v = xr[lane];
    float am = fabsf(v);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffu, am, o, 16));
    const float sc = am > 0.f ? am / 127.f : 1.f;
    const int qi = __float2int_rn(v / sc);
    // lane m stores 4 PRE-SPLATTED packs for elements {2m, 2m+1, 2m+8, 2m+9}
    if (lane < 4) {
        uint4* dst = (uint4*)xq + (size_t)slot * (n >> 4) * 4 + g * 4 + lane;
        int a = __shfl_sync(0xffffu, qi, 2 * lane, 16);
        int b = __shfl_sync(0xffffu, qi, 2 * lane + 1, 16);
        int c = __shfl_sync(0xffffu, qi, 2 * lane + 8, 16);
        int d = __shfl_sync(0xffffu, qi, 2 * lane + 9, 16);
        const uint32_t sa = __byte_perm((uint32_t)a, 0, 0x0000u);
        const uint32_t sb = __byte_perm((uint32_t)b, 0, 0x0000u);
        const uint32_t sc_ = __byte_perm((uint32_t)c, 0, 0x0000u);
        const uint32_t sd = __byte_perm((uint32_t)d, 0, 0x0000u);
        *dst = make_uint4(sa, sb, sc_, sd);
    }
    if (lane == 0) xqs[slot * (n >> 4) + g] = sc;
}

extern "C" void ink_launch_exl3_moe(const void* descs_g, const void* descs_u, const void* descs_d,
                                    const int* ids, const float* x,
                                    float* xh, float* gu, float* hid, float* dn,
                                    const float* w, float* out, int nslots, void* stream,
                                    void* staging, void* sdescs_g, void* sdescs_u, void* sdescs_d,
                                    int on_dev, int v2_descs_ok) {
    const cudaStream_t st = (cudaStream_t)stream;
    static cudaStream_t st2 = nullptr;
    static cudaEvent_t ev_ids = nullptr, ev_dstg = nullptr;
    if (!on_dev) {
        if (!st2) { cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking);
                    cudaEventCreateWithFlags(&ev_ids, cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&ev_dstg, cudaEventDisableTiming); }
        // D staging runs on st2 in parallel with the GU compute below.
        cudaEventRecord(ev_ids, st);
        cudaStreamWaitEvent(st2, ev_ids, 0);
        inkwell_exl3_moe_stage_p<<<nslots, 256, 0, st2>>>(
            (const uint64_t*)descs_d, ids, (uint8_t*)staging, (uint64_t*)sdescs_d, 2);
        cudaEventRecord(ev_dstg, st2);
        inkwell_exl3_moe_stage_p<<<nslots, 256, 0, st>>>(
            (const uint64_t*)descs_g, ids, (uint8_t*)staging, (uint64_t*)sdescs_g, 0);
        inkwell_exl3_moe_stage_p<<<nslots, 256, 0, st>>>(
            (const uint64_t*)descs_u, ids, (uint8_t*)staging, (uint64_t*)sdescs_u, 1);
    }
    const uint64_t* G = on_dev ? (const uint64_t*)descs_g : (const uint64_t*)sdescs_g;
    const uint64_t* U = on_dev ? (const uint64_t*)descs_u : (const uint64_t*)sdescs_u;
    const uint64_t* D = on_dev ? (const uint64_t*)descs_d : (const uint64_t*)sdescs_d;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    const bool hadvec = !ink_no_hadvec() && nslots == 8;
    // gate + up fused: blockIdx.z selects the descriptor set
    dim3 gin(hadvec ? (KG + 255) / 256 : (KG + 511) / 512, nslots, 2);
    if (hadvec) inkwell_exl3_had_in_moe_vec<<<gin, 64, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG, nullptr, 0);
    else inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG, nullptr, 0, nullptr, nullptr);
    int nky = 1;
    { const char* e = getenv("INKWELL_MOE_NKY"); if (e) nky = atoi(e); if (nky < 1) nky = 1; }
    int nky_s = 1;
    { const char* e = getenv("INKWELL_SGEMV_NKY"); if (e) nky_s = atoi(e); if (nky_s < 1) nky_s = 1; }
    static int no_sgemv = -1;
    if (no_sgemv < 0) no_sgemv = getenv("INKWELL_NO_SGEMV") ? atoi(getenv("INKWELL_NO_SGEMV")) : 0;
    const bool sg = !no_sgemv;                  // stream-decode gemv replaces v1+v2 serial
    static int v2_env = -1;
    if (v2_env < 0) v2_env = getenv("INKWELL_MOE_V2") ? atoi(getenv("INKWELL_MOE_V2")) : 0;
    // Descriptor eligibility is immutable and was scanned on the host at load;
    // the tables themselves may be device-resident and are never dereferenced here.
    int v2_ok = !sg && (v2_env > 0) && nky == 1 && v2_descs_ok;
    static int* e8 = nullptr;
    if (v2_ok && !e8) {
        cudaMalloc(&e8, 65 * 4);
        int h[65]; for (int i = 0; i < 65; ++i) h[i] = i;
        cudaMemcpy(e8, h, 260, cudaMemcpyHostToDevice);
    }
    const int nky_eff = sg ? nky_s : nky;       // nky of the kernel actually launched
    if (nky_eff > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    static uint32_t* xq8 = nullptr; static float* xqs8 = nullptr;
    static uint32_t* xq8b = nullptr; static float* xqs8b = nullptr;
    const bool d4 = v2_ok && getenv("INKWELL_MOE_D4A");
    if (d4 && !xq8) {
        cudaMalloc(&xq8, 64 * 2304); cudaMalloc(&xqs8, 64 * 144 * 4);
        cudaMalloc(&xq8b, 64 * 2304); cudaMalloc(&xqs8b, 64 * 144 * 4);
    }
    // GU stays fp32 (int8-x loses ~10%/window here) -> no GU quants
    if (sg) {
        dim3 gg((NG / 16 + 7) / 8, nslots, 2 * nky_s);
        launch_moe_s(gg, st, G, U, ids, xh, (const float*)gu, 2 * NG,
                        (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky_s, nullptr, 0);
    } else if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(NG / 64, nslots, 2), 256, 0, st>>>(G, U, ids, e8, xh,
        (float*)gu, 2 * NG, (float*)dn, ND, NG, KG / 16, NG / 16, nullptr, nullptr, 0,
        nullptr, nullptr, nullptr, nullptr);   // GU stays fp32: int8-x loses ~10%/window
    else {
        dim3 gg(NG / 64, nslots, 2 * nky);
        inkwell_exl3_gemv_moe<<<gg, 256, 0, st>>>(G, U, ids, xh, (const float*)gu, 2 * NG,
                                                 (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky, nullptr, 0);
    }
    dim3 gout(hadvec ? (NG + 383) / 384 : (NG + 511) / 512, nslots, 2);
    if (hadvec) {
        inkwell_exl3_had_out_moe_vec<<<gout, 96, 0, st>>>(G, U, ids, dn, gu, NG, 2 * NG, 0, ND, NG);
        if (!on_dev) cudaStreamWaitEvent(st, ev_dstg, 0);
        inkwell_situ_had_in_moe<<<dim3(3, nslots), 96, 0, st>>>(gu, D, ids, xh, 4.f, 25.f, NG);
    } else {
        inkwell_exl3_had_out_moe<<<gout, 128, 0, st>>>(G, U, ids, dn, gu, NG, 2 * NG, 0, ND, NG);
        inkwell_situ_pair<<<(unsigned)((nslots * NG + 255) / 256), 256, 0, st>>>(gu, nullptr, hid, 4.f, 25.f, NG, nslots);
        dim3 din((KD + 511) / 512, nslots);
        if (!on_dev) cudaStreamWaitEvent(st, ev_dstg, 0);
        inkwell_exl3_had_in_moe<<<din, 128, 0, st>>>(D, nullptr, ids, hid, xh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
    }
    if (nky_eff > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    if (d4) inkwell_exl3_quant_x_i8p<<<dim3(nslots, KD / 16), 16, 0, st>>>(xh, xq8, xqs8, KD);
    if (sg) {
        dim3 gd((ND / 16 + 7) / 8, nslots, nky_s);
        launch_moe_s(gd, st, D, nullptr, ids, xh, nullptr, 0,
                        dn, ND, 0, KD / 16, ND / 16, KD, ND, nky_s, nullptr, 0);
    } else if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(ND / 64, nslots, 1), 256, 0, st>>>(D, nullptr, ids, e8, xh,
        nullptr, 0, (float*)dn, ND, 0, KD / 16, ND / 16, nullptr, nullptr, 0,
        d4 ? xq8 : nullptr, d4 ? xqs8 : nullptr, nullptr, nullptr);
    else {
        dim3 gd(ND / 64, nslots, nky);
        inkwell_exl3_gemv_moe<<<gd, 256, 0, st>>>(D, nullptr, ids, xh, nullptr, 0,
                                                 dn, ND, 0, KD / 16, ND / 16, KD, ND, nky, nullptr, 0);
    }
    if (hadvec) {
        inkwell_had_out_wsum_moe<<<dim3(ND / 128), 256, 0, st>>>(D, ids, dn, w, out, ND);
    } else {
        dim3 dout((ND + 511) / 512, nslots, 1);
        inkwell_exl3_had_out_moe<<<dout, 128, 0, st>>>(D, nullptr, ids, dn, xh /*reuse xh as y*/, ND, ND, 0, ND, 0);
        inkwell_expert_wsum<<<(unsigned)((ND + 255) / 256), 256, 0, st>>>(xh, w, nullptr, out, ND, nslots);
    }
}

// Cached variant: resolve -> copy misses -> same downstream chain.
extern "C" void ink_launch_exl3_moe_cached(const void* descs_g, const void* descs_u, const void* descs_d,
                                    const int* ids, const float* x,
                                    float* xh, float* gu, float* hid, float* dn,
                                    const float* w, float* out, int nslots, void* stream,
                                    void* cache, void* tag, void* map, void* miss, void* cnt, int k,
                                    int64_t slot_proj,
                                    void* sdescs_g, void* sdescs_u, void* sdescs_d,
                                    int v2_descs_ok, const void* ptag, const void* pready, void* bank, int* pf_hit,
                                    int* hm, const int* gexp, uint64_t* soff, int* fl) {
    const cudaStream_t st = (cudaStream_t)stream;
    // INKWELL_SKIP_CCOPY=1 (timing debug): ccopy writes slot metadata but
    // moves no data — bounds how much staging sits on the critical path.
    static int skip_cc = -1;
    if (skip_cc < 0) { const char* e = getenv("INKWELL_SKIP_CCOPY"); skip_cc = e ? atoi(e) : 0; }
    if (soff) {
        inkwell_exl3_kslot_resolve<<<1, 32, 0, st>>>(ids,
            (int*)tag, (uint64_t*)soff, (int*)map, (int*)miss, (int*)cnt, nslots, k,
            gexp, fl, (const int*)ptag, (const int*)pready, pf_hit, nullptr, 0,
            getenv("INKWELL_RESOLVE_DIRECT") ? 1 : 0, hm);
    } else {
        inkwell_exl3_cache_resolve<<<1, 32, 0, st>>>(ids,
            (int*)tag, (int*)map, (int*)miss, (int*)cnt, nslots, k,
            (const int*)ptag, (const int*)pready, pf_hit, nullptr, 0,
            getenv("INKWELL_RESOLVE_DIRECT") ? 1 : 0, hm);
    }
    static cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    static cudaStream_t ss = nullptr;
    {
        // meta on the main stream (sdescs only, ~µs); bulk trel copies on the
        // side stream so they overlap the had_in + their join delays gemv only.
        if (!ev_fork) {
            cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming);
            cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking);
        }
        inkwell_exl3_moe_ccopy<<<dim3(nslots * 3, 1), 32, 0, st>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            ids, (uint8_t*)cache, (const int*)map, (const int*)miss,
            (uint8_t*)bank, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, skip_cc ? 3 : 1, nullptr, soff);
        cudaEventRecord(ev_fork, st);
        cudaStreamWaitEvent(ss, ev_fork, 0);
        inkwell_exl3_moe_ccopy<<<dim3(nslots * 3, 16), 128, 0, ss>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            ids, (uint8_t*)cache, (const int*)map, (const int*)miss,
            (uint8_t*)bank, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, skip_cc ? 3 : 2, nullptr, soff);
        cudaEventRecord(ev_join, ss);
    }
    const uint64_t* G = (const uint64_t*)sdescs_g;
    const uint64_t* U = (const uint64_t*)sdescs_u;
    const uint64_t* D = (const uint64_t*)sdescs_d;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    const bool hadvec = !ink_no_hadvec() && nslots == 8;
    int nky = 1;
    { const char* e = getenv("INKWELL_MOE_NKY"); if (e) nky = atoi(e); if (nky < 1) nky = 1; }
    int nky_s = 1;
    { const char* e = getenv("INKWELL_SGEMV_NKY"); if (e) nky_s = atoi(e); if (nky_s < 1) nky_s = 1; }
    static int no_sgemv = -1;
    if (no_sgemv < 0) no_sgemv = getenv("INKWELL_NO_SGEMV") ? atoi(getenv("INKWELL_NO_SGEMV")) : 0;
    const bool sg = !no_sgemv;                  // stream-decode gemv replaces v1+v2 serial
    static int v2_env = -1;
    if (v2_env < 0) v2_env = getenv("INKWELL_MOE_V2") ? atoi(getenv("INKWELL_MOE_V2")) : 0;
    // Descriptor eligibility is immutable and was scanned on the host at load;
    // the tables themselves may be device-resident and are never dereferenced here.
    int v2_ok = !sg && (v2_env > 0) && nky == 1 && v2_descs_ok;
    static int* e8 = nullptr;
    if (v2_ok && !e8) {
        cudaMalloc(&e8, 65 * 4);
        int h[65]; for (int i = 0; i < 65; ++i) h[i] = i;
        cudaMemcpy(e8, h, 260, cudaMemcpyHostToDevice);
    }
    static uint32_t* xq8c = nullptr; static float* xqs8c = nullptr;
    static uint32_t* xq8d = nullptr; static float* xqs8d = nullptr;
    const bool d4 = v2_ok && getenv("INKWELL_MOE_D4A");
    if (d4 && !xq8c) {
        cudaMalloc(&xq8c, 64 * 2304); cudaMalloc(&xqs8c, 64 * 144 * 4);
        cudaMalloc(&xq8d, 64 * 2304); cudaMalloc(&xqs8d, 64 * 144 * 4);
    }
    const int nky_eff = sg ? nky_s : nky;       // nky of the kernel actually launched
    // gate+up fused in one launch: blockIdx.z selects the descriptor set.
    // Hit slots run BEFORE ev_join so their gemv overlaps the bulk copy of
    // the missed experts on the side stream; miss slots run right after.
    dim3 gin(hadvec ? (KG + 255) / 256 : (KG + 511) / 512, nslots, 2);
    if (hadvec) inkwell_exl3_had_in_moe_vec<<<gin, 64, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG,
                                                (const int*)miss, 1);
    else inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG,
                                                (const int*)miss, 1, nullptr, nullptr);
    if (nky_eff > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    // GU stays fp32
    if (sg) {
        dim3 gg((NG / 16 + 7) / 8, nslots, 2 * nky_s);
        launch_moe_s(gg, st, G, U, ids, xh, (const float*)gu, 2 * NG,
                        (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky_s,
                        (const int*)miss, 1);
    } else if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(NG / 64, nslots, 2), 256, 0, st>>>(G, U, ids, e8, xh,
        (float*)gu, 2 * NG, (float*)dn, ND, NG, KG / 16, NG / 16, nullptr, (const int*)miss, 1,
        nullptr, nullptr, nullptr, nullptr);
    else {
        dim3 gg(NG / 64, nslots, 2 * nky);
        inkwell_exl3_gemv_moe<<<gg, 256, 0, st>>>(G, U, ids, xh, (const float*)gu, 2 * NG,
                                                 (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky,
                                                 (const int*)miss, 1);
    }
    cudaStreamWaitEvent(st, ev_join, 0);
    if (hadvec) inkwell_exl3_had_in_moe_vec<<<gin, 64, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG,
                                                (const int*)miss, 2);
    else inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG,
                                                (const int*)miss, 2, nullptr, nullptr);
    if (sg) {
        dim3 gg((NG / 16 + 7) / 8, nslots, 2 * nky_s);
        launch_moe_s(gg, st, G, U, ids, xh, (const float*)gu, 2 * NG,
                        (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky_s,
                        (const int*)miss, 2);
    } else if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(NG / 64, nslots, 2), 256, 0, st>>>(G, U, ids, e8, xh,
        (float*)gu, 2 * NG, (float*)dn, ND, NG, KG / 16, NG / 16, nullptr, (const int*)miss, 2,
        nullptr, nullptr, nullptr, nullptr);
    else {
        dim3 gg(NG / 64, nslots, 2 * nky);
        inkwell_exl3_gemv_moe<<<gg, 256, 0, st>>>(G, U, ids, xh, (const float*)gu, 2 * NG,
                                                 (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky,
                                                 (const int*)miss, 2);
    }
    dim3 gout(hadvec ? (NG + 383) / 384 : (NG + 511) / 512, nslots, 2);
    if (hadvec) {
        inkwell_exl3_had_out_moe_vec<<<gout, 96, 0, st>>>(G, U, ids, dn, gu, NG, 2 * NG, 0, ND, NG);
        inkwell_situ_had_in_moe<<<dim3(3, nslots), 96, 0, st>>>(gu, D, ids, xh, 4.f, 25.f, NG);
    } else {
        inkwell_exl3_had_out_moe<<<gout, 128, 0, st>>>(G, U, ids, dn, gu, NG, 2 * NG, 0, ND, NG);
        inkwell_situ_pair<<<(unsigned)((nslots * NG + 255) / 256), 256, 0, st>>>(gu, nullptr, hid, 4.f, 25.f, NG, nslots);
        dim3 din((KD + 511) / 512, nslots);
        inkwell_exl3_had_in_moe<<<din, 128, 0, st>>>(D, nullptr, ids, hid, xh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
    }
    if (nky_eff > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    if (d4) inkwell_exl3_quant_x_i8p<<<dim3(nslots, KD / 16), 16, 0, st>>>(xh, xq8c, xqs8c, KD);
    if (sg) {
        dim3 gd((ND / 16 + 7) / 8, nslots, nky_s);
        launch_moe_s(gd, st, D, nullptr, ids, xh, nullptr, 0,
                        dn, ND, 0, KD / 16, ND / 16, KD, ND, nky_s, nullptr, 0);
    } else if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(ND / 64, nslots, 1), 256, 0, st>>>(D, nullptr, ids, e8, xh,
        nullptr, 0, (float*)dn, ND, 0, KD / 16, ND / 16, nullptr, nullptr, 0,
        d4 ? xq8c : nullptr, d4 ? xqs8c : nullptr, nullptr, nullptr);
    else {
        dim3 gd(ND / 64, nslots, nky);
        inkwell_exl3_gemv_moe<<<gd, 256, 0, st>>>(D, nullptr, ids, xh, nullptr, 0,
                                                 dn, ND, 0, KD / 16, ND / 16, KD, ND, nky, nullptr, 0);
    }
    if (hadvec) {
        inkwell_had_out_wsum_moe<<<dim3(ND / 128), 256, 0, st>>>(D, ids, dn, w, out, ND);
    } else {
        dim3 dout((ND + 511) / 512, nslots, 1);
        inkwell_exl3_had_out_moe<<<dout, 128, 0, st>>>(D, nullptr, ids, dn, xh /*reuse xh as y*/, ND, ND, 0, ND, 0);
        inkwell_expert_wsum<<<(unsigned)((ND + 255) / 256), 256, 0, st>>>(xh, w, nullptr, out, ND, nslots);
    }
}

extern "C" void ink_launch_exl3_stage(const void* descs_g, const void* descs_u,
                                    const void* descs_d, const int* ids,
                                    void* staging, void* sg, void* su, void* sd,
                                    void* stream) {
    cudaStream_t st = (cudaStream_t)stream;
    inkwell_exl3_moe_stage<<<24, 256, 0, st>>>(
        (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
        ids, (uint8_t*)staging, (uint64_t*)sg, (uint64_t*)su, (uint64_t*)sd);
}

// Fused convolve step: reads state[3, n_sconv] region + x_in[ch], shifts
// x[1..4] back into state, computes ssm_conv + optional per-head taps +
// residual + scale — one kernel replaces ~150 ggml ops per conv site.
extern "C" __global__ void inkwell_conv_step(
    const float* __restrict__ x_in,     // [ch]
    float* __restrict__ state,          // [3, n_sconv] base of this slot
    int region_off,                     // channel offset in n_sconv
    const float* __restrict__ w,        // [4, ch] conv taps
    const float* __restrict__ dyn,      // [4, n_heads] or nullptr
    int ch, int n_heads, float scale,
    float* __restrict__ out,            // [ch]
    float* __restrict__ snap)           // [3, n_sconv] per-position state save (nullable)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    // state is [3, n_sconv] contiguous: element (row r, channel c) at
    // st[3*region_off + 3*c + r]
    float* st = state + 3 * (size_t)region_off;
    const float x0 = st[3 * c], x1 = st[3 * c + 1], x2 = st[3 * c + 2];
    const float x3 = x_in[c];
    // shift: state = x[1..4]
    st[3 * c] = x1; st[3 * c + 1] = x2; st[3 * c + 2] = x3;
    if (snap) { float* sp = snap + 3 * (size_t)region_off; sp[3 * c] = x1; sp[3 * c + 1] = x2; sp[3 * c + 2] = x3; }
    // w is [4, ch] contiguous: element (tap j, channel c) at w[4*c + j]
    float acc = w[4 * c] * x0 + w[4 * c + 1] * x1 + w[4 * c + 2] * x2 + w[4 * c + 3] * x3;
    if (dyn) {
        // dyn is [4, n_heads]: element (lag, h) at dyn[4*h + lag];
        // lag 0 pairs with x[3] (cur), lag 3 with x[0] (oldest)
        const int h = c / (ch / n_heads);
        acc += dyn[4 * h] * x3 + dyn[4 * h + 1] * x2
             + dyn[4 * h + 2] * x1 + dyn[4 * h + 3] * x0;
    }
    out[c] = (x3 + acc) * scale;
}

extern "C" void ink_launch_conv_step(const float* x_in, float* state, int region_off,
        const float* w, const float* dyn, int ch, int n_heads, float scale,
        float* out, float* snap, void* stream) {
    const int thr = ch < 256 ? ch : 256;
    inkwell_conv_step<<<(ch + thr - 1) / thr, thr, 0, (cudaStream_t)stream>>>(
        x_in, state, region_off, w, dyn, ch, n_heads, scale, out, snap);
}

// ================= union-batched (verify ntok) MoE =================
// Expand 8 tokens × top-8 (64 selections) into (expert,consumer) pairs sorted
// by expert id. ~20 unique experts/visit typical → the gemv reads each expert
// once for all its consumer tokens (LFM-style weight sharing for MoE).
__global__ void inkwell_exl3_union_expand(
    const int* __restrict__ ids8,          // [64] token-major: ids8[t*8+j]
    int nslots,                            // bound on populated slots (kp*8 spec)
    int* __restrict__ uids,                // [64] unique expert ids
    int* __restrict__ pnexp,               // [1]
    int* __restrict__ eids,                // [64] expert id per expanded slot
    int* __restrict__ euni,                // [64] union index per expanded slot
    int* __restrict__ etok,                // [64] token index per expanded slot
    int* __restrict__ estart,              // [65] consumer range start per union idx
    int* __restrict__ epos,                // [64] (t*8+j) -> expanded slot
    const uint64_t* __restrict__ descs_g, const uint64_t* __restrict__ descs_u,
    const uint64_t* __restrict__ descs_d,
    uint64_t* __restrict__ udesc_g, uint64_t* __restrict__ udesc_u,
    uint64_t* __restrict__ udesc_d) {
    __shared__ int sh_cnt[128], sh_start[128], sh_uidx[128];
    const int e = threadIdx.x;
    int cnt = 0;
    for (int s = 0; s < nslots; ++s) cnt += (ids8[s] == e);
    sh_cnt[e] = cnt;
    __syncthreads();
    if (e == 0) {
        // Split experts with >8 consumers into multiple union entries so the
        // gemv's per-slot consumer arrays (xhb[8], s[8]) never overflow.
        int off = 0, nu = 0;
        for (int i = 0; i < 128; ++i) {
            int c = sh_cnt[i];
            sh_start[i] = off;
            sh_uidx[i] = nu;
            while (c > 0) {
                const int take = min(8, c);
                uids[nu] = i;
                estart[nu] = off;
                ++nu; off += take; c -= take;
            }
        }
        for (int i = nu; i <= 64; ++i) estart[i] = off;
        for (int i = nu; i < 64; ++i) { eids[i] = 0; euni[i] = 0; etok[i] = 0; }
        *pnexp = nu;
    }
    __syncthreads();
    if (cnt) {
        int p = sh_start[e];
        for (int s = 0; s < nslots; ++s) {
            if (ids8[s] == e) {
                eids[p] = e; euni[p] = sh_uidx[e] + (p - sh_start[e]) / 8; etok[p] = s >> 3; epos[s] = p;
                ++p;
            }
        }
        if (udesc_g) {
            // uncached path: stage the union's desc entries indexed by expert id
            const uint64_t* sg = descs_g + e * 4;
            const uint64_t* su = descs_u + e * 4;
            const uint64_t* sd = descs_d + e * 4;
            uint64_t* og = udesc_g + e * 4;
            uint64_t* ou = udesc_u + e * 4;
            uint64_t* od = udesc_d + e * 4;
            for (int q = 0; q < 4; ++q) { og[q] = sg[q]; ou[q] = su[q]; od[q] = sd[q]; }
        }
    }
}

// gemv_moe over the union: blockIdx.y = union index; the expert's tile is read
// once and applied to estart[u+1]-estart[u] consumer x-vectors (packed
// consecutive expanded slots). Output dn[expanded_slot][n].
__global__ void inkwell_exl3_gemv_moe_ntok(const uint64_t* __restrict__ descs,
                                      const uint64_t* __restrict__ descs2,
                                      const int* __restrict__ uids,
                                      const int* __restrict__ estart,
                                      const float* __restrict__ xh,
                                      const float* __restrict__ xh2, int xh2s,
                                      float* __restrict__ t, int tslot, int tzoff,
                                      int tk, int tn, int k, int n, int nky,
                                      const int* __restrict__ missf, int skipmode,
                                      const int* __restrict__ pnexp) {
    const int slot = blockIdx.y;
    if (pnexp && slot >= *pnexp) return;
    if (skipmode && ((missf[slot] != 0) == (skipmode == 1))) return;
    const int nsel = descs2 ? 2 : 1;
    const int ky = blockIdx.z / nsel;
    const int zsel = blockIdx.z % nsel;
    const uint64_t* d = (zsel ? descs2 : descs) + uids[slot] * 4;
    const int sx0 = estart[slot];
    const int nc = estart[slot + 1] - sx0;
    const int xstride = zsel ? xh2s : k;
    const float* xhb_g = (zsel ? xh2 : xh) + (size_t)sx0 * xstride;
    const int tk_lo = (ky * tk) / nky;
    const int tk_hi = ((ky + 1) * tk) / nky;
    const uint32_t* tile_base = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int g = threadIdx.x >> 6;
    const int u = threadIdx.x & 63;
    const int c = u >> 2, rp = u & 3;
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk);
    __shared__ uint32_t rings[4 * 8 * 32];
    __shared__ float xhb[8][16][8];   // [consumer][fragment][batch row]
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    float s[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float sx[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    const bool live = tn_i < tn;
    uint32_t widx[4];
    {
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int nw = r_bits >> 5;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = rp + 4 * q;
            const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                        | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
            const int s_i = halfk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
            const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
            const int i0 = b0 >> 5;
            const int s0 = 48 - (b0 & 31);
            const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
            widx[q] = (uint32_t)(i0 | (i1 << 6) | (s0 << 12));
        }
    }
    {
        const int tkn0 = min(8, tk_hi - tk_lo);
        for (int idx = threadIdx.x; idx < tkn0 * 16 * nc; idx += 256) {
            const int cc = idx / (tkn0 * 16), r16 = idx % (tkn0 * 16);
            xhb[cc][r16 & 15][r16 >> 4] =
                xhb_g[(size_t)cc * xstride + (tk_lo + (r16 >> 4)) * 16 + (r16 & 15)];
        }
        if (live && u < (w + 3) / 4) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn0; ++b) {
                const uint4 v4 = *(const uint4*)&tile_base[((size_t)((tk_lo + b) * tn + tn_i)) * (size_t)w + u * 4];
                rg[b * 32 + u * 4] = v4.x; rg[b * 32 + u * 4 + 1] = v4.y;
                rg[b * 32 + u * 4 + 2] = v4.z; rg[b * 32 + u * 4 + 3] = v4.w;
            }
        }
        if (tkn0 < 8) {
            for (int idx = threadIdx.x; idx < nc * 16 * (8 - tkn0); idx += 256) {
                const int cc = idx / (16 * (8 - tkn0)), r = idx % (16 * (8 - tkn0));
                xhb[cc][r & 15][tkn0 + (r >> 4)] = 0.f;
            }
        }
    }
    __syncthreads();
    for (int tk0 = tk_lo; tk0 < tk_hi; tk0 += 8) {
        const int tkn = min(8, tk_hi - tk0);
        const int nk0 = tk0 + 8;
        const int tkn2 = min(8, tk_hi - nk0);
        uint4 pfb4[8];
        int pfidx[4]; float pfx[4]; int npf = 0;
        if (live && nk0 < tk_hi && u < (w + 3) / 4) {
            for (int b = 0; b < tkn2; ++b)
                pfb4[b] = *(const uint4*)&tile_base[((size_t)((nk0 + b) * tn + tn_i)) * (size_t)w + u * 4];
        }
        if (nk0 < tk_hi) {
            for (int idx = threadIdx.x; idx < nc * tkn2 * 16; idx += 256) {
                if (npf < 4) {
                    pfidx[npf] = idx;
                    const int cc = idx / (tkn2 * 16), r16 = idx % (tkn2 * 16);
                    pfx[npf] = xhb_g[(size_t)cc * xstride + (nk0 + (r16 >> 4)) * 16 + (r16 & 15)];
                    ++npf;
                }
            }
        }
        if (live) {
            const uint32_t* rg = rings + g * 256;
            // decode all 32 bs values once (b x q), then per-consumer float4 FMAs
            // over the transposed xhb[c][frag][b]: ~2x fewer instr per MAC.
            float bs4[4][8];
#ifndef MNTOK_NOCOMP
            for (int b = 0; b < tkn; ++b) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t wi = widx[q];
                    const uint64_t v = ((uint64_t)rg[b * 32 + (wi & 63)] << 32)
                                     | (uint64_t)rg[b * 32 + ((wi >> 6) & 63)];
                    const uint32_t w16 = (uint32_t)(v >> (wi >> 12)) & 0xffffu;
                    bs4[q][b] = (float)(int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
                }
            }
            for (int c2 = 0; c2 < nc; ++c2) {
                float s2 = 0.f, sx2 = 0.f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float4 x0 = *(const float4*)&xhb[c2][rp + 4 * q][0];
                    const float4 x1 = *(const float4*)&xhb[c2][rp + 4 * q][4];
                    s2 += x0.x * bs4[q][0] + x0.y * bs4[q][1] + x0.z * bs4[q][2] + x0.w * bs4[q][3]
                        + x1.x * bs4[q][4] + x1.y * bs4[q][5] + x1.z * bs4[q][6] + x1.w * bs4[q][7];
                    sx2 += (x0.x + x0.z) + (x0.y + x0.w) + (x1.x + x1.z) + (x1.y + x1.w);
                }
                s[c2] += s2;
                sx[c2] += sx2;
            }
#endif
        }
        __syncthreads();
        if (live && nk0 < tk_hi && u < (w + 3) / 4) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn2; ++b) {
                rg[b * 32 + u * 4] = pfb4[b].x; rg[b * 32 + u * 4 + 1] = pfb4[b].y;
                rg[b * 32 + u * 4 + 2] = pfb4[b].z; rg[b * 32 + u * 4 + 3] = pfb4[b].w;
            }
        }
        if (nk0 < tk_hi) {
            for (int i = 0; i < npf; ++i) {
                const int idx = pfidx[i];
                const int cc = idx / (tkn2 * 16), r16 = idx % (tkn2 * 16);
                xhb[cc][r16 & 15][r16 >> 4] = pfx[i];
            }
            if (tkn2 < 8) {
                for (int idx = threadIdx.x; idx < nc * 16 * (8 - tkn2); idx += 256) {
                    const int cc = idx / (16 * (8 - tkn2)), r = idx % (16 * (8 - tkn2));
                    xhb[cc][r & 15][tkn2 + (r >> 4)] = 0.f;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int c2 = 0; c2 < 8; ++c2) {
        s[c2] += __shfl_xor_sync(0xffffffffu, s[c2], 1);
        sx[c2] += __shfl_xor_sync(0xffffffffu, sx[c2], 1);
        s[c2] += __shfl_xor_sync(0xffffffffu, s[c2], 2);
        sx[c2] += __shfl_xor_sync(0xffffffffu, sx[c2], 2);
    }
    if (live && rp == 0) {
        for (int c2 = 0; c2 < nc; ++c2) {
            float* dstp = &t[(size_t)(sx0 + c2) * n + zsel * tzoff + tn_i * 16 + c];
            const float vv = fmaf(KINV, s[c2], KOFF * sx[c2]);
            if (nky > 1) atomicAdd(dstp, vv); else *dstp = vv;
        }
    }
}

// out[t*n+i] = sum_j w8[t*8+j] * dn[epos[t*8+j]*n + i]
__global__ void inkwell_exl3_expert_wsum8(const float* __restrict__ dn,
                                          const float* __restrict__ w8,
                                          const int* __restrict__ epos,
                                          float* __restrict__ out, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += w8[t * 8 + j] * dn[(size_t)epos[t * 8 + j] * n + i];
    out[(size_t)t * n + i] = acc;
}

// Union-batched verify MoE: one launch chain for 8 tokens.
// Scratch layout (caller-owned): xh_e[64][2304] gu_e[64][2304] hid_e[64][1152]
// dn_e[64][2304]; small bufs uids/eids/euni/etok/epos[64], estart[65], unexp[1].
extern "C" void ink_launch_exl3_moe_ntok(
    const void* descs_g, const void* descs_u, const void* descs_d,
    const float* x8, const int* ids8, const float* w8, float* out8,
    float* xh_e, float* gu_e, float* hid_e, float* dn_e,
    int* uids, int* eids, int* euni, int* etok, int* estart, int* epos, int* unexp,
    void* cache, int* tag, int* map, int* miss, int* cnt, int k, int64_t slot_proj,
    void* sdescs_g, void* sdescs_u, void* sdescs_d,
    int on_dev, int v2_descs_ok, void* stream, int key_base, int nslots, int* hm,
    const int* gexp, uint64_t* soff, int* fl) {
    const cudaStream_t st = (cudaStream_t)stream;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    const bool cached = cache != nullptr;
    if (nslots <= 0) nslots = 64;
    if (cached) {
        inkwell_exl3_union_expand<<<1, 128, 0, st>>>(ids8, nslots, uids, unexp,
            eids, euni, etok, estart, epos, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr);
    } else {
        // uncached/on_dev: write per-expert-id desc tables directly (no staging)
        inkwell_exl3_union_expand<<<1, 128, 0, st>>>(ids8, nslots, uids, unexp,
            eids, euni, etok, estart, epos,
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d);
    }
    int nexp = 64;  // union cap; kernels early-exit on *unexp
    const int dbg = getenv("INKWELL_MNTOK_DBG") ? atoi(getenv("INKWELL_MNTOK_DBG")) : 0;
    if (dbg) { cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok expand ok nexp=%d err=%d\n", nexp, (int)e); }
    if (dbg == 1) return;

    static cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
    static cudaStream_t ss = nullptr;
    if (cached && !ev_fork) {
        cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming);
        cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking);
    }
    int nky = 1;
    { const char* e = getenv("INKWELL_MOE_NKY"); if (e) nky = atoi(e); if (nky < 1) nky = 1; }
    // v2 kernel: ka <= 4 (WM=32) and pairable window layout, all experts.
    static int v2_env = -1;
    if (v2_env < 0) v2_env = getenv("INKWELL_MOE_V2") ? atoi(getenv("INKWELL_MOE_V2")) : 0;
    // Descriptor eligibility is immutable and was scanned on the host at load;
    // the tables themselves may be device-resident and are never dereferenced here.
    int v2_ok = (v2_env > 0) && nky == 1 && v2_descs_ok;

    const uint64_t* G = (const uint64_t*)sdescs_g;
    const uint64_t* U = (const uint64_t*)sdescs_u;
    const uint64_t* D = (const uint64_t*)sdescs_d;
    static int skip_cc = -1;
    if (skip_cc < 0) { const char* e = getenv("INKWELL_SKIP_CCOPY"); skip_cc = e ? atoi(e) : 0; }
    if (cached) {
        if (soff) {
            inkwell_exl3_kslot_resolve<<<1, 32, 0, st>>>(uids,
                tag, (uint64_t*)soff, map, miss, cnt, nexp, k,
                gexp, fl, nullptr, nullptr, nullptr, unexp,
                key_base, getenv("INKWELL_RESOLVE_DIRECT") ? 1 : 0, hm);
        } else {
            inkwell_exl3_cache_resolve<<<1, 32, 0, st>>>(uids,
                tag, map, miss, cnt, nexp, k, nullptr, nullptr, nullptr, unexp,
                key_base, getenv("INKWELL_RESOLVE_DIRECT") ? 1 : 0, hm);
        }
        inkwell_exl3_moe_ccopy<<<dim3(nexp * 3, 1), 32, 0, st>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            uids, (uint8_t*)cache, map, miss,
            nullptr, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, skip_cc ? 3 : 1, unexp, soff);
        cudaEventRecord(ev_fork, st);
        cudaStreamWaitEvent(ss, ev_fork, 0);
        inkwell_exl3_moe_ccopy<<<dim3(nexp * 3, 16), 128, 0, ss>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            uids, (uint8_t*)cache, map, miss,
            nullptr, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, skip_cc ? 3 : 2, unexp, soff);
        cudaEventRecord(ev_join, ss);
    }
    dim3 gin((KG + 511) / 512, 64, 2);
    dim3 gg(NG / 64, nexp, 2 * nky);
    dim3 gout((NG + 511) / 512, 64, 2);
    dim3 din((KD + 511) / 512, 64);
    dim3 gd(ND / 64, nexp, nky);
    dim3 dout((ND + 511) / 512, 64, 1);
    if (nky > 1) cudaMemsetAsync(dn_e, 0, 64 * ND * 4, st);
    if (dbg) { cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok pre-gu err=%d\n", (int)e); }
    if (dbg == 2) return;
    if (cached) {
        // hits first: resident already, gemv overlaps the miss bulk copy on ss
        inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, eids, x8, xh_e, gu_e,
            2 * NG, KG, miss, 2, euni, etok);
        if (dbg) { cudaStreamSynchronize(st); fprintf(stderr, "  hadin2 err=%d\n", (int)cudaGetLastError()); }
        if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(NG / 64, 64, 2), 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, unexp, miss, 2);
        else inkwell_exl3_gemv_moe_ntok<<<gg, 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, KG, ND, nky, miss, 2, unexp);
        cudaStreamWaitEvent(st, ev_join, 0);
        inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, eids, x8, xh_e, gu_e,
            2 * NG, KG, miss, 1, euni, etok);
        if (dbg) { cudaStreamSynchronize(st); fprintf(stderr, "  hadin1 err=%d\n", (int)cudaGetLastError()); }
        if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(NG / 64, 64, 2), 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, unexp, miss, 1);
        else inkwell_exl3_gemv_moe_ntok<<<gg, 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, KG, ND, nky, miss, 1, unexp);
        if (dbg) { cudaStreamSynchronize(st); fprintf(stderr, "  gemv1 err=%d\n", (int)cudaGetLastError()); }
    } else {
        inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, eids, x8, xh_e, gu_e,
            2 * NG, KG, nullptr, 0, euni, etok);
        if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(NG / 64, 64, 2), 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, unexp);
        else inkwell_exl3_gemv_moe_ntok<<<gg, 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, KG, ND, nky, nullptr, 0, unexp);
    }
    if (dbg) { cudaStreamSynchronize(st); cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok gu done err=%d\n", (int)e); }
    if (dbg == 3) return;
    inkwell_exl3_had_out_moe<<<gout, 128, 0, st>>>(G, U, eids, dn_e, gu_e, NG, 2 * NG, 0, ND, NG);
    inkwell_situ_pair<<<(unsigned)((64 * NG + 255) / 256), 256, 0, st>>>(gu_e, nullptr, hid_e, 4.f, 25.f, NG, 64);
    if (dbg) { cudaStreamSynchronize(st); cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok situ done err=%d\n", (int)e); }
    if (dbg == 4) return;
    // down chain: had_in reads hid_e[slot] per expanded slot (etok null → slot offset)
    inkwell_exl3_had_in_moe<<<din, 128, 0, st>>>(D, nullptr, eids, hid_e, xh_e,
        nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
    if (nky > 1) cudaMemsetAsync(dn_e, 0, 64 * ND * 4, st);
    if (v2_ok) exl3v2::gemv_moe_n<2><<<dim3(ND / 64, 64, 1), 256, 0, st>>>(D, nullptr, uids, estart, xh_e,
        nullptr, 0, dn_e, ND, 0, KD / 16, ND / 16, unexp);
    else inkwell_exl3_gemv_moe_ntok<<<gd, 256, 0, st>>>(D, nullptr, uids, estart, xh_e,
        nullptr, 0, dn_e, ND, 0, KD / 16, ND / 16, KD, ND, nky, nullptr, 0, unexp);
    if (dbg) { cudaStreamSynchronize(st); cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok dgemv done err=%d\n", (int)e); }
    if (dbg == 5) return;
    inkwell_exl3_had_out_moe<<<dout, 128, 0, st>>>(D, nullptr, eids, dn_e, xh_e, ND, ND, 0, ND, 0);
    inkwell_exl3_expert_wsum8<<<dim3((ND + 255) / 256, 8), 256, 0, st>>>(xh_e, w8, epos, out8, ND);
    if (dbg) { cudaStreamSynchronize(st); cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok all done err=%d\n", (int)e); }
}

// ======================== DSpark draft helper kernels ========================
// All f32, single-block-per-row launches on stream 0. Sizes are tiny
// (2304..201024) — latency-bound by design, no graph.

// mix + rmsnorm: out[t] = rmsnorm( sum_j alpha[j]*feats_tap_j[row t] ) * w
// feats layout [4][nt][ch]; one block per row, 256 threads.
extern "C" __global__ void inkwell_draft_mix(
    const float* __restrict__ feats, const float* __restrict__ alpha,
    const float* __restrict__ w, float* __restrict__ out,
    int nt, int ch, float eps)
{
    const int t = blockIdx.x;
    const int c0 = threadIdx.x;
    __shared__ float ss[256];
    float acc = 0.f;
    for (int c = c0; c < ch; c += 256) {
        float v = 0.f;
        for (int j = 0; j < 4; ++j)
            v += alpha[j] * feats[(size_t)j * nt * ch + (size_t)t * ch + c];
        acc += v * v;
    }
    ss[c0] = acc;
    __syncthreads();
    for (int s = 128; s; s >>= 1) { if (c0 < s) ss[c0] += ss[c0 + s]; __syncthreads(); }
    const float inv = rsqrtf(ss[0] / ch + eps);
    for (int c = c0; c < ch; c += 256) {
        float v = 0.f;
        for (int j = 0; j < 4; ++j)
            v += alpha[j] * feats[(size_t)j * nt * ch + (size_t)t * ch + c];
        out[(size_t)t * ch + c] = v * inv * w[c];
    }
}

// out[r] = rmsnorm(x[r]) * w ; res variant adds res to x first and saves the sum.
extern "C" __global__ void inkwell_draft_rms(
    const float* __restrict__ x, const float* __restrict__ res,
    const float* __restrict__ w, float* __restrict__ out, float* __restrict__ res_out,
    int ch, float eps)
{
    const int t = blockIdx.x;
    const int c0 = threadIdx.x;
    __shared__ float ss[256];
    float acc = 0.f;
    for (int c = c0; c < ch; c += 256) {
        float v = x[(size_t)t * ch + c] + (res ? res[(size_t)t * ch + c] : 0.f);
        acc += v * v;
    }
    ss[c0] = acc;
    __syncthreads();
    for (int s = 128; s; s >>= 1) { if (c0 < s) ss[c0] += ss[c0 + s]; __syncthreads(); }
    const float inv = rsqrtf(ss[0] / ch + eps);
    for (int c = c0; c < ch; c += 256) {
        float v = x[(size_t)t * ch + c] + (res ? res[(size_t)t * ch + c] : 0.f);
        out[(size_t)t * ch + c] = v * inv * w[c];
        if (res_out) res_out[(size_t)t * ch + c] = v;
    }
}

// rope (rotate-halves) in place on x[n][nh][hd]; pos per row.
extern "C" __global__ void inkwell_draft_rope(
    float* __restrict__ x, const int* __restrict__ pos, int nh, int hd, float theta)
{
    const int r = blockIdx.y, h = blockIdx.x, i = threadIdx.x; // i < hd/2
    const int hh = hd / 2;
    if (i >= hh) return;
    const float f = pos[r] * powf(theta, -2.f * i / hd);
    const float co = cosf(f), si = sinf(f);
    float* p = x + ((size_t)r * nh + h) * hd;
    const float x1 = p[i], x2 = p[i + hh];
    p[i] = x1 * co - x2 * si;
    p[i + hh] = x2 * co + x1 * si;
}

// write k/v block rows into ctx caches at absolute positions pos[r].
extern "C" __global__ void inkwell_draft_kvwrite(
    float* __restrict__ kc, float* __restrict__ vc,
    const float* __restrict__ k, const float* __restrict__ v,
    const int* __restrict__ pos, int kvch)
{
    const int r = blockIdx.x, c = threadIdx.x;
    const size_t dst = (size_t)pos[r] * kvch + c;
    const size_t src = (size_t)r * kvch + c;
    kc[dst] = k[src];
    vc[dst] = v[src];
}

// attention: q[8][nh][hd] attends keys = kc rows [win_lo,c) + kb rows [0,8)
// (the block's own keys, non-causal). One block per (row, head): scores in
// dynamic smem, then softmax, then weighted V sum. kv heads nkv, g = nh/nkv.
extern "C" __global__ void inkwell_draft_attn(
    const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
    const float* __restrict__ kb, const float* __restrict__ vb, float* __restrict__ out,
    int c, int win_lo, int nb, int nkv, int g, int hd)
{
    extern __shared__ float s[];
    const int r = blockIdx.y, h = blockIdx.x;
    const int kvh = h / g;
    const int nctx = c - win_lo;
    const int T = nctx + nb;
    const int tid = threadIdx.x;
    const float* qr = q + ((size_t)r * (nkv * g) + h) * hd;
    const float scale = rsqrtf((float)hd);
    for (int t = tid; t < T; t += blockDim.x) {
        const float* kp = (t < nctx)
            ? kc + ((size_t)(win_lo + t) * nkv + kvh) * hd
            : kb + ((size_t)(t - nctx) * nkv + kvh) * hd;
        float d = 0.f;
        for (int e = 0; e < hd; ++e) d += qr[e] * kp[e];
        s[t] = d * scale;
    }
    __syncthreads();
    // softmax
    __shared__ float red[256];
    float mx = -1e30f;
    for (int t = tid; t < T; t += blockDim.x) mx = fmaxf(mx, s[t]);
    red[tid] = mx; __syncthreads();
    for (int st = 128; st; st >>= 1) { if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]); __syncthreads(); }
    mx = red[0];
    float sm = 0.f;
    for (int t = tid; t < T; t += blockDim.x) { s[t] = expf(s[t] - mx); sm += s[t]; }
    __syncthreads();
    red[tid] = sm; __syncthreads();
    for (int st = 128; st; st >>= 1) { if (tid < st) red[tid] += red[tid + st]; __syncthreads(); }
    const float inv = 1.f / red[0];
    // out[r][h][e] = sum_t s[t]*v[t]
    for (int e = tid; e < hd; e += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < T; ++t) {
            const float* vp = (t < nctx)
                ? vc + ((size_t)(win_lo + t) * nkv + kvh) * hd
                : vb + ((size_t)(t - nctx) * nkv + kvh) * hd;
            acc += s[t] * vp[e];
        }
        out[((size_t)r * (nkv * g) + h) * hd + e] = acc * inv;
    }
}

// x = silu(g) * u ; g,u,x [n][ff]
extern "C" __global__ void inkwell_draft_silu(const float* __restrict__ g,
        const float* __restrict__ u, float* __restrict__ x, int ff)
{
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < ff; c += blockDim.x) {
        const float g_ = g[(size_t)t * ff + c];
        x[(size_t)t * ff + c] = (g_ / (1.f + expf(-g_))) * u[(size_t)t * ff + c];
    }
}

// GRU-ish markov update: raw = rnn_proj([state | w1prev | hid | f])
// gt = sig(raw[:r]); cand = tanh(raw[r:2r]); state' = gt*state + (1-gt)*cand;
// latent = tanh(raw[2r:3r])
extern "C" __global__ void inkwell_draft_gru(const float* __restrict__ raw,
        const float* __restrict__ bias,
        float* __restrict__ state, float* __restrict__ latent, int r)
{
    const int c = threadIdx.x;
    if (c >= r) return;
    const float gt = 1.f / (1.f + expf(-(raw[c] + bias[c])));
    const float cand = tanhf(raw[r + c] + bias[r + c]);
    state[c] = gt * state[c] + (1.f - gt) * cand;
    latent[c] = tanhf(raw[2 * r + c] + bias[2 * r + c]);
}

// out = a + b + c elementwise
extern "C" __global__ void inkwell_draft_add3(const float* __restrict__ a,
        const float* __restrict__ b, const float* __restrict__ c_,
        float* __restrict__ out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = a[i] + b[i] + c_[i];
}

// gather bf16 row idx of w[n][ldw] -> f32 out
extern "C" __global__ void inkwell_draft_bf16row(const unsigned short* __restrict__ w,
        int idx, int ldw, float* __restrict__ out)
{
    const int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (c >= ldw) return;
    const unsigned int u = (unsigned int)w[(size_t)idx * ldw + c] << 16;
    out[c] = __uint_as_float(u);
}

// argmax over cap*tanh(z/cap) -> u32
extern "C" __global__ void inkwell_draft_argmax(const float* __restrict__ z,
        int n, float cap, int* __restrict__ out)
{
    __shared__ float mv[256];
    __shared__ int mi[256];
    const int tid = threadIdx.x;
    float bv = -1e30f; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = cap * tanhf(z[i] / cap);
        if (v > bv) { bv = v; bi = i; }
    }
    mv[tid] = bv; mi[tid] = bi;
    __syncthreads();
    for (int s = 128; s; s >>= 1) {
        if (tid < s && mv[tid + s] > mv[tid]) { mv[tid] = mv[tid + s]; mi[tid] = mi[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = mi[0];
}

extern "C" void ink_launch_draft_mix(const float* f, const float* a, const float* w,
        float* o, int rows, int tstride, int ch, float eps, void* s) {
    inkwell_draft_mix<<<rows, 256, 0, (cudaStream_t)s>>>(f, a, w, o, tstride, ch, eps);
}
extern "C" void ink_launch_draft_rms(const float* x, const float* res, const float* w,
        float* o, float* ro, int n, int ch, float eps, void* s) {
    inkwell_draft_rms<<<n, 256, 0, (cudaStream_t)s>>>(x, res, w, o, ro, ch, eps);
}
extern "C" void ink_launch_draft_rope(float* x, const int* pos, int n, int nh, int hd,
        float theta, void* s) {
    dim3 g(nh, n);
    inkwell_draft_rope<<<g, hd / 2, 0, (cudaStream_t)s>>>(x, pos, nh, hd, theta);
}
extern "C" void ink_launch_draft_kvwrite(float* kc, float* vc, const float* k,
        const float* v, const int* pos, int n, int kvch, void* s) {
    inkwell_draft_kvwrite<<<n, kvch, 0, (cudaStream_t)s>>>(kc, vc, k, v, pos, kvch);
}
extern "C" void ink_launch_draft_attn(const float* q, const float* kc, const float* vc,
        const float* kb, const float* vb, float* o, int c, int win_lo, int nb,
        int nkv, int g, int hd, void* s) {
    dim3 gr(nkv * g, nb);
    const int T = c - win_lo + nb;
    inkwell_draft_attn<<<gr, 256, T * 4, (cudaStream_t)s>>>(q, kc, vc, kb, vb, o, c, win_lo, nb, nkv, g, hd);
}
extern "C" void ink_launch_draft_silu(const float* g, const float* u, float* x, int n, int ff, void* s) {
    inkwell_draft_silu<<<n, 256, 0, (cudaStream_t)s>>>(g, u, x, ff);
}
extern "C" void ink_launch_draft_gru(const float* raw, const float* bias, float* st, float* lat, int r, void* s) {
    inkwell_draft_gru<<<1, r, 0, (cudaStream_t)s>>>(raw, bias, st, lat, r);
}
extern "C" void ink_launch_draft_add3(const float* a, const float* b, const float* c,
        float* o, int n, void* s) {
    inkwell_draft_add3<<<(n + 255) / 256, 256, 0, (cudaStream_t)s>>>(a, b, c, o, n);
}
extern "C" void ink_launch_draft_bf16row(const void* w, int idx, int ldw, float* o, void* s) {
    inkwell_draft_bf16row<<<(ldw + 255) / 256, 256, 0, (cudaStream_t)s>>>((const unsigned short*)w, idx, ldw, o);
}
extern "C" void ink_launch_draft_argmax(const float* z, int n, float cap, int* o, void* s) {
    inkwell_draft_argmax<<<1, 256, 0, (cudaStream_t)s>>>(z, n, cap, o);
}

// fused L2 clip: y = (x*w) * min(1, thr/(||(x*w)||_2 + eps)) per column.
// x: [d, h] contiguous; w optional same layout; one block per column.
__global__ void inkwell_clip_l2(const float* __restrict__ x, const float* __restrict__ w,
                                float* __restrict__ y, int d, float thr) {
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    __shared__ float red[8];
    __shared__ float fshr;
    float c = 0.f;
    if (i < d) {
        c = x[(size_t)h * d + i];
        if (w) c *= w[(size_t)h * d + i];
    }
    float v = c * c;
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((i & 31) == 0) red[i >> 5] = v;
    __syncthreads();
    if (i == 0) {
        float n = red[0];
        for (int j = 1; j < (256 >> 5); ++j) n += red[j];
        fshr = fminf(1.f, thr / (sqrtf(n) + 1e-6f));
    }
    __syncthreads();
    if (i < d) y[(size_t)h * d + i] = c * fshr;
}

extern "C" void ink_launch_clip_l2(const float* x, const float* w, float* y,
                                   int d, int h, float thr, void* stream) {
    inkwell_clip_l2<<<h, 256, 0, (cudaStream_t)stream>>>(x, w, y, d, thr);
}

// multi-projection EXL3 gemv: nz independent (trel,suh,svh,x,y,scratch,k,n)
// projections issued on per-z side streams so the cuda-graph capture records
// them as parallel branches (same trick as the MoE D-stage overlap).
static cudaStream_t mz_ss[8] = {};
static cudaEvent_t mz_ev[9] = {};
static cudaEvent_t mz_gev[8] = {};
extern "C" void ink_launch_exl3_gemv_mz(
    const void* const* trel, const void* const* suh, const void* const* svh,
    const float* const* x, float* y, long long ystride,
    float* const* scr, const long long* k, const long long* n,
    const long long* ka, const long long* halfk, int nz, void* stream) {
    if (!mz_ss[0]) {
        for (int i = 0; i < 8; ++i) {
            cudaStreamCreateWithFlags(&mz_ss[i], cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&mz_ev[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&mz_gev[i], cudaEventDisableTiming);
        }
        cudaEventCreateWithFlags(&mz_ev[8], cudaEventDisableTiming);
    }
    cudaStream_t st = (cudaStream_t)stream;
    { static int hm = -1; if (hm < 0) { hm = getenv("INKWELL_NO_HADMERGE") ? 0 : 1; }
      if (hm && nz > 1) {
          // group zs by identical (x, k); the group leader's stream runs one
          // merged had_in covering all members before their gemv tails.
          int glead[8], zgrp[8], ng = 0;
          for (int z = 0; z < nz; ++z) {
              int g = -1;
              for (int gg = 0; gg < ng; ++gg)
                  if (x[glead[gg]] == x[z] && k[glead[gg]] == k[z]) g = gg;
              if (g < 0) { glead[ng] = z; g = ng++; }
              zgrp[z] = g;
          }
          cudaEventRecord(mz_ev[8], st);
          for (int g = 0; g < ng; ++g) {
              HadNArgs a; a.n = 0; a.k = (int)k[glead[g]];
              for (int z = 0; z < nz; ++z) if (zgrp[z] == g) {
                  a.x[a.n] = x[z];
                  a.suh[a.n] = (const uint16_t*)suh[z];
                  a.xh[a.n] = scr[z];
                  ++a.n;
              }
              const int lz = glead[g];
              cudaStreamWaitEvent(mz_ss[lz], mz_ev[8], 0);
              inkwell_exl3_had_in_n<<<dim3((unsigned)(a.k / 128), (unsigned)a.n), 128, 512, mz_ss[lz]>>>(a);
              cudaEventRecord(mz_gev[g], mz_ss[lz]);
          }
          for (int z = 0; z < nz; ++z) {
              if (z != glead[zgrp[z]])
                  cudaStreamWaitEvent(mz_ss[z], mz_gev[zgrp[z]], 0);
              exl3_gemv_tail(trel[z], svh[z], y + (size_t)z * ystride,
                  k[z], n[z], ka[z], halfk[z], scr[z], mz_ss[z]);
              cudaEventRecord(mz_ev[z], mz_ss[z]);
              cudaStreamWaitEvent(st, mz_ev[z], 0);
          }
          return;
      } }
    cudaEventRecord(mz_ev[8], st);
    for (int z = 0; z < nz; ++z) {
        cudaStreamWaitEvent(mz_ss[z], mz_ev[8], 0);
        ink_launch_exl3_gemv(trel[z], suh[z], svh[z], x[z], y + (size_t)z * ystride,
            k[z], n[z], ka[z], halfk[z], scr[z], mz_ss[z]);
        cudaEventRecord(mz_ev[z], mz_ss[z]);
        cudaStreamWaitEvent(st, mz_ev[z], 0);
    }
}

// Batched k/v short-conv projection GEMVs (exact mmvf-256 reduction order).
template <int R>
__global__ void inkwell_kv_sconv_batched(
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

extern "C" void ink_launch_kv_sconv_batched(const void* wk, const void* wv,
        const float* xk, const float* xv, float* dk, float* dv,
        int nrows, int ncols2, void* stream) {
    inkwell_kv_sconv_batched<1><<<dim3((unsigned)nrows, 2), 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)wk, (const __nv_bfloat16*)wv, xk, xv, dk, dv, nrows, ncols2);
}

// ===================== fused router: sigmoid+bias+top-k+normalize =====================
// out[0..8) = normalized*scale weights of the selected experts;
// out[8..16) = selected expert ids as i32 bit patterns (read as f32 tensor).
__global__ void inkwell_router(const float* __restrict__ logits,
                               const float* __restrict__ bias,
                               float* __restrict__ out, int k, int n_exp,
                               float wscale, float wfloor) {
    __shared__ float s_sel[256];
    __shared__ float s_prb[256];
    __shared__ int s_ids[64];
    const int t = threadIdx.x;
    const float l = (t < n_exp) ? logits[t] : -INFINITY;
    const float p = 1.f / (1.f + __expf(-l));
    if (t < n_exp) { s_prb[t] = p; s_sel[t] = p + bias[t]; }
    __syncthreads();
    // top-k by serial argmax rounds (thread 0) — 128-256 elems, ~2us
    if (t == 0) {
        float wsum = 0.f;
        for (int j = 0; j < k; ++j) {
            int bi = -1; float bv = -INFINITY;
            for (int e = 0; e < n_exp; ++e)
                if (s_sel[e] > bv) { bv = s_sel[e]; bi = e; }
            s_ids[j] = bi;
            s_sel[bi] = -INFINITY;
            wsum += s_prb[bi];
        }
        if (wsum < wfloor) wsum = wfloor;
        const float inv = wscale / wsum;
        for (int j = 0; j < k; ++j) {
            out[j] = s_prb[s_ids[j]] * inv;
            out[k + j] = __int_as_float(s_ids[j]);
        }
    }
}

// Parallel 4-warp router for the fixed 128-expert decode gate. The selection
// rounds use the same first-lowest-index tie break as thread-0 serial argmax.
__device__ __forceinline__ void inkwell_router_par_128_body(
        const float* __restrict__ logits,
        const float* __restrict__ bias,
        float* __restrict__ out,
        int k, float wscale, float wfloor) {
    __shared__ float s_prb[128];
    __shared__ float s_wv[4];
    __shared__ int s_wi[4];
    __shared__ int s_best;
    __shared__ int s_ids[16];

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
                const int oi = __shfl_xor_sync(0xffffffffu, bi, m);
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

__global__ void inkwell_router_par_128(const float* __restrict__ logits,
        const float* __restrict__ bias, float* __restrict__ out,
        int k, float wscale, float wfloor) {
    inkwell_router_par_128_body(logits, bias, out, k, wscale, wfloor);
}

extern "C" void ink_launch_router(const void* logits, const void* bias, void* out,
                                  int k, int n_exp, float wscale, float wfloor, void* stream) {
    if (n_exp == 128) {
        inkwell_router_par_128<<<1, 128, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const float*)bias, (float*)out, k, wscale, wfloor);
    } else {
        inkwell_router<<<1, 256, 0, (cudaStream_t)stream>>>(
            (const float*)logits, (const float*)bias, (float*)out, k, n_exp, wscale, wfloor);
    }
}

// Fused mlp.gate BF16 GEMV + router. Rows keep the exact ggml mmvf-256
// accumulation/reduction order; the last block runs the parallel router.
__device__ float inkwell_gate_router_logits[128];
__device__ unsigned int inkwell_gate_router_ticket;

template <int R>
__global__ void inkwell_gate_router_fused_exact(
        const __nv_bfloat16* __restrict__ w,
        const float* __restrict__ x,
        const float* __restrict__ bias,
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
        if (lane == 0) inkwell_gate_router_logits[r0 + r] = v;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(&inkwell_gate_router_ticket, 1u);
    __syncthreads();
    if (ticket != gridDim.x - 1) return;
    __threadfence();
    if (tid == 0) inkwell_gate_router_ticket = 0u;

    __shared__ float s_log[128];
    if (tid < 128) s_log[tid] = __ldcg(&inkwell_gate_router_logits[tid]);
    __syncthreads();
    inkwell_router_par_128_body(s_log, bias, out, k, wscale, wfloor);
}

extern "C" void ink_launch_gate_router(const void* w, const float* x, const void* bias,
        void* out, int nrows, int ncols2, int k, float wscale, float wfloor, void* stream) {
    if (nrows == 128) {
        inkwell_gate_router_fused_exact<4><<<nrows / 4, 256, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)w, x, (const float*)bias, (float*)out,
            ncols2, k, wscale, wfloor);
    }
}

// Fused depth-stack mix, multi-query:
// out[:,qc] = normed(states) @ softmax(scale * normed(states)^T @ q[:,qc])
// states [embd, rows] col-major; q [embd, qcols]; out [embd, qcols].
// rows <= 64, qcols <= 8.
__global__ void inkwell_mix(const float* __restrict__ st,
                            const float* __restrict__ q,
                            float* __restrict__ out,
                            int embd, int rows, int qcols, float scale) {
    __shared__ float s_rms[64];
    __shared__ float s_p[64 * 8];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int j = w; j < rows; j += nw) {
        const float* col = st + (size_t)j * embd;
        float ss = 0.f;
        for (int i = lane; i < embd; i += 32) ss += col[i] * col[i];
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(~0u, ss, m);
        const float rms = sqrtf(ss / (float)embd + 1e-6f);
        if (lane == 0) s_rms[j] = rms;
        for (int qc = 0; qc < qcols; ++qc) {
            float dp = 0.f;
            for (int i = lane; i < embd; i += 32) dp += col[i] / rms * q[(size_t)qc * embd + i];
            for (int m = 16; m; m >>= 1) dp += __shfl_xor_sync(~0u, dp, m);
            if (lane == 0) s_p[j * 8 + qc] = dp;
        }
    }
    __syncthreads();
    if (threadIdx.x < qcols) {
        const int qc = threadIdx.x;
        float mx = -INFINITY;
        for (int j = 0; j < rows; ++j) { s_p[j * 8 + qc] *= scale; mx = fmaxf(mx, s_p[j * 8 + qc]); }
        float d = 0.f;
        for (int j = 0; j < rows; ++j) { s_p[j * 8 + qc] = __expf(s_p[j * 8 + qc] - mx); d += s_p[j * 8 + qc]; }
        for (int j = 0; j < rows; ++j) s_p[j * 8 + qc] /= d;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < embd * qcols; i += (int)blockDim.x) {
        const int c = i / embd, e = i - c * embd;
        float a = 0.f;
        for (int j = 0; j < rows; ++j) a += st[(size_t)j * embd + e] / s_rms[j] * s_p[j * 8 + c];
        out[i] = a;
    }
}
// Fused multi-block mix (default; INKWELL_NO_MIXMB=1 = single-block kernel
// above). The single-block kernel is latency-bound: rows is 1..6 in practice,
// so 1-6 warps each walk 72 dependent global loads per pass. Here block
// (bx, qc) owns 256 output elements of query qc. Its warps first recompute
// rms[j] and p[j][qc] for every row j (rows x 9 KiB, from L2): the lane's 72
// column and query values are preloaded into registers, then the SAME
// lane-strided sums (t = 0..71 in order) and xor-shuffle trees run. Thread 0
// then does the sequential softmax exactly as before, and each thread
// accumulates its element over j in order. One launch, no global scratch.
// Bit-identical to the single-block kernel (md5-checked).
#define MIX_NPL 72   // embd / 32 for embd = 2304; other embd use the old kernel

__global__ void __launch_bounds__(256) inkwell_mix_fused(const float* __restrict__ st,
                                const float* __restrict__ q, float* __restrict__ out,
                                int embd, int rows, float scale) {
    __shared__ float s_rms[64];
    __shared__ float s_p[64];
    const int qc = blockIdx.y;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float qq[MIX_NPL];
#pragma unroll
    for (int t = 0; t < MIX_NPL; ++t) qq[t] = q[(size_t)qc * embd + lane + 32 * t];
    for (int j = w; j < rows; j += nw) {
        const float* col = st + (size_t)j * embd;
        float c[MIX_NPL];
#pragma unroll
        for (int t = 0; t < MIX_NPL; ++t) c[t] = col[lane + 32 * t];
        float ss = 0.f;
#pragma unroll
        for (int t = 0; t < MIX_NPL; ++t) ss += c[t] * c[t];
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(~0u, ss, m);
        const float rms = sqrtf(ss / (float)embd + 1e-6f);
        float dp = 0.f;
#pragma unroll
        for (int t = 0; t < MIX_NPL; ++t) dp += c[t] / rms * qq[t];
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
        for (int u = 0; u < 8; ++u) v[u] = (j0 + u < rows) ? st[(size_t)(j0 + u) * embd + e] : 0.f;
#pragma unroll
        for (int u = 0; u < 8; ++u) if (j0 + u < rows) a += v[u] / s_rms[j0 + u] * s_p[j0 + u];
    }
    out[(size_t)qc * embd + e] = a;
}

extern "C" void ink_launch_mix(const void* st, const void* q, void* out,
                               int embd, int rows, int qcols, float scale, void* stream) {
    static int mb = -1;
    if (mb < 0) mb = getenv("INKWELL_NO_MIXMB") ? 0 : 1;
    if (mb && embd == 32 * MIX_NPL && rows >= 1 && rows <= 64 && qcols >= 1 && qcols <= 8) {
        inkwell_mix_fused<<<dim3((embd + 255) / 256, qcols), 256, 0, (cudaStream_t)stream>>>(
            (const float*)st, (const float*)q, (float*)out, embd, rows, scale);
        return;
    }
    inkwell_mix<<<1, 256, 0, (cudaStream_t)stream>>>(
        (const float*)st, (const float*)q, (float*)out, embd, rows, qcols, scale);
}


// ---- loop-controller glue fusions (graph-rewrite path in runner.rs) ----

// y = (cur - forget*cur) + gain*msg with gain = tanh(g3[0..n]),
// forget = tanh(g3[ng..ng+n]). Replaces tanh x2 + mul/sub/mul/add.
// __f*_rn intrinsics keep the same op order as the unfused chain
// (--use_fast_math could otherwise contract into fma).
__global__ void inkwell_ctrl_apply(const float* __restrict__ g3,
        const float* __restrict__ cur, const float* __restrict__ msg,
        float* __restrict__ y, int n, int ng) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gn = tanhf(g3[i]);
    const float fg = tanhf(g3[ng + i]);
    const float t = __fsub_rn(cur[i], __fmul_rn(fg, cur[i]));
    y[i] = __fadd_rn(t, __fmul_rn(gn, msg[i]));
}
extern "C" void ink_launch_ctrl_apply(const float* g3, const float* cur,
        const float* msg, float* y, int n, int ng, void* stream) {
    inkwell_ctrl_apply<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        g3, cur, msg, y, n, ng);
}

// concat(rms_norm(x0), rms_norm(x1)) in one launch: block bx normalizes
// input bx into y + bx*n. Mirrors ggml rms_norm_f32 (vec path, 1024 thr,
// vpt=1) including the block_reduce<SUM> tree so sums match bit-exactly.
__global__ void __launch_bounds__(1024) inkwell_rms2cat(
        const float* __restrict__ x0, const float* __restrict__ x1,
        float* __restrict__ y, int n, float eps) {
    __shared__ float s_sum[32];
    const float* x = blockIdx.x ? x1 : x0;
    float* dst = y + (size_t)blockIdx.x * n;
    const int tid = threadIdx.x;
    const int n4 = n / 4;
    float4 xs = make_float4(0.f, 0.f, 0.f, 0.f);
    float tmp = 0.f;
    if (tid < n4) {
        xs = ((const float4*)x)[tid];
        tmp += xs.x * xs.x + xs.y * xs.y + xs.z * xs.z + xs.w * xs.w;
    }
    for (int m = 16; m; m >>= 1) tmp += __shfl_xor_sync(0xffffffffu, tmp, m);
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) s_sum[wid] = tmp;
    __syncthreads();
    tmp = (lane < (int)(blockDim.x >> 5)) ? s_sum[lane] : 0.f;
    for (int m = 16; m; m >>= 1) tmp += __shfl_xor_sync(0xffffffffu, tmp, m);
    const float scale = rsqrtf(__fdiv_rn(tmp, (float)n) + eps);
    if (tid < n4) {
        ((float4*)dst)[tid] = make_float4(scale * xs.x, scale * xs.y,
                                          scale * xs.z, scale * xs.w);
    }
}
extern "C" void ink_launch_rms2cat(const float* x0, const float* x1,
        float* y, int n, float eps, void* stream) {
    inkwell_rms2cat<<<2, 1024, 0, (cudaStream_t)stream>>>(x0, x1, y, n, eps);
}

// rel-bias gather: y[j + h*nkv] = mm[rel_pos[j] + h*ext] - sinks[h].
// Replaces sub + permute/cont + get_rows + permute/cont.
__global__ void inkwell_relbias_gather(const float* __restrict__ mm,
        const float* __restrict__ sinks, const int* __restrict__ rel_pos,
        float* __restrict__ y, int nkv, int nh, int ext) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nkv * nh) return;
    const int h = i / nkv, j = i - h * nkv;
    y[i] = __fsub_rn(mm[(size_t)rel_pos[j] + (size_t)h * ext], sinks[h]);
}
extern "C" void ink_launch_relbias_gather(const float* mm, const float* sinks,
        const void* rel_pos, float* y, int nkv, int nh, int ext, void* stream) {
    const int t = nkv * nh;
    inkwell_relbias_gather<<<(t + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        mm, sinks, (const int*)rel_pos, y, nkv, nh, ext);
}

// ---- ctrl gf_cat GEMV + ctrl_apply fusion (INKWELL_NO_GFAPPLY=1 disables) ----
// One warp per output element i: gain = dot(row i, gate), forget =
// dot(row fo+i, gate), then the ctrl_apply tail. Replicates mmvf's
// warp-per-row 64-col dot (lane -> one pair, shfl_xor 16..1) so the sums
// are bit-identical to the unfused mmvf + inkwell_ctrl_apply chain.
__device__ __forceinline__ float2 gf_wload(const float* p, int i) {
    return ((const float2*)p)[i];
}
__device__ __forceinline__ float2 gf_wload(const __nv_bfloat16* p, int i) {
    const __nv_bfloat162 v = ((const __nv_bfloat162*)p)[i];
    return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
}
template <typename T>
__global__ void inkwell_gf_apply(const T* __restrict__ x,
        const float* __restrict__ y, const float* __restrict__ cur,
        const float* __restrict__ msg, float* __restrict__ dst, int n, int fo) {
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * 8 + w;
    if (i >= n) return;
    float g = 0.f, f = 0.f;
    const float2 t = ((const float2*)y)[lane];
    const float2 tg = gf_wload(x + (size_t)i * 64, lane);
    const float2 tf = gf_wload(x + (size_t)(fo + i) * 64, lane);
    g += tg.x * t.x; g += tg.y * t.y;
    f += tf.x * t.x; f += tf.y * t.y;
    for (int m = 16; m; m >>= 1) {
        g += __shfl_xor_sync(0xffffffffu, g, m);
        f += __shfl_xor_sync(0xffffffffu, f, m);
    }
    if (lane == 0) {
        const float gn = tanhf(g), fg = tanhf(f);
        const float r = __fsub_rn(cur[i], __fmul_rn(fg, cur[i]));
        dst[i] = __fadd_rn(r, __fmul_rn(gn, msg[i]));
    }
}
// ---- bit-exact controller kernels with no 18.7 KiB gate staging buffer ----
// Pointer tables travel in kernel parameter memory (by value), not in device
// memory. A null gate pointer selects the fixed batched-triad scratch below.
struct InkCtrlRmsPtrs {
    const float* x0[3];
    const float* x1[3];
    const __nv_bfloat16* w[3];
    float* dst[3];
};
struct InkCtrlGfPtrs {
    const __nv_bfloat16* w[3];
    const float* y[3];
    const float* cur[3];
    const float* msg[3];
    float* dst[3];
};
struct InkCtrlTriadIo {
    const float* cur[3];
    const float* msg[3];
    const void* down[3];
    const void* gf[3];
    float* dst;
};
__device__ float inkwell_ctrl_gate_scratch[3 * 64];

template <int R, int B>
__global__ void inkwell_rms2down_nosmem_mr(InkCtrlRmsPtrs ps, int n, float eps) {
    __shared__ float s_sum[32];
    __shared__ float s_buf[R][32];
    const int b = (B == 1) ? 0 : blockIdx.y;
    const float* __restrict__ x0 = ps.x0[b];
    const float* __restrict__ x1 = ps.x1[b];
    const __nv_bfloat16* __restrict__ w = ps.w[b];
    float* __restrict__ dst = ps.dst[b];
    if (!dst) dst = inkwell_ctrl_gate_scratch + b * 64;

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
        for (int r = 0; r < R; ++r) s_buf[r][tid] = 0.0f;
    }
    __syncthreads();

    const int r0 = blockIdx.x * R;
    const size_t stride2 = (size_t)n;
    const __nv_bfloat162* wbase = (const __nv_bfloat162*)w + (size_t)r0 * stride2;
    const float2* x0_2 = (const float2*)x0;
    const float2* x1_2 = (const float2*)x1;
    const int half2 = n >> 1;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    for (int c = tid; c < n; c += 256) {
        const bool in0 = (c < half2);
        const float2 raw = in0 ? __ldg(&x0_2[c]) : __ldg(&x1_2[c - half2]);
        const float sc = in0 ? scales[0] : scales[1];
        const float2 tg = make_float2(__fmul_rn(sc, raw.x), __fmul_rn(sc, raw.y));
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const __nv_bfloat162 tw = wbase[(size_t)r * stride2 + c];
            acc[r] += __bfloat162float(tw.x) * tg.x;
            acc[r] += __bfloat162float(tw.y) * tg.y;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], m);
        if (lane == 0) s_buf[r][wid] = acc[r];
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

template <int B>
__global__ void inkwell_gf_apply_batch(InkCtrlGfPtrs ps, int n, int fo) {
    const int b = (B == 1) ? 0 : blockIdx.y;
    const __nv_bfloat16* __restrict__ x = ps.w[b];
    const float* __restrict__ y = ps.y[b];
    if (!y) y = inkwell_ctrl_gate_scratch + b * 64;
    const float* __restrict__ cur = ps.cur[b];
    const float* __restrict__ msg = ps.msg[b];
    float* __restrict__ dst = ps.dst[b];

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

extern "C" void ink_launch_ctrl_triad(const InkCtrlTriadIo* io, int n, float eps,
        int fo, void* stream) {
    InkCtrlRmsPtrs rp{};
    InkCtrlGfPtrs gp{};
    for (int b = 0; b < 3; ++b) {
        rp.x0[b] = io->cur[b];
        rp.x1[b] = io->msg[b];
        rp.w[b] = (const __nv_bfloat16*)io->down[b];
        rp.dst[b] = nullptr;
        gp.w[b] = (const __nv_bfloat16*)io->gf[b];
        gp.y[b] = nullptr;
        gp.cur[b] = io->cur[b];
        gp.msg[b] = io->msg[b];
        gp.dst[b] = io->dst + (size_t)b * (size_t)n;
    }
    const cudaStream_t st = (cudaStream_t)stream;
    inkwell_rms2down_nosmem_mr<4, 3><<<dim3(16, 3), 256, 0, st>>>(rp, n, eps);
    inkwell_gf_apply_batch<3><<<dim3((unsigned)(n + 7) / 8, 3), 256, 0, st>>>(gp, n, fo);
}

extern "C" void ink_launch_gf_apply(const void* x, int x_is_bf16,
        const float* y, const float* cur, const float* msg,
        float* dst, int n, int fo, void* stream) {
    const unsigned g = (unsigned)(n + 7) / 8;
    if (x_is_bf16)
        inkwell_gf_apply<__nv_bfloat16><<<g, 256, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)x, y, cur, msg, dst, n, fo);
    else
        inkwell_gf_apply<float><<<g, 256, 0, (cudaStream_t)stream>>>(
            (const float*)x, y, cur, msg, dst, n, fo);
}

// ---- rms2cat + ctrl_down GEMV fusion (INKWELL_NO_RMSDOWN=1 disables) ----
// Per output row: recompute concat(rms_norm(x0), rms_norm(x1)) in smem with
// the exact 1024-thread association of inkwell_rms2cat emulated on 256
// threads (real thread tid owns virtual tids {tid, tid+256, tid+512}; real
// warp r covers virtual warps {r, r+8, r+16}), then run the mmvf-256 dot
// (256 thr, tid-strided pairs, shfl_xor 16..1 + 32-entry cross-warp buffer).
// Bit-identical to the unfused inkwell_rms2cat + mmvf chain.
template <typename T>
__global__ void inkwell_rms2down(const float* __restrict__ x0,
        const float* __restrict__ x1, const T* __restrict__ w,
        float* __restrict__ dst, int n, float eps) {
    __shared__ float sg[4608];     // gate: [rms(x0) ; rms(x1)]
    __shared__ float s_sum[32];
    __shared__ float s_buf[32];
    const int tid = threadIdx.x, lane = tid & 31, w_ = tid >> 5;
    const int n4 = n / 4;
    if (tid >= 24 && tid < 32) s_sum[tid] = 0.f;  // virtual warps 24..31
    for (int bx = 0; bx < 2; ++bx) {
        const float* x = bx ? x1 : x0;
        // virtual-tid partials, same float4 -> thread map as the 1024-thr kernel
        float p[3]; float4 q[3];
        for (int k = 0; k < 3; ++k) {
            const int vt = tid + k * 256;
            q[k] = vt < n4 ? ((const float4*)x)[vt] : make_float4(0,0,0,0);
            p[k] = q[k].x*q[k].x + q[k].y*q[k].y + q[k].z*q[k].z + q[k].w*q[k].w;
        }
        __syncthreads();
        // real warp w covers virtual warps {w, w+8, w+16}
        for (int k = 0; k < 3; ++k) {
            float v = (tid + k * 256 < n4) ? p[k] : 0.f;
            for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
            if (lane == 0) s_sum[w_ + 8 * k] = v;
        }
        __syncthreads();
        float t = s_sum[lane];
        for (int m = 16; m; m >>= 1) t += __shfl_xor_sync(0xffffffffu, t, m);
        const float scale = rsqrtf(__fdiv_rn(t, (float)n) + eps);
        for (int k = 0; k < 3; ++k) {
            const int vt = tid + k * 256;
            if (vt < n4) ((float4*)(sg + bx * n))[vt] =
                make_float4(scale * q[k].x, scale * q[k].y, scale * q[k].z, scale * q[k].w);
        }
        __syncthreads();
    }
    if (tid >= 8 && tid < 32) s_buf[tid] = 0.f;  // mmvf buf_iw is zero-filled
    __syncthreads();
    // mmvf-identical dot: this block's row, 2304 pairs, tid-strided
    const T* row = w + (size_t)blockIdx.x * (2 * n);
    const float2* gy = (const float2*)sg;
    float s = 0.f;
    for (int c = tid; c < n; c += 256) {
        const float2 tw = gf_wload(row, c);
        const float2 tg = gy[c];
        s += tw.x * tg.x;
        s += tw.y * tg.y;
    }
    for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    s_buf[w_] = s;
    __syncthreads();
    if (w_ == 0) {
        s = s_buf[lane];
        for (int m = 16; m; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (lane == 0) dst[blockIdx.x] = s;
    }
}
extern "C" void ink_launch_rms2down(const float* x0, const float* x1,
        const void* w, int w_is_bf16, float* dst, int n, int nrows,
        float eps, void* stream) {
    static int no_ctrlbatch = -1;
    if (no_ctrlbatch < 0) no_ctrlbatch = (getenv("INKWELL_CTRLBATCH") && !getenv("INKWELL_NO_CTRLBATCH")) ? 0 : 1;
    if (w_is_bf16 && !no_ctrlbatch && n == 2304 && nrows % 4 == 0) {
        InkCtrlRmsPtrs ps{};
        ps.x0[0] = x0;
        ps.x1[0] = x1;
        ps.w[0] = (const __nv_bfloat16*)w;
        ps.dst[0] = dst;
        inkwell_rms2down_nosmem_mr<4, 1><<<(unsigned)(nrows / 4), 256, 0, (cudaStream_t)stream>>>(ps, n, eps);
    } else if (w_is_bf16)
        inkwell_rms2down<__nv_bfloat16><<<(unsigned)nrows, 256, 0, (cudaStream_t)stream>>>(
            x0, x1, (const __nv_bfloat16*)w, dst, n, eps);
    else
        inkwell_rms2down<float><<<(unsigned)nrows, 256, 0, (cudaStream_t)stream>>>(
            x0, x1, (const float*)w, dst, n, eps);
}

// ---- device argmax: Rust total_cmp order + last-max tiebreak ----
// key(f) is a monotonic u32 map of the f32 bit pattern (IEEE totalOrder,
// identical to f32::total_cmp); on equal keys the larger index wins, which
// matches Iterator::max_by returning the last equal maximum.
__device__ __forceinline__ unsigned am_key(float f) {
    const unsigned b = __float_as_uint(f);
    return b ^ (b & 0x80000000u ? 0xFFFFFFFFu : 0x80000000u);
}
__device__ __forceinline__ void am_better(unsigned k, int i, unsigned& bk, int& bi) {
    if (k > bk || (k == bk && i > bi)) { bk = k; bi = i; }
}
// out: [0]=idx, [1]=key (final, written by last block); [8..8+2*grid] partials.
// ctr: self-resetting ticket (must start 0).
__global__ void __launch_bounds__(256) inkwell_argmax(
        const float* __restrict__ x, int n, unsigned* __restrict__ out,
        unsigned* __restrict__ ctr) {
    unsigned bk = 0; int bi = -1;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        am_better(am_key(__ldg(x + i)), i, bk, bi);
    }
    __shared__ unsigned sk[256];
    __shared__ int si[256];
    __shared__ bool last;
    for (int r = 0; r < 2; ++r) {
        sk[threadIdx.x] = bk; si[threadIdx.x] = bi;
        __syncthreads();
        for (int s = blockDim.x / 2; s; s >>= 1) {
            if (threadIdx.x < s) {
                const unsigned k2 = sk[threadIdx.x + s];
                const int i2 = si[threadIdx.x + s];
                if (k2 > sk[threadIdx.x] ||
                    (k2 == sk[threadIdx.x] && i2 > si[threadIdx.x])) {
                    sk[threadIdx.x] = k2; si[threadIdx.x] = i2;
                }
            }
            __syncthreads();
        }
        if (r == 0) {
            if (threadIdx.x == 0) {
                out[8 + 2 * blockIdx.x] = sk[0];
                out[9 + 2 * blockIdx.x] = (unsigned)si[0];
                __threadfence();
                last = atomicAdd(ctr, 1) == gridDim.x - 1;
                if (last) *ctr = 0;
            }
            __syncthreads();
            if (!last) return;
            bk = 0; bi = -1;
            for (int b = threadIdx.x; b < (int)gridDim.x; b += blockDim.x) {
                am_better(out[8 + 2 * b], (int)out[9 + 2 * b], bk, bi);
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) { out[0] = (unsigned)si[0]; out[1] = sk[0]; }
}
extern "C" void ink_launch_argmax(const float* x, int n, unsigned* out,
        unsigned* ctr, void* stream) {
    const int nb = (n + 255) / 256 > 128 ? 128 : (n + 255) / 256;
    inkwell_argmax<<<nb, 256, 0, (cudaStream_t)stream>>>(x, n, out, ctr);
}
