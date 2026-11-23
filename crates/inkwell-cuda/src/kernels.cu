// Elementwise + small-vector kernels for inkwell.
// Reference implementations — correctness first (golden-test path),
// fusion comes later per the autotuning ledger.

#include <cstdio>
#include <cuda_fp16.h>

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
extern "C" void ink_launch_attn_decode(const void* K, const void* V, const void* Q,
                                       const void* bias, const void* mask, void* out,
                                       long long nh, long long hkv, long long n_kv, long long hd,
                                       long long k_s1, long long k_s2, long long v_s1, long long v_s2,
                                       long long q_s2, long long b_s2, float scale, int kv16, void* stream) {
    // k/v strides arrive in BYTES (dtype-agnostic); q_s2/b_s2 stay f32 elements
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
    for (int tt = 0; tt < NTOK; ++tt)
        inkwell_exl3_had_in<<<(unsigned)(k / 128), 128, 512, (cudaStream_t)stream>>>(
            x + (size_t)tt * k, (const uint16_t*)suh, scratch + (size_t)tt * k, (int)k);
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

extern "C" void ink_launch_exl3_gemv(const void* trel, const void* suh, const void* svh,
                                     const float* x, float* y,
                                     long long k, long long n, long long ka, long long halfk,
                                     float* scratch, void* stream) {
    const int tk = (int)(k / 16), tn = (int)(n / 16);
    inkwell_exl3_had_in<<<(unsigned)(k / 128), 128, 512, (cudaStream_t)stream>>>(
        x, (const uint16_t*)suh, scratch /* xh */, (int)k);
    const int nb = (tn + 3) / 4;
    // aim for ~256 blocks: split K when tn is small
    int nky = 1;
    { static int tgt = -1; if (tgt < 0) { const char* e = getenv("INKWELL_GEMV_NKY_TGT"); tgt = e ? atoi(e) : 512; }
      while (nb * nky < tgt && nky < 8 && nky * 16 <= tk) nky *= 2; }
    if (nky > 1) cudaMemsetAsync(scratch + k, 0, nky * n * 4, (cudaStream_t)stream);
    dim3 gsz((unsigned)nb, (unsigned)nky, 1);
    inkwell_exl3_gemv_dp4a<<<gsz, 256, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)trel, scratch, scratch + k, tk, tn, (int)ka, (int)halfk);
    inkwell_exl3_had_out<<<(unsigned)(n / 128), 128, 512, (cudaStream_t)stream>>>(
        scratch + k, (const uint16_t*)svh, y, (int)n, nky, (int)n);
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
        if (live && u < w) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn0; ++b)
                rg[b * 32 + u] = tile_base[((size_t)((tk_lo + b) * tn + tn_i)) * (size_t)w + u];
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

__global__ void inkwell_exl3_had_out_moe(const uint64_t* __restrict__ descs,
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
        int* __restrict__ pf_hit) {
    // LFU eviction: cnt[e] counts uses; the least-used resident is evicted.
    // pf_hit[0]=bank hits, pf_hit[1]=real misses (filled via serial ccopy)
    if (threadIdx.x != 0) return;
    const int pf = pready ? *pready : 0;
    int claimed = 0;
    // pass 1: hits (and prefetch-bank hits) — never evict slots needed now
    for (int s = 0; s < nslots; ++s) {
        const int e = ids[s];
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
            if (v < k) claimed |= 1 << v;
        } else {
            miss[s] = 1;
            map[s] = -1;
        }
    }
    // pass 2: evict the least-used unclaimed resident for each miss
    for (int s = 0; s < nslots; ++s) {
        if (!miss[s]) continue;
        const int e = ids[s];
        int best = -1, bestc = 0x7fffffff;
        for (int j = 0; j < k; ++j) {
            if (claimed & (1 << j)) continue;
            const int c = cnt[tag[j]];
            if (c < bestc) { bestc = c; best = j; }
        }
        if (best < 0) {
            // no free slot (nslots > k): stream this expert's host descriptors
            map[s] = EXL3_DIRECT;
            continue;
        }
        const int v = best;
        claimed |= 1 << v;
        tag[v] = e;
        map[s] = v;
        if (pf_hit) atomicAdd(pf_hit + 1, 1);
    }
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
                                       int mode) {
    const int slot = blockIdx.x / 3;
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
    if (mslot >= EXL3_PF_BASE) { cbase = bank; mslot -= EXL3_PF_BASE; }
    uint8_t* dst = cbase + (size_t)(mslot * 3 + proj) * (size_t)slot_proj;
    if (threadIdx.x == 0 && part == 0) {
        uint64_t* o = sd + ids[slot] * 4;
        o[0] = (uint64_t)(uintptr_t)dst;
        o[1] = (uint64_t)(uintptr_t)(dst + trel_b);
        o[2] = (uint64_t)(uintptr_t)(dst + trel_b + suh_b);
        o[3] = d[3];
    }
    if (!miss[slot]) return;
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
extern "C" void ink_launch_exl3_moe(const void* descs_g, const void* descs_u, const void* descs_d,
                                    const int* ids, const float* x,
                                    float* xh, float* gu, float* hid, float* dn,
                                    const float* w, float* out, int nslots, void* stream,
                                    void* staging, void* sdescs_g, void* sdescs_u, void* sdescs_d,
                                    int on_dev) {
    const cudaStream_t st = (cudaStream_t)stream;
    if (!on_dev) {
        inkwell_exl3_moe_stage<<<nslots * 3, 256, 0, st>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            ids, (uint8_t*)staging, (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d);
    }
    const uint64_t* G = on_dev ? (const uint64_t*)descs_g : (const uint64_t*)sdescs_g;
    const uint64_t* U = on_dev ? (const uint64_t*)descs_u : (const uint64_t*)sdescs_u;
    const uint64_t* D = on_dev ? (const uint64_t*)descs_d : (const uint64_t*)sdescs_d;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    // gate + up fused: blockIdx.z selects the descriptor set
    dim3 gin((KG + 511) / 512, nslots, 2);
    inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG, nullptr, 0, nullptr, nullptr);
    int nky = 1;
    { const char* e = getenv("INKWELL_MOE_NKY"); if (e) nky = atoi(e); if (nky < 1) nky = 1; }
    if (nky > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    dim3 gg(NG / 64, nslots, 2 * nky);
    inkwell_exl3_gemv_moe<<<gg, 256, 0, st>>>(G, U, ids, xh, (const float*)gu, 2 * NG,
                                             (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky, nullptr, 0);
    dim3 gout((NG + 511) / 512, nslots, 2);
    inkwell_exl3_had_out_moe<<<gout, 128, 0, st>>>(G, U, ids, dn, gu, NG, 2 * NG, 0, ND, NG);
    inkwell_situ_pair<<<(unsigned)((nslots * NG + 255) / 256), 256, 0, st>>>(gu, nullptr, hid, 4.f, 25.f, NG, nslots);
    dim3 din((KD + 511) / 512, nslots);
    inkwell_exl3_had_in_moe<<<din, 128, 0, st>>>(D, nullptr, ids, hid, xh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
    if (nky > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    dim3 gd(ND / 64, nslots, nky);
    inkwell_exl3_gemv_moe<<<gd, 256, 0, st>>>(D, nullptr, ids, xh, nullptr, 0,
                                             dn, ND, 0, KD / 16, ND / 16, KD, ND, nky, nullptr, 0);
    dim3 dout((ND + 511) / 512, nslots, 1);
    inkwell_exl3_had_out_moe<<<dout, 128, 0, st>>>(D, nullptr, ids, dn, xh /*reuse xh as y*/, ND, ND, 0, ND, 0);
    inkwell_expert_wsum<<<(unsigned)((ND + 255) / 256), 256, 0, st>>>(xh, w, nullptr, out, ND, nslots);
}

// Cached variant: resolve -> copy misses -> same downstream chain.
extern "C" void ink_launch_exl3_moe_cached(const void* descs_g, const void* descs_u, const void* descs_d,
                                    const int* ids, const float* x,
                                    float* xh, float* gu, float* hid, float* dn,
                                    const float* w, float* out, int nslots, void* stream,
                                    void* cache, void* tag, void* map, void* miss, void* cnt, int k,
                                    int64_t slot_proj,
                                    void* sdescs_g, void* sdescs_u, void* sdescs_d,
                                    const void* ptag, const void* pready, void* bank, int* pf_hit) {
    const cudaStream_t st = (cudaStream_t)stream;
    inkwell_exl3_cache_resolve<<<1, 32, 0, st>>>(ids,
        (int*)tag, (int*)map, (int*)miss, (int*)cnt, nslots, k,
        (const int*)ptag, (const int*)pready, pf_hit);
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
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, 1);
        cudaEventRecord(ev_fork, st);
        cudaStreamWaitEvent(ss, ev_fork, 0);
        inkwell_exl3_moe_ccopy<<<dim3(nslots * 3, 16), 128, 0, ss>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            ids, (uint8_t*)cache, (const int*)map, (const int*)miss,
            (uint8_t*)bank, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, 2);
        cudaEventRecord(ev_join, ss);
    }
    const uint64_t* G = (const uint64_t*)sdescs_g;
    const uint64_t* U = (const uint64_t*)sdescs_u;
    const uint64_t* D = (const uint64_t*)sdescs_d;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    int nky = 1;
    { const char* e = getenv("INKWELL_MOE_NKY"); if (e) nky = atoi(e); if (nky < 1) nky = 1; }
    // gate+up fused in one launch: blockIdx.z selects the descriptor set.
    // Hit slots run BEFORE ev_join so their gemv overlaps the bulk copy of
    // the missed experts on the side stream; miss slots run right after.
    dim3 gin((KG + 511) / 512, nslots, 2);
    inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG,
                                                (const int*)miss, 1, nullptr, nullptr);
    if (nky > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    dim3 gg(NG / 64, nslots, 2 * nky);
    inkwell_exl3_gemv_moe<<<gg, 256, 0, st>>>(G, U, ids, xh, (const float*)gu, 2 * NG,
                                             (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky,
                                             (const int*)miss, 1);
    cudaStreamWaitEvent(st, ev_join, 0);
    inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, ids, x, xh, (float*)gu, 2 * NG, KG,
                                                (const int*)miss, 2, nullptr, nullptr);
    inkwell_exl3_gemv_moe<<<gg, 256, 0, st>>>(G, U, ids, xh, (const float*)gu, 2 * NG,
                                             (float*)dn, ND, NG, KG / 16, NG / 16, KG, ND, nky,
                                             (const int*)miss, 2);
    dim3 gout((NG + 511) / 512, nslots, 2);
    inkwell_exl3_had_out_moe<<<gout, 128, 0, st>>>(G, U, ids, dn, gu, NG, 2 * NG, 0, ND, NG);
    inkwell_situ_pair<<<(unsigned)((nslots * NG + 255) / 256), 256, 0, st>>>(gu, nullptr, hid, 4.f, 25.f, NG, nslots);
    dim3 din((KD + 511) / 512, nslots);
    inkwell_exl3_had_in_moe<<<din, 128, 0, st>>>(D, nullptr, ids, hid, xh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
    if (nky > 1) cudaMemsetAsync(dn, 0, nslots * ND * 4, st);
    dim3 gd(ND / 64, nslots, nky);
    inkwell_exl3_gemv_moe<<<gd, 256, 0, st>>>(D, nullptr, ids, xh, nullptr, 0,
                                             dn, ND, 0, KD / 16, ND / 16, KD, ND, nky, nullptr, 0);
    dim3 dout((ND + 511) / 512, nslots, 1);
    inkwell_exl3_had_out_moe<<<dout, 128, 0, st>>>(D, nullptr, ids, dn, xh /*reuse xh as y*/, ND, ND, 0, ND, 0);
    inkwell_expert_wsum<<<(unsigned)((ND + 255) / 256), 256, 0, st>>>(xh, w, nullptr, out, ND, nslots);
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
    float* __restrict__ out)            // [ch]
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
        float* out, void* stream) {
    const int thr = ch < 256 ? ch : 256;
    inkwell_conv_step<<<(ch + thr - 1) / thr, thr, 0, (cudaStream_t)stream>>>(
        x_in, state, region_off, w, dyn, ch, n_heads, scale, out);
}

// ================= union-batched (verify ntok) MoE =================
// Expand 8 tokens × top-8 (64 selections) into (expert,consumer) pairs sorted
// by expert id. ~20 unique experts/visit typical → the gemv reads each expert
// once for all its consumer tokens (LFM-style weight sharing for MoE).
__global__ void inkwell_exl3_union_expand(
    const int* __restrict__ ids8,          // [64] token-major: ids8[t*8+j]
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
    for (int s = 0; s < 64; ++s) cnt += (ids8[s] == e);
    sh_cnt[e] = cnt;
    __syncthreads();
    if (e == 0) {
        int off = 0, nu = 0;
        for (int i = 0; i < 128; ++i) {
            sh_start[i] = off;
            if (sh_cnt[i]) { uids[nu] = i; sh_uidx[i] = nu; estart[nu] = off; ++nu; }
            off += sh_cnt[i];
        }
        estart[nu] = 64;
        *pnexp = nu;
    }
    __syncthreads();
    if (cnt) {
        int p = sh_start[e];
        for (int s = 0; s < 64; ++s) {
            if (ids8[s] == e) {
                eids[p] = e; euni[p] = sh_uidx[e]; etok[p] = s >> 3; epos[s] = p;
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
                                      const int* __restrict__ missf, int skipmode) {
    const int slot = blockIdx.y;
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
    __shared__ float xhb[8][8][16];   // [consumer][batch row][16]
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
            xhb[cc][r16 >> 4][r16 & 15] =
                xhb_g[(size_t)cc * xstride + (tk_lo + (r16 >> 4)) * 16 + (r16 & 15)];
        }
        if (live && u < w) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn0; ++b)
                rg[b * 32 + u] = tile_base[((size_t)((tk_lo + b) * tn + tn_i)) * (size_t)w + u];
        }
    }
    __syncthreads();
    for (int tk0 = tk_lo; tk0 < tk_hi; tk0 += 8) {
        const int tkn = min(8, tk_hi - tk0);
        const int nk0 = tk0 + 8;
        const int tkn2 = min(8, tk_hi - nk0);
        uint32_t pfb[8];
        int pfidx[4]; float pfx[4]; int npf = 0;
        if (live && nk0 < tk_hi && u < w) {
            for (int b = 0; b < tkn2; ++b)
                pfb[b] = tile_base[((size_t)((nk0 + b) * tn + tn_i)) * (size_t)w + u];
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
            for (int b = 0; b < tkn; ++b) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint32_t wi = widx[q];
                    const uint64_t v = ((uint64_t)rg[b * 32 + (wi & 63)] << 32)
                                     | (uint64_t)rg[b * 32 + ((wi >> 6) & 63)];
                    const uint32_t w16 = (uint32_t)(v >> (wi >> 12)) & 0xffffu;
                    const int bs = (int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
#pragma unroll
                    for (int c2 = 0; c2 < 8; ++c2) {
                        if (c2 < nc) {
                            const float xr = xhb[c2][b][rp + 4 * q];
                            s[c2] += xr * (float)bs;
                            sx[c2] += xr;
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (live && nk0 < tk_hi && u < w) {
            uint32_t* rg = rings + g * 256;
            for (int b = 0; b < tkn2; ++b) rg[b * 32 + u] = pfb[b];
        }
        if (nk0 < tk_hi) {
            for (int i = 0; i < npf; ++i) {
                const int idx = pfidx[i];
                const int cc = idx / (tkn2 * 16), r16 = idx % (tkn2 * 16);
                xhb[cc][r16 >> 4][r16 & 15] = pfx[i];
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
    int on_dev, void* stream) {
    const cudaStream_t st = (cudaStream_t)stream;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    const bool cached = cache != nullptr;
    if (cached) {
        inkwell_exl3_union_expand<<<1, 128, 0, st>>>(ids8, uids, unexp,
            eids, euni, etok, estart, epos, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr);
    } else {
        // uncached/on_dev: write per-expert-id desc tables directly (no staging)
        inkwell_exl3_union_expand<<<1, 128, 0, st>>>(ids8, uids, unexp,
            eids, euni, etok, estart, epos,
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d);
    }
    int nexp = 64;
    cudaMemcpyAsync(&nexp, unexp, 4, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
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

    const uint64_t* G = (const uint64_t*)sdescs_g;
    const uint64_t* U = (const uint64_t*)sdescs_u;
    const uint64_t* D = (const uint64_t*)sdescs_d;
    if (cached) {
        inkwell_exl3_cache_resolve<<<1, 32, 0, st>>>(uids,
            tag, map, miss, cnt, nexp, k, nullptr, nullptr, nullptr);
        inkwell_exl3_moe_ccopy<<<dim3(nexp * 3, 1), 32, 0, st>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            uids, (uint8_t*)cache, map, miss,
            nullptr, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, 1);
        cudaEventRecord(ev_fork, st);
        cudaStreamWaitEvent(ss, ev_fork, 0);
        inkwell_exl3_moe_ccopy<<<dim3(nexp * 3, 16), 128, 0, ss>>>(
            (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
            uids, (uint8_t*)cache, map, miss,
            nullptr, slot_proj,
            (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d, 2);
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
        inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, eids, x8, xh_e, gu_e,
            2 * NG, KG, miss, 1, euni, etok);
        if (dbg) { cudaStreamSynchronize(st); fprintf(stderr, "  hadin1 err=%d\n", (int)cudaGetLastError()); }
        inkwell_exl3_gemv_moe_ntok<<<gg, 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, KG, ND, nky, miss, 1);
        if (dbg) { cudaStreamSynchronize(st); fprintf(stderr, "  gemv1 err=%d\n", (int)cudaGetLastError()); }
        cudaStreamWaitEvent(st, ev_join, 0);
        inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, eids, x8, xh_e, gu_e,
            2 * NG, KG, miss, 2, euni, etok);
        if (dbg) { cudaStreamSynchronize(st); fprintf(stderr, "  hadin2 err=%d\n", (int)cudaGetLastError()); }
        inkwell_exl3_gemv_moe_ntok<<<gg, 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, KG, ND, nky, miss, 2);
    } else {
        inkwell_exl3_had_in_moe<<<gin, 128, 0, st>>>(G, U, eids, x8, xh_e, gu_e,
            2 * NG, KG, nullptr, 0, euni, etok);
        inkwell_exl3_gemv_moe_ntok<<<gg, 256, 0, st>>>(G, U, uids, estart, xh_e,
            (const float*)gu_e, 2 * NG, dn_e, ND, NG, KG / 16, NG / 16, KG, ND, nky, nullptr, 0);
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
    inkwell_exl3_gemv_moe_ntok<<<gd, 256, 0, st>>>(D, nullptr, uids, estart, xh_e,
        nullptr, 0, dn_e, ND, 0, KD / 16, ND / 16, KD, ND, nky, nullptr, 0);
    if (dbg) { cudaStreamSynchronize(st); cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok dgemv done err=%d\n", (int)e); }
    if (dbg == 5) return;
    inkwell_exl3_had_out_moe<<<dout, 128, 0, st>>>(D, nullptr, eids, dn_e, xh_e, ND, ND, 0, ND, 0);
    inkwell_exl3_expert_wsum8<<<dim3((ND + 255) / 256, 8), 256, 0, st>>>(xh_e, w8, epos, out8, ND);
    if (dbg) { cudaStreamSynchronize(st); cudaError_t e = cudaGetLastError(); fprintf(stderr, "moe_ntok all done err=%d\n", (int)e); }
}
