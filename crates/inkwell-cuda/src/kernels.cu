// Elementwise + small-vector kernels for inkwell.
// Reference implementations — correctness first (golden-test path),
// fusion comes later per the autotuning ledger.

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
__global__ void inkwell_attn_decode(const float* __restrict__ K, const float* __restrict__ V,
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
    const float* Kh = K + hk * k_s2;
    for (int j = w; j < n_kv; j += nw) {
        const float* kj = Kh + j * k_s1;
        float a = 0.f;
        for (int d = lane; d < hd; d += 32) a += qs[d] * kj[d];
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
    const float* Vh = V + hk * v_s2;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int j = 0; j < n_kv; ++j) o += scores[j] * Vh[j * v_s1 + d];
        out[h * hd + d] = o * inv;
    }
}
extern "C" void ink_launch_attn_decode(const void* K, const void* V, const void* Q,
                                       const void* bias, const void* mask, void* out,
                                       long long nh, long long hkv, long long n_kv, long long hd,
                                       long long k_s1, long long k_s2, long long v_s1, long long v_s2,
                                       long long q_s2, long long b_s2, float scale, void* stream) {
    inkwell_attn_decode<<<(unsigned)nh, 256, (size_t)(n_kv + hd) * 4, (cudaStream_t)stream>>>(
        (const float*)K, (const float*)V, (const float*)Q, (const float*)bias,
        (const float*)mask, (float*)out, (int)nh, (int)hkv, (int)n_kv, (int)hd,
        (int)k_s1, (int)k_s2, (int)v_s1, (int)v_s2, (int)q_s2, (int)b_s2, scale);
}

#include <cuda_fp16.h>
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
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[(i0 + 1) % nw];
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
                                     float* __restrict__ y, int n) {
    extern __shared__ float sm[];
    const int blk = blockIdx.x;
    const int th = threadIdx.x;
    const int base = blk * 128;
    sm[th] = (base + th < n) ? t[base + th] : 0.f;
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
    const int g = threadIdx.x >> 4;      // 4 tiles per block
    const int c = threadIdx.x & 15;      // column within tile
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk); // ≤32 u32 per tile
    __shared__ float xhs[16];
    __shared__ uint32_t rings[4 * 32];
    const bool live = tn_i < tn;
    float s = 0.f;
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        if (threadIdx.x < 16) xhs[c] = xh[tk_i * 16 + c];
        if (live) {
            const uint32_t* ring = trel + (size_t)(tk_i * tn + tn_i) * (size_t)w;
            uint32_t* rg = rings + g * 32;
            for (int j = c; j < w; j += 16) rg[j] = ring[j];
        }
        __syncthreads();
        if (live) {
            const uint32_t* rg = rings + g * 32;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                // invert the tile_perm map: i = f(r, c)
                const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                            | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
                const uint32_t win = exl3_window(rg, i, ka, halfk);
                s += xhs[r] * exl3_mul1(win); // fp16 hfma per element — spec-exact
            }
        }
        __syncthreads();
    }
    if (live) t[tn_i * 16 + c] = s;
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
    inkwell_exl3_gemv_dp4a<<<(tn + 3) / 4, 64, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)trel, scratch, scratch + k, tk, tn, (int)ka, (int)halfk);
    inkwell_exl3_had_out<<<(unsigned)(n / 128), 128, 512, (cudaStream_t)stream>>>(
        scratch + k, (const uint16_t*)svh, y, (int)n);
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

__global__ void inkwell_exl3_had_in_moe(const uint64_t* __restrict__ descs,
                                        const int* __restrict__ ids,
                                        const float* __restrict__ x,
                                        float* __restrict__ xh, int k) {
    extern __shared__ float sm[];
    const int slot = blockIdx.y;
    const uint16_t* suh = (const uint16_t*)(uintptr_t)descs[ids[slot] * 4 + 1];
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
    if (base + t < k) xh[slot * k + base + t] = sm[t] * 0.08838834764831845f;
}

// grid (tn, nslots): block decodes tile tn_i of expert ids[slot]; acc -> t[slot][n]
__global__ void inkwell_exl3_gemv_moe(const uint64_t* __restrict__ descs,
                                      const int* __restrict__ ids,
                                      const float* __restrict__ xh,
                                      float* __restrict__ t,
                                      int tk, int tn, int k, int n) {
    const int slot = blockIdx.y;
    const uint64_t* d = descs + ids[slot] * 4;
    const uint32_t* tile_base = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int g = threadIdx.x >> 4;      // 4 tiles per block
    const int c = threadIdx.x & 15;      // column within tile
    const int tn_i = blockIdx.x * 4 + g;
    const int w = r_bits_words(ka, halfk); // ≤32 u32 per tile
    __shared__ float xhs[16];            // same xh slice for all 4 tiles
    __shared__ uint32_t rings[4 * 32];
    float s = 0.f;
    const bool live = tn_i < tn;
    for (int tk_i = 0; tk_i < tk; ++tk_i) {
        if (c < 16 && g == 0) {} // all threads share xhs; load once
        if (threadIdx.x < 16) xhs[c] = xh[slot * k + tk_i * 16 + c];
        if (live) {
            const uint32_t* ring = tile_base + (size_t)(tk_i * tn + tn_i) * (size_t)w;
            uint32_t* rg = rings + g * 32;
            for (int j = c; j < w; j += 16) rg[j] = ring[j];
        }
        __syncthreads();
        if (live) {
            const uint32_t* rg = rings + g * 32;
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3)
                            | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
                s += xhs[r] * exl3_mul1(exl3_window(rg, i, ka, halfk));
            }
        }
        __syncthreads();
    }
    if (live) t[slot * n + tn_i * 16 + c] = s;
}

__global__ void inkwell_exl3_had_out_moe(const uint64_t* __restrict__ descs,
                                         const int* __restrict__ ids,
                                         const float* __restrict__ t,
                                         float* __restrict__ y, int n, int stride, int off) {
    extern __shared__ float sm[];
    const int slot = blockIdx.y;
    const uint16_t* svh = (const uint16_t*)(uintptr_t)descs[ids[slot] * 4 + 2];
    const int blk = blockIdx.x;
    const int th = threadIdx.x;
    const int base = blk * 128;
    sm[th] = (base + th < n) ? t[slot * n + base + th] : 0.f;
    __syncthreads();
    for (int s = 1; s < 128; s <<= 1) {
        const float u = (th & s) ? sm[th - s] - sm[th] : sm[th] + sm[th + s];
        __syncthreads();
        sm[th] = u;
        __syncthreads();
    }
    if (base + th < n) y[slot * stride + off + base + th] = sm[th] * 0.08838834764831845f
        * __half2float(__ushort_as_half(svh[base + th]));
}

// Stage selected experts pinned-host → VRAM staging (coalesced uint4 copies,
// latency hidden by occupancy). Rewrites staged desc tables (indexed by
// ids[slot]) to point at staging; downstream kernels unchanged.
// Per (slot,proj) layout: [trellis 663552][suh 4608][svh 4608] = 672768 B.
#define EXL3_STAGE_PROJ 672768u
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
    uint8_t* dst = staging + (size_t)(slot * 3 + proj) * EXL3_STAGE_PROJ;
    const int suh_b = (proj == 2 ? 2304 : 4608);  // down in=1152→2304B; gate/up in=2304→4608B
    const int svh_b = (proj == 2 ? 4608 : 2304);
    const uint4* s0 = (const uint4*)(uintptr_t)d[0];
    const uint4* s1 = (const uint4*)(uintptr_t)d[1];
    const uint4* s2 = (const uint4*)(uintptr_t)d[2];
    const int n0 = 663552 / 16, n1 = suh_b / 16, n2 = svh_b / 16;
    uint4* t0 = (uint4*)dst;
    uint4* t1 = (uint4*)(dst + 663552);
    uint4* t2 = (uint4*)(dst + 663552 + suh_b);
    const int t = threadIdx.x;
    for (int i = t; i < n0; i += blockDim.x) t0[i] = s0[i];
    for (int i = t; i < n1; i += blockDim.x) t1[i] = s1[i];
    for (int i = t; i < n2; i += blockDim.x) t2[i] = s2[i];
    if (t == 0) {
        uint64_t* o = sd + ids[slot] * 4;
        o[0] = (uint64_t)(uintptr_t)dst;
        o[1] = (uint64_t)(uintptr_t)(dst + 663552);
        o[2] = (uint64_t)(uintptr_t)(dst + 663552 + suh_b);
        o[3] = d[3];
    }
}

// Full per-slot expert chain: stage -> gate -> situ -> down -> wsum.
extern "C" void ink_launch_exl3_moe(const void* descs_g, const void* descs_u, const void* descs_d,
                                    const int* ids, const float* x,
                                    float* xh, float* gu, float* hid, float* dn,
                                    const float* w, float* out, int nslots, void* stream,
                                    void* staging, void* sdescs_g, void* sdescs_u, void* sdescs_d) {
    const cudaStream_t st = (cudaStream_t)stream;
    inkwell_exl3_moe_stage<<<nslots * 3, 256, 0, st>>>(
        (const uint64_t*)descs_g, (const uint64_t*)descs_u, (const uint64_t*)descs_d,
        ids, (uint8_t*)staging, (uint64_t*)sdescs_g, (uint64_t*)sdescs_u, (uint64_t*)sdescs_d);
    const uint64_t* G = (const uint64_t*)sdescs_g;
    const uint64_t* U = (const uint64_t*)sdescs_u;
    const uint64_t* D = (const uint64_t*)sdescs_d;
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    // gate + up into gu[slot][0..1152] and gu[slot][1152..2304]
    for (int p = 0; p < 2; ++p) {
        const uint64_t* dd = p ? U : G;
        dim3 gin(KG / 128, nslots);
        inkwell_exl3_had_in_moe<<<gin, 128, 512, st>>>(dd, ids, x, xh, KG);
        dim3 gg(NG / 64, nslots);
        inkwell_exl3_gemv_moe<<<gg, 64, 0, st>>>(dd, ids, xh, dn /*reuse dn as t*/, KG / 16, NG / 16, KG, NG);
        dim3 gout(NG / 128, nslots);
        inkwell_exl3_had_out_moe<<<gout, 128, 512, st>>>(dd, ids, dn, gu, NG, 2 * NG, p * NG);
    }
    inkwell_situ_pair<<<(unsigned)((nslots * NG + 255) / 256), 256, 0, st>>>(gu, nullptr, hid, 4.f, 25.f, NG, nslots);
    dim3 din(KD / 128, nslots);
    // KD=1152 not multiple of 128*? 1152/128=9 blocks
    inkwell_exl3_had_in_moe<<<din, 128, 512, st>>>(D, ids, hid, xh, KD);
    dim3 gd(ND / 64, nslots);
    inkwell_exl3_gemv_moe<<<gd, 64, 0, st>>>(D, ids, xh, dn, KD / 16, ND / 16, KD, ND);
    dim3 dout(ND / 128, nslots);
    inkwell_exl3_had_out_moe<<<dout, 128, 512, st>>>(D, ids, dn, xh /*reuse xh as y*/, ND, ND, 0);
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
