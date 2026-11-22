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
