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
