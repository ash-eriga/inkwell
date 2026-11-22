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
