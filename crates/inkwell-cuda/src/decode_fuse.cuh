#ifndef INKWELL_DECODE_FUSE_CUH
#define INKWELL_DECODE_FUSE_CUH

#define DF_MIX_NPL 72
#define DF_MIX_MAX_ROWS 9

struct InkMixPtrs {
    const float* p[DF_MIX_MAX_ROWS + 1];
};

__global__ void __launch_bounds__(256) inkwell_mix_fused_ptrs(InkMixPtrs rows,
        const float* __restrict__ q, float* __restrict__ out,
        int embd, int nrows, float scale) {
    __shared__ float s_rms[64];
    __shared__ float s_p[64];
    const int qc = blockIdx.y;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    float qq[DF_MIX_NPL];
#pragma unroll
    for (int t = 0; t < DF_MIX_NPL; ++t) qq[t] = q[(size_t)qc * embd + lane + 32 * t];
    for (int j = w; j < nrows; j += nw) {
        const float* col = rows.p[j];
        float c[DF_MIX_NPL];
#pragma unroll
        for (int t = 0; t < DF_MIX_NPL; ++t) c[t] = col[lane + 32 * t];
        float ss = 0.f;
#pragma unroll
        for (int t = 0; t < DF_MIX_NPL; ++t) ss += c[t] * c[t];
        for (int m = 16; m; m >>= 1) ss += __shfl_xor_sync(~0u, ss, m);
        const float rms = sqrtf(ss / (float)embd + 1e-6f);
        float dp = 0.f;
#pragma unroll
        for (int t = 0; t < DF_MIX_NPL; ++t) dp += c[t] / rms * qq[t];
        for (int m = 16; m; m >>= 1) dp += __shfl_xor_sync(~0u, dp, m);
        if (lane == 0) { s_rms[j] = rms; s_p[j] = dp; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < nrows; ++j) { s_p[j] *= scale; mx = fmaxf(mx, s_p[j]); }
        float d = 0.f;
        for (int j = 0; j < nrows; ++j) { s_p[j] = __expf(s_p[j] - mx); d += s_p[j]; }
        for (int j = 0; j < nrows; ++j) s_p[j] /= d;
    }
    __syncthreads();
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= embd) return;
    float a = 0.f;
    for (int j0 = 0; j0 < nrows; j0 += 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = (j0 + u < nrows) ? rows.p[j0 + u][e] : 0.f;
#pragma unroll
        for (int u = 0; u < 8; ++u) if (j0 + u < nrows) a += v[u] / s_rms[j0 + u] * s_p[j0 + u];
    }
    out[(size_t)qc * embd + e] = a;
}

extern "C" void ink_launch_mix_ptrs(const float* const* rowptr, const float* q, float* out,
        int embd, int rows, int qcols, float scale, void* stream) {
    InkMixPtrs p;
    for (int i = 0; i < DF_MIX_MAX_ROWS + 1; ++i) p.p[i] = nullptr;
    for (int i = 0; i < rows && i < DF_MIX_MAX_ROWS; ++i) p.p[i] = rowptr[i];
    inkwell_mix_fused_ptrs<<<dim3((embd + 255) / 256, qcols), 256, 0, (cudaStream_t)stream>>>(
        p, q, out, embd, rows, scale);
}

__global__ void inkwell_conv_step_add(
    const float* __restrict__ x_in,
    float* __restrict__ state,
    int region_off,
    const float* __restrict__ w,
    const float* __restrict__ dyn,
    const float* __restrict__ residual,
    int ch, int n_heads, float scale,
    float* __restrict__ out,
    float* __restrict__ snap)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) return;
    float* st = state + 3 * (size_t)region_off;
    const float x0 = st[3 * c], x1 = st[3 * c + 1], x2 = st[3 * c + 2];
    const float x3 = x_in[c];
    st[3 * c] = x1; st[3 * c + 1] = x2; st[3 * c + 2] = x3;
    if (snap) { float* sp = snap + 3 * (size_t)region_off; sp[3 * c] = x1; sp[3 * c + 1] = x2; sp[3 * c + 2] = x3; }
    float acc = w[4 * c] * x0 + w[4 * c + 1] * x1 + w[4 * c + 2] * x2 + w[4 * c + 3] * x3;
    if (dyn) {
        const int h = c / (ch / n_heads);
        acc += dyn[4 * h] * x3 + dyn[4 * h + 1] * x2
             + dyn[4 * h + 2] * x1 + dyn[4 * h + 3] * x0;
    }
    const float conv = (x3 + acc) * scale;
    out[c] = __fadd_rn(residual[c], conv);
}

extern "C" void ink_launch_conv_step_add(const float* x_in, float* state, int region_off,
        const float* w, const float* dyn, const float* residual, int ch, int n_heads, float scale,
        float* out, float* snap, void* stream) {
    const int thr = ch < 256 ? ch : 256;
    inkwell_conv_step_add<<<(ch + thr - 1) / thr, thr, 0, (cudaStream_t)stream>>>(
        x_in, state, region_off, w, dyn, residual, ch, n_heads, scale, out, snap);
}

__device__ __forceinline__ float df_clip_factor(float c, float thr) {
    __shared__ float red[8];
    __shared__ float fshr;
    float v = c * c;
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float n = red[0];
        for (int j = 1; j < 8; ++j) n += red[j];
        fshr = fminf(1.f, thr / (sqrtf(n) + 1e-6f));
    }
    __syncthreads();
    return fshr;
}

__device__ __forceinline__ float df_conv_channel(
        const float* __restrict__ x_in, float* __restrict__ state, int region_off,
        const float* __restrict__ w, const float* __restrict__ dyn,
        int ch, int n_heads, float scale, int c) {
    float* st = state + 3 * (size_t)region_off;
    const float x0 = st[3 * c], x1 = st[3 * c + 1], x2 = st[3 * c + 2];
    const float x3 = x_in[c];
    st[3 * c] = x1; st[3 * c + 1] = x2; st[3 * c + 2] = x3;
    float acc = w[4 * c] * x0 + w[4 * c + 1] * x1 + w[4 * c + 2] * x2 + w[4 * c + 3] * x3;
    if (dyn) {
        const int h = c / (ch / n_heads);
        acc += dyn[4 * h] * x3 + dyn[4 * h + 1] * x2
             + dyn[4 * h + 2] * x1 + dyn[4 * h + 3] * x0;
    }
    return (x3 + acc) * scale;
}

__global__ void __launch_bounds__(256) inkwell_attn_qkvr_prep(
        const float* __restrict__ proj, int nmax,
        const float* __restrict__ dynkv,
        const float* __restrict__ kw, const float* __restrict__ vw,
        const float* __restrict__ qw, const float* __restrict__ rw,
        const float* __restrict__ tau,
        float* __restrict__ state, int k_region_off, int v_region_off,
        float* __restrict__ out,
        int hd, int nh, int hkv, int nrel,
        float conv_scale, float qk_thr, float rel_thr) {
    __shared__ __align__(16) float s_conv[128];
    const int b = blockIdx.x;
    const int i = threadIdx.x;
    if (b < nh) {
        const int h = b;
        float c = 0.f;
        if (i < hd) {
            c = proj[(size_t)h * hd + i];
            if (qw) c *= qw[(size_t)h * hd + i];
        }
        const float f = df_clip_factor(c, qk_thr);
        if (i < hd) {
            float y = c * f;
            if (tau) y *= tau[0];
            out[(size_t)h * hd + i] = y;
        }
        return;
    }
    if (b < nh + hkv) {
        const int h = b - nh;
        const int ch = hd * hkv;
        if (i < hd) {
            const int c = h * hd + i;
            s_conv[i] = df_conv_channel(proj + 3 * (size_t)nmax, state, k_region_off,
                kw, dynkv, ch, hkv, conv_scale, c);
        }
        __syncthreads();
        const float c = i < hd ? s_conv[i] : 0.f;
        const float f = df_clip_factor(c, qk_thr);
        if (i < hd) out[(size_t)nh * hd + (size_t)h * hd + i] = c * f;
        return;
    }
    if (b < nh + 2 * hkv) {
        const int h = b - nh - hkv;
        const int ch = hd * hkv;
        if (i < hd) {
            const int c = h * hd + i;
            s_conv[i] = df_conv_channel(proj + 4 * (size_t)nmax, state, v_region_off,
                vw, dynkv ? dynkv + 32 : nullptr, ch, hkv, conv_scale, c);
        }
        __syncthreads();
        if (i < 32) {
            const float4 xs = *(const float4*)(s_conv + 4 * i);
            float tmp = xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
            for (int o = 16; o; o >>= 1) tmp += __shfl_xor_sync(0xffffffffu, tmp, o);
            const float scale = rsqrtf(tmp / (float)hd + 1e-6f);
            const float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
            *(float4*)(out + (size_t)(nh + hkv) * hd + (size_t)h * hd + 4 * i) = r;
        }
        return;
    }
    const int h = b - nh - 2 * hkv;
    if (h < nh) {
        const float* rp = proj + 2 * (size_t)nmax;
        float c = 0.f;
        if (i < nrel) {
            c = rp[(size_t)h * nrel + i];
            if (rw) c *= rw[(size_t)h * nrel + i];
        }
        const float f = df_clip_factor(c, rel_thr);
        if (i < nrel) {
            float y = c * f;
            if (tau) y *= tau[0];
            out[(size_t)(nh + 2 * hkv) * hd + (size_t)h * nrel + i] = y;
        }
    }
}

extern "C" void ink_launch_attn_qkvr_prep(
        const float* proj, int nmax, const float* dynkv,
        const float* kw, const float* vw, const float* qw, const float* rw,
        const float* tau, float* state, int k_region_off, int v_region_off,
        float* out, int hd, int nh, int hkv, int nrel,
        float conv_scale, float qk_thr, float rel_thr, void* stream) {
    inkwell_attn_qkvr_prep<<<2 * nh + 2 * hkv, 256, 0, (cudaStream_t)stream>>>(
        proj, nmax, dynkv, kw, vw, qw, rw, tau, state, k_region_off, v_region_off,
        out, hd, nh, hkv, nrel, conv_scale, qk_thr, rel_thr);
}

#endif
