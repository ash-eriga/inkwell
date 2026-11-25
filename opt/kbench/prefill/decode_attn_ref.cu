// Verbatim extract of the production split-width decode attention path.
// Used only by the prefill harness as the comparison oracle.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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


