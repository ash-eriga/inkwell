#include "kernels.cu"
extern "C" const unsigned char ink_ch2nopless_cubin_start[1] = {0};
extern "C" const unsigned char ink_ch2nopless_cubin_end[1] = {0};

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#define F4E_SRC_N 4096
#define F4E_NLAYER 8

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        exit(2);
    }
}

struct InkF4ePtr {
    const unsigned char* kp;
    const unsigned char* vp;
    const unsigned char* mp;
};

__device__ __forceinline__ InkF3dRawKV ink_f4e_load_kv(const InkF4ePtr& p, int sl) {
    InkF3dRawKV r;
    const int grp = sl >> 3;
    r.kscw = __ldg((const unsigned short*)p.mp + grp);
    r.vw = __ldg((const unsigned int*)(p.mp + 4) + grp);
    r.kb = __ldg((const uint2*)p.kp);
    r.vb = __ldg((const unsigned int*)p.vp);
    return r;
}

template<int KS, int VS, int MS>
__device__ __forceinline__ void ink_f4e_step_kv(InkF4ePtr& p) {
    p.kp += KS;
    p.vp += VS;
    p.mp += MS;
}

template<bool LOW, bool CONST_BIAS>
__device__ __forceinline__ void ink_f4e_process_token(
        const InkF3dRawKV& qcur, const bool valid,
        const float bcur[3], const float mcur,
        const uint2 qpack[3], const uint2 qpackl[3], const float qscale_s[3],
        float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    const float kscv = ink_kvq_half_as_float(qcur.kscw);
    int isum[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int hx = __dp4a((int)qpack[g].x, (int)qcur.kb.x, 0);
        const int hy = __dp4a((int)qpack[g].y, (int)qcur.kb.y, 0);
        if constexpr (LOW) {
            int ls = __dp4a((int)qpackl[g].x, (int)qcur.kb.x, 0);
            ls = __dp4a((int)qpackl[g].y, (int)qcur.kb.y, ls);
            isum[g] = ((hx + hy) << 7) + ls;
        } else {
            isum[g] = ((hx + hy) << 7);
        }
    }
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) isum[g] += __shfl_xor_sync(0xffffffffu, isum[g], o);
    }
    float sc[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) sc[g] = (float)isum[g] * kscv;
    #pragma unroll
    for (int g = 0; g < 3; ++g) sc[g] += __shfl_xor_sync(0xffffffffu, sc[g], 8);
    #pragma unroll
    for (int g = 0; g < 3; ++g) sc[g] *= qscale_s[g];
    if constexpr (!CONST_BIAS) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) sc[g] += bcur[g] + mcur;
    }

    const float vsc = ink_kvq_half_as_float(qcur.vw & 0xffffu);
    const float vmn = ink_kvq_half_as_float(qcur.vw >> 16);
    const unsigned int lo4 = qcur.vb & 0x0f0f0f0fu;
    const unsigned int hi4 = (qcur.vb >> 4) & 0x0f0f0f0fu;
    float qf[8];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        qf[2 * k] = (float)((lo4 >> (8 * k)) & 0xffu);
        qf[2 * k + 1] = (float)((hi4 >> (8 * k)) & 0xffu);
    }
    if (valid) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            const float mold = mst[g];
            if (sc[g] > mold) {
                const float c = __expf(mold - sc[g]);
                mst[g] = sc[g];
                lst[g] = fmaf(lst[g], c, 1.f);
                bacc[g] = fmaf(bacc[g], c, vmn);
                #pragma unroll
                for (int f = 0; f < 8; ++f) acc[g][f] = fmaf(vsc, qf[f], acc[g][f] * c);
            } else {
                const float p = __expf(sc[g] - mold);
                lst[g] += p;
                bacc[g] = fmaf(p, vmn, bacc[g]);
                const float wv = p * vsc;
                #pragma unroll
                for (int f = 0; f < 8; ++f) acc[g][f] = fmaf(wv, qf[f], acc[g][f]);
            }
        }
    }
}

template<int NST, int KS, int VS, int MS, bool LOW, bool CONST_BIAS>
__device__ __forceinline__ void ink_f4e_rows(
        InkF4ePtr qp, const int j0, const int j1, const int stream, const int sl,
        const float* __restrict__ bias, const float* __restrict__ mask,
        const int hq0, const int b_s2, const int pos, const int n_kv,
        const int bias_mode, const int swa, const int rel_extent, const int mask_zero,
        const uint2 qpack[3], const uint2 qpackl[3], const float qscale_s[3],
        float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    InkF3dRawKV qcur{}, qnext{};
    float bcur[3] = {}, bnext[3] = {}, mcur = 0.f, mnext = 0.f;
    const int first = j0 + stream;
    if (first < j1) {
        qcur = ink_f4e_load_kv(qp, sl);
        if constexpr (!CONST_BIAS) {
            #pragma unroll
            for (int g = 0; g < 3; ++g) bcur[g] = ink_attn_bias_value(bias, first, hq0 + g, b_s2, pos, n_kv, bias_mode, swa, rel_extent);
            mcur = mask_zero ? 0.f : mask[first];
        }
    }
    for (int jb = j0; jb < j1; jb += NST) {
        const int j = jb + stream;
        const bool valid = j < j1;
        const int jn = jb + NST + stream;
        ink_f4e_step_kv<KS, VS, MS>(qp);
        if (jn < j1) {
            qnext = ink_f4e_load_kv(qp, sl);
            if constexpr (!CONST_BIAS) {
                #pragma unroll
                for (int g = 0; g < 3; ++g) bnext[g] = ink_attn_bias_value(bias, jn, hq0 + g, b_s2, pos, n_kv, bias_mode, swa, rel_extent);
                mnext = mask_zero ? 0.f : mask[jn];
            }
        }
        ink_f4e_process_token<LOW, CONST_BIAS>(qcur, valid, bcur, mcur, qpack, qpackl, qscale_s,
                                               acc, bacc, mst, lst);
        qcur = qnext;
        if constexpr (!CONST_BIAS) {
            #pragma unroll
            for (int g = 0; g < 3; ++g) bcur[g] = bnext[g];
            mcur = mnext;
        }
    }
}

template<int NST, int KS, int VS, int MS, bool LOW>
__device__ __forceinline__ void ink_f4e_row_epilogue(
        const int hq0, const int chunk, const int nh, const int hk, const int stream, const int sl,
        const int rw, const int w, const int d0, const bool const_bias,
        const float* __restrict__ bias, const int b_s2, const int rel_extent,
        float* __restrict__ part, float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    constexpr int NW = NST / 2;
    extern __shared__ float sw[];
    float* sacc = sw;
    float* smm = sacc + NW * 3 * 128;
    float* sml = smm + NW * 3;
    float* smb = sml + NW * 3;

    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const float mo = __shfl_xor_sync(0xffffffffu, mst[g], 16);
        const float lo = __shfl_xor_sync(0xffffffffu, lst[g], 16);
        const float bo = __shfl_xor_sync(0xffffffffu, bacc[g], 16);
        const float mx = fmaxf(mst[g], mo);
        const float wc = __expf(mst[g] - mx);
        const float wo = __expf(mo - mx);
        mst[g] = mx;
        lst[g] = lst[g] * wc + lo * wo;
        bacc[g] = bacc[g] * wc + bo * wo;
        #pragma unroll
        for (int f = 0; f < 8; ++f) {
            const float ao = __shfl_xor_sync(0xffffffffu, acc[g][f], 16);
            acc[g][f] = acc[g][f] * wc + ao * wo;
        }
    }

    if (rw == 0) {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            if (sl == 0) {
                smm[w * 3 + g] = mst[g];
                sml[w * 3 + g] = lst[g];
            }
            smb[(w * 3 + g) * 16 + sl] = bacc[g];
            #pragma unroll
            for (int f = 0; f < 8; ++f) sacc[(w * 3 + g) * 128 + d0 + f] = acc[g][f];
        }
    }
    __syncthreads();

    if (stream < 3) {
        const int g = stream;
        float mx = -1e30f, ls = 0.f, bs = 0.f, xv[8] = {};
        #pragma unroll
        for (int ws = 0; ws < NW; ++ws) {
            const float mm = smm[ws * 3 + g];
            const float ll = sml[ws * 3 + g];
            const float bb = smb[(ws * 3 + g) * 16 + sl];
            float av[8];
            #pragma unroll
            for (int f = 0; f < 8; ++f) av[f] = sacc[(ws * 3 + g) * 128 + d0 + f];
            if (mm > mx) {
                const float c = __expf(mx - mm);
                mx = mm;
                ls = ls * c + ll;
                bs = bs * c + bb;
                #pragma unroll
                for (int f = 0; f < 8; ++f) xv[f] = xv[f] * c + av[f];
            } else {
                const float c = __expf(mm - mx);
                ls = fmaf(ll, c, ls);
                bs = fmaf(bb, c, bs);
                #pragma unroll
                for (int f = 0; f < 8; ++f) xv[f] = fmaf(av[f], c, xv[f]);
            }
        }
        float* pb = part + ((size_t)chunk * nh + hq0 + g) * (128 + 2);
        if (const_bias && sl == 0) mx += __ldg(bias + rel_extent + (hq0 + g) * b_s2);
        if (sl == 0) { pb[0] = mx; pb[1] = ls; }
        #pragma unroll
        for (int f = 0; f < 8; ++f) pb[2 + d0 + f] = xv[f] + bs;
    }
}

template<int NST, int METAW, bool LOW = true>
__global__ void inkwell_attn_kvq_f4e_hm(InkKvQSlot slot, const int* __restrict__ posp,
        const float* __restrict__ Q, const float* __restrict__ bias,
        const float* __restrict__ mask, float* __restrict__ part,
        int nh, const int* __restrict__ d_nkv, int q_s2, int b_s2, float scale,
        int chunk_len, int mask_zero, int bias_mode, int swa, int rel_extent,
        const unsigned char* __restrict__ qpacks, const float* __restrict__ qscales) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sl = lane & 15;
    const int rw = lane >> 4;
    const int w = tid >> 5;
    const int stream = tid >> 4;
    const int chunk = blockIdx.x;
    const int hk = blockIdx.y;
    const int ratio = nh / 8;
    if (ratio != 3) return;
    const int hq0 = hk * ratio;
    const int n_kv = min(d_nkv[0], slot.cap);
    const int j0 = chunk * chunk_len;
    if (j0 >= n_kv) return;
    const int j1 = min(n_kv, j0 + chunk_len);
    const int d0 = sl * 8;
    const int pos = posp[0];

    uint2 qpack[3], qpackl[3];
    float qscale_s[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = hq0 + g;
        qpack[g] = __ldg((const uint2*)(qpacks + h * 128 + d0));
        qpackl[g] = __ldg((const uint2*)(qpacks + 3072 + h * 128 + d0));
        qscale_s[g] = __ldg(qscales + h) * scale * 0.0078125f;
    }

    float acc[3][8] = {};
    float bacc[3] = {}, mst[3], lst[3] = {};
    #pragma unroll
    for (int g = 0; g < 3; ++g) mst[g] = -1e30f;

    const int row = j0 + stream;
    const size_t hb = (size_t)hk * slot.cap + row;
    InkF4ePtr qp;
    qp.kp = slot.kq + hb * 128 + d0;
    qp.vp = slot.vq + hb * 64 + sl * 4;
    qp.mp = slot.meta + hb * METAW;
    constexpr int KS = NST * 128;
    constexpr int VS = NST * 64;
    constexpr int MS = NST * METAW;
    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;
    if (const_bias) {
        ink_f4e_rows<NST, KS, VS, MS, LOW, true>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
            bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
    } else {
        ink_f4e_rows<NST, KS, VS, MS, LOW, false>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
            bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
    }
    ink_f4e_row_epilogue<NST, KS, VS, MS, LOW>(hq0, chunk, nh, hk, stream, sl, rw, w, d0, const_bias,
        bias, b_s2, rel_extent, part, acc, bacc, mst, lst);
}

template<int NST>
__global__ void inkwell_attn_kvq_f4e_pack(InkKvQSlot slot, const int* __restrict__ posp,
        const float* __restrict__ Q, const float* __restrict__ bias,
        const float* __restrict__ mask, float* __restrict__ part,
        int nh, const int* __restrict__ d_nkv, int q_s2, int b_s2, float scale,
        int chunk_len, int mask_zero, int bias_mode, int swa, int rel_extent,
        const unsigned char* __restrict__ qpacks, const float* __restrict__ qscales) {
    static_assert(NST == 8, "packed tile8 layout currently pairs with NST=8");
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sl = lane & 15;
    const int rw = lane >> 4;
    const int w = tid >> 5;
    const int stream = tid >> 4;
    const int chunk = blockIdx.x;
    const int hk = blockIdx.y;
    const int ratio = nh / 8;
    if (ratio != 3) return;
    const int hq0 = hk * ratio;
    const int n_kv = min(d_nkv[0], slot.cap);
    const int j0 = chunk * chunk_len;
    if (j0 >= n_kv) return;
    const int j1 = min(n_kv, j0 + chunk_len);
    const int d0 = sl * 8;
    const int pos = posp[0];

    uint2 qpack[3], qpackl[3];
    float qscale_s[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = hq0 + g;
        qpack[g] = __ldg((const uint2*)(qpacks + h * 128 + d0));
        qpackl[g] = __ldg((const uint2*)(qpacks + 3072 + h * 128 + d0));
        qscale_s[g] = __ldg(qscales + h) * scale * 0.0078125f;
    }

    float acc[3][8] = {};
    float bacc[3] = {}, mst[3], lst[3] = {};
    #pragma unroll
    for (int g = 0; g < 3; ++g) mst[g] = -1e30f;

    const int row = j0 + stream;
    const int tile = row >> 3;
    const int r = row & 7;
    const int ntiles = slot.cap >> 3;
    const unsigned char* tb = slot.kq + ((size_t)hk * ntiles + tile) * 1632;
    InkF4ePtr qp;
    qp.kp = tb + r * 128 + d0;
    qp.vp = tb + 1024 + r * 64 + sl * 4;
    qp.mp = tb + 1536 + r * 12;
    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;
    if (const_bias) {
        ink_f4e_rows<NST, 1632, 1632, 1632, true, true>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
            bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
    } else {
        ink_f4e_rows<NST, 1632, 1632, 1632, true, false>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
            bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
    }
    ink_f4e_row_epilogue<NST, 1632, 1632, 1632, true>(hq0, chunk, nh, hk, stream, sl, rw, w, d0, const_bias,
        bias, b_s2, rel_extent, part, acc, bacc, mst, lst);
}

struct InkF4ePairRaw {
    InkF3dRawKV a;
    InkF3dRawKV b;
};

__device__ __forceinline__ InkF4ePairRaw ink_f4e_load_pair(const InkF4ePtr& p, int sl) {
    InkF4ePairRaw r{};
    const int grp = sl >> 3;
    const uint4 kb = __ldg((const uint4*)p.kp);
    const uint2 vb = __ldg((const uint2*)p.vp);
    r.a.kb = make_uint2(kb.x, kb.y);
    r.b.kb = make_uint2(kb.z, kb.w);
    r.a.vb = vb.x;
    r.b.vb = vb.y;
    r.a.kscw = __ldg((const unsigned short*)p.mp + grp);
    r.a.vw = __ldg((const unsigned int*)(p.mp + 4) + grp);
    r.b.kscw = __ldg((const unsigned short*)(p.mp + 12) + grp);
    r.b.vw = __ldg((const unsigned int*)(p.mp + 16) + grp);
    return r;
}

template<bool CONST_BIAS>
__device__ __forceinline__ void ink_f4e_rows_vec2(
        InkF4ePtr qp, const int j0, const int j1, const int stream, const int sl,
        const float* __restrict__ bias, const float* __restrict__ mask,
        const int hq0, const int b_s2, const int pos, const int n_kv,
        const int bias_mode, const int swa, const int rel_extent, const int mask_zero,
        const uint2 qpack[3], const uint2 qpackl[3], const float qscale_s[3],
        float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    constexpr int NST = 8;
    InkF4ePairRaw qcur{}, qnext{};
    float bcur[2][3] = {}, bnext[2][3] = {}, mcur[2] = {}, mnext[2] = {};
    const int first = j0 + stream;
    if (first < j1) {
        qcur = ink_f4e_load_pair(qp, sl);
        if constexpr (!CONST_BIAS) {
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int jj = first + t * NST;
                if (jj < j1) {
                    #pragma unroll
                    for (int g = 0; g < 3; ++g) bcur[t][g] = ink_attn_bias_value(bias, jj, hq0 + g, b_s2, pos, n_kv, bias_mode, swa, rel_extent);
                    mcur[t] = mask_zero ? 0.f : mask[jj];
                }
            }
        }
    }
    for (int jb = j0; jb < j1; jb += 2 * NST) {
        const int ja = jb + stream;
        const int jbv = ja + NST;
        const int jn = jb + 2 * NST + stream;
        qp.kp += 2048;
        qp.vp += 1024;
        qp.mp += 192;
        if (jn < j1) {
            qnext = ink_f4e_load_pair(qp, sl);
            if constexpr (!CONST_BIAS) {
                #pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int jj = jn + t * NST;
                    if (jj < j1) {
                        #pragma unroll
                        for (int g = 0; g < 3; ++g) bnext[t][g] = ink_attn_bias_value(bias, jj, hq0 + g, b_s2, pos, n_kv, bias_mode, swa, rel_extent);
                        mnext[t] = mask_zero ? 0.f : mask[jj];
                    }
                }
            }
        }
        ink_f4e_process_token<true, CONST_BIAS>(qcur.a, ja < j1, bcur[0], mcur[0], qpack, qpackl, qscale_s,
                                                acc, bacc, mst, lst);
        ink_f4e_process_token<true, CONST_BIAS>(qcur.b, jbv < j1, bcur[1], mcur[1], qpack, qpackl, qscale_s,
                                                acc, bacc, mst, lst);
        qcur = qnext;
        if constexpr (!CONST_BIAS) {
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                #pragma unroll
                for (int g = 0; g < 3; ++g) bcur[t][g] = bnext[t][g];
                mcur[t] = mnext[t];
            }
        }
    }
}

__global__ void inkwell_attn_kvq_f4e_vec2(InkKvQSlot slot, const int* __restrict__ posp,
        const float* __restrict__ Q, const float* __restrict__ bias,
        const float* __restrict__ mask, float* __restrict__ part,
        int nh, const int* __restrict__ d_nkv, int q_s2, int b_s2, float scale,
        int chunk_len, int mask_zero, int bias_mode, int swa, int rel_extent,
        const unsigned char* __restrict__ qpacks, const float* __restrict__ qscales) {
    constexpr int NST = 8;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sl = lane & 15;
    const int rw = lane >> 4;
    const int w = tid >> 5;
    const int stream = tid >> 4;
    const int chunk = blockIdx.x;
    const int hk = blockIdx.y;
    const int ratio = nh / 8;
    if (ratio != 3) return;
    const int hq0 = hk * ratio;
    const int n_kv = min(d_nkv[0], slot.cap);
    const int j0 = chunk * chunk_len;
    if (j0 >= n_kv) return;
    const int j1 = min(n_kv, j0 + chunk_len);
    const int d0 = sl * 8;
    const int pos = posp[0];

    uint2 qpack[3], qpackl[3];
    float qscale_s[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int h = hq0 + g;
        qpack[g] = __ldg((const uint2*)(qpacks + h * 128 + d0));
        qpackl[g] = __ldg((const uint2*)(qpacks + 3072 + h * 128 + d0));
        qscale_s[g] = __ldg(qscales + h) * scale * 0.0078125f;
    }

    float acc[3][8] = {};
    float bacc[3] = {}, mst[3], lst[3] = {};
    #pragma unroll
    for (int g = 0; g < 3; ++g) mst[g] = -1e30f;

    const int row = j0 + stream;
    const int tile = row >> 4;
    const int s = row & 7;
    const int ntiles = slot.cap >> 4;
    InkF4ePtr qp;
    qp.kp = slot.kq + ((size_t)hk * ntiles + tile) * 2048 + s * 256 + sl * 16;
    qp.vp = slot.vq + ((size_t)hk * ntiles + tile) * 1024 + s * 128 + sl * 8;
    qp.mp = slot.meta + ((size_t)hk * ntiles + tile) * 192 + s * 24;
    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;
    if (const_bias) {
        ink_f4e_rows_vec2<true>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
            bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
    } else {
        ink_f4e_rows_vec2<false>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
            bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
    }
    ink_f4e_row_epilogue<NST, 2048, 1024, 192, true>(hq0, chunk, nh, hk, stream, sl, rw, w, d0, const_bias,
        bias, b_s2, rel_extent, part, acc, bacc, mst, lst);
}

__global__ void ink_f4e_quant_from_dump(const __half* __restrict__ kd, const __half* __restrict__ vd,
                                        InkKvQSlot slot, int src_n) {
    __shared__ float ek[128], ev[128], mn[128], mx[128];
    __shared__ unsigned char qk[128], qv[128], qklo[128];
    __shared__ float ksc_s[2], vsc_s[4], vmn_s[4];
    const int hk = blockIdx.y;
    const int d = threadIdx.x;
    const int q0 = blockIdx.x * 8;
    const int q1 = min(q0 + 8, slot.cap);
    for (int qrow = q0; qrow < q1; ++qrow) {
        const int sr = qrow % src_n;
        const size_t src = ((size_t)hk * src_n + sr) * 128 + d;
        ink_kvq_store_quant(slot, qrow, hk, d, __half2float(kd[src]), __half2float(vd[src]),
                            ek, ev, mn, mx, qk, qv, qklo, ksc_s, vsc_s, vmn_s);
    }
}

__global__ void ink_f4e_transpose_hm(const unsigned char* __restrict__ skq,
        const unsigned char* __restrict__ svq, const unsigned char* __restrict__ smeta,
        unsigned char* __restrict__ dkq, unsigned char* __restrict__ dvq, unsigned char* __restrict__ dmeta,
        int cap, int meta_w) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t kn = (size_t)cap * 1024;
    const size_t vn = (size_t)cap * 512;
    const size_t mn = (size_t)cap * 96;
    if (i < kn) {
        const int row = (int)(i >> 10);
        const int x = (int)(i & 1023);
        const int hk = x >> 7;
        const int b = x & 127;
        dkq[((size_t)hk * cap + row) * 128 + b] = skq[i];
    }
    if (i < vn) {
        const int row = (int)(i >> 9);
        const int x = (int)(i & 511);
        const int hk = x >> 6;
        const int b = x & 63;
        dvq[((size_t)hk * cap + row) * 64 + b] = svq[i];
    }
    if (i < mn) {
        const int row = (int)(i / 96);
        const int x = (int)(i % 96);
        const int hk = x / 12;
        const int b = x % 12;
        if (meta_w == 12) dmeta[((size_t)hk * cap + row) * 12 + b] = smeta[i];
        else dmeta[((size_t)hk * cap + row) * 16 + b] = smeta[i];
    }
}

__global__ void ink_f4e_transpose_pack(const unsigned char* __restrict__ skq,
        const unsigned char* __restrict__ svq, const unsigned char* __restrict__ smeta,
        unsigned char* __restrict__ pack, int cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t kn = (size_t)cap * 1024;
    const size_t vn = (size_t)cap * 512;
    const size_t mn = (size_t)cap * 96;
    const int ntiles = cap >> 3;
    if (i < kn) {
        const int row = (int)(i >> 10);
        const int x = (int)(i & 1023);
        const int hk = x >> 7;
        const int b = x & 127;
        unsigned char* tb = pack + ((size_t)hk * ntiles + (row >> 3)) * 1632;
        tb[(row & 7) * 128 + b] = skq[i];
    }
    if (i < vn) {
        const int row = (int)(i >> 9);
        const int x = (int)(i & 511);
        const int hk = x >> 6;
        const int b = x & 63;
        unsigned char* tb = pack + ((size_t)hk * ntiles + (row >> 3)) * 1632;
        tb[1024 + (row & 7) * 64 + b] = svq[i];
    }
    if (i < mn) {
        const int row = (int)(i / 96);
        const int x = (int)(i % 96);
        const int hk = x / 12;
        const int b = x % 12;
        unsigned char* tb = pack + ((size_t)hk * ntiles + (row >> 3)) * 1632;
        tb[1536 + (row & 7) * 12 + b] = smeta[i];
    }
}

__global__ void ink_f4e_transpose_vec2(const unsigned char* __restrict__ skq,
        const unsigned char* __restrict__ svq, const unsigned char* __restrict__ smeta,
        unsigned char* __restrict__ dkq, unsigned char* __restrict__ dvq, unsigned char* __restrict__ dmeta,
        int cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t kn = (size_t)cap * 1024;
    const size_t vn = (size_t)cap * 512;
    const size_t mn = (size_t)cap * 96;
    const int ntiles = cap >> 4;
    if (i < kn) {
        const int row = (int)(i >> 10);
        const int x = (int)(i & 1023);
        const int hk = x >> 7;
        const int b = x & 127;
        const int t = row & 15;
        const int s = t & 7;
        const int half = t >> 3;
        const int sl = b >> 3;
        const int bb = b & 7;
        dkq[((size_t)hk * ntiles + (row >> 4)) * 2048 + s * 256 + sl * 16 + half * 8 + bb] = skq[i];
    }
    if (i < vn) {
        const int row = (int)(i >> 9);
        const int x = (int)(i & 511);
        const int hk = x >> 6;
        const int b = x & 63;
        const int t = row & 15;
        const int s = t & 7;
        const int half = t >> 3;
        const int sl = b >> 2;
        const int bb = b & 3;
        dvq[((size_t)hk * ntiles + (row >> 4)) * 1024 + s * 128 + sl * 8 + half * 4 + bb] = svq[i];
    }
    if (i < mn) {
        const int row = (int)(i / 96);
        const int x = (int)(i % 96);
        const int hk = x / 12;
        const int b = x % 12;
        const int t = row & 15;
        const int s = t & 7;
        const int half = t >> 3;
        dmeta[((size_t)hk * ntiles + (row >> 4)) * 192 + s * 24 + half * 12 + b] = smeta[i];
    }
}

template<int LAYOUT>
__global__ void ink_f4e_memrows(InkKvQSlot slot, const int* __restrict__ d_nkv,
                                int chunk_len, unsigned int* __restrict__ out) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sl = lane & 15;
    const int w = tid >> 5;
    const int stream = tid >> 4;
    const int chunk = blockIdx.x;
    const int hk = blockIdx.y;
    const int n_kv = min(d_nkv[0], slot.cap);
    const int j0 = chunk * chunk_len;
    if (j0 >= n_kv) return;
    const int j1 = min(n_kv, j0 + chunk_len);
    unsigned int x = 0;
    if constexpr (LAYOUT == 2) {
        const int row = j0 + stream;
        const int tile = row >> 4;
        const int s = row & 7;
        const int ntiles = slot.cap >> 4;
        const unsigned char* kp = slot.kq + ((size_t)hk * ntiles + tile) * 2048 + s * 256 + sl * 16;
        const unsigned char* vp = slot.vq + ((size_t)hk * ntiles + tile) * 1024 + s * 128 + sl * 8;
        const unsigned char* mp = slot.meta + ((size_t)hk * ntiles + tile) * 192 + s * 24;
        for (int jb = j0; jb < j1; jb += 16) {
            const uint4 kb = __ldg((const uint4*)kp);
            const uint2 vb = __ldg((const uint2*)vp);
            const int grp = sl >> 3;
            x ^= kb.x ^ kb.y ^ kb.z ^ kb.w ^ vb.x ^ vb.y;
            x ^= __ldg((const unsigned short*)mp + grp);
            x ^= __ldg((const unsigned int*)(mp + 4) + grp);
            x ^= __ldg((const unsigned short*)(mp + 12) + grp);
            x ^= __ldg((const unsigned int*)(mp + 16) + grp);
            kp += 2048; vp += 1024; mp += 192;
        }
    } else {
        const int row = j0 + stream;
        const unsigned char *kp, *vp, *mp;
        int ks, vs, ms;
        if constexpr (LAYOUT == 0) {
            kp = slot.kq + (size_t)row * 1024 + hk * 128 + sl * 8;
            vp = slot.vq + (size_t)row * 512 + hk * 64 + sl * 4;
            mp = slot.meta + (size_t)row * 96 + hk * 12;
            ks = 8 * 1024; vs = 8 * 512; ms = 8 * 96;
        } else {
            const size_t hb = (size_t)hk * slot.cap + row;
            kp = slot.kq + hb * 128 + sl * 8;
            vp = slot.vq + hb * 64 + sl * 4;
            mp = slot.meta + hb * 12;
            ks = 8 * 128; vs = 8 * 64; ms = 8 * 12;
        }
        for (int jb = j0; jb < j1; jb += 8) {
            const uint2 kb = __ldg((const uint2*)kp);
            const unsigned int vb = __ldg((const unsigned int*)vp);
            const int grp = sl >> 3;
            x ^= kb.x ^ kb.y ^ vb;
            x ^= __ldg((const unsigned short*)mp + grp);
            x ^= __ldg((const unsigned int*)(mp + 4) + grp);
            kp += ks; vp += vs; mp += ms;
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) out[((size_t)chunk * gridDim.y + hk) * 4 + w] = x;
}

__global__ void ink_f4e_seqread(const uint4* __restrict__ p, size_t n16, unsigned int* __restrict__ out) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned int x = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        const uint4 v = __ldg(p + i);
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x ^= __shfl_xor_sync(0xffffffffu, x, o);
    if ((threadIdx.x & 31) == 0) out[blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)] = x;
}

static bool read_file(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const size_t n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(n);
    f.read((char*)out.data(), (std::streamsize)n);
    return f.good() || f.eof();
}

struct DevLayer {
    InkKvQSlot ctl{};
    InkKvQSlot hm{};
    InkKvQSlot hm16{};
    InkKvQSlot pack{};
    InkKvQSlot vec{};
    float* q = nullptr;
    float* bias = nullptr;
    unsigned char *ckq = nullptr, *cvq = nullptr, *cmeta = nullptr;
    unsigned char *hkq = nullptr, *hvq = nullptr, *hmeta = nullptr, *hmeta16 = nullptr;
    unsigned char* packbuf = nullptr;
    unsigned char *vkq = nullptr, *vvq = nullptr, *vmeta = nullptr;
};

static float time_kernel(const std::vector<std::function<void()>>& fns, int which, int reps) {
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        fns[which]();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        ts.push_back(ms);
        cuda_check(cudaGetLastError(), "timed launch");
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

template<int NST>
static inline void launch_ctl_row(const DevLayer& L, const dim3& gp, int chunk_len,
                                  const int* dpos, const float* mask, float* part, const int* dnkv,
                                  int nh, int q_s2, int b_s2, float scale,
                                  const unsigned char* qbuf, const float* qscales) {
    constexpr int NW = NST / 2;
    const size_t sm = (size_t)(NW * 3 * 128 + NW * 3 * 2 + NW * 3 * 16) * 4;
    inkwell_attn_kvq_gqa3_v2_row<INK_KVQ_K8V4, NST><<<gp, dim3(NST * 16), sm>>>(
        L.ctl, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

template<int NST, int METAW, bool LOW = true>
static inline void launch_hm(const DevLayer& L, const dim3& gp, int chunk_len,
                             const int* dpos, const float* mask, float* part, const int* dnkv,
                             int nh, int q_s2, int b_s2, float scale,
                             const unsigned char* qbuf, const float* qscales) {
    constexpr int NW = NST / 2;
    const size_t sm = (size_t)(NW * 3 * 128 + NW * 3 * 2 + NW * 3 * 16) * 4;
    const InkKvQSlot& s = METAW == 16 ? L.hm16 : L.hm;
    inkwell_attn_kvq_f4e_hm<NST, METAW, LOW><<<gp, dim3(NST * 16), sm>>>(
        s, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

static inline void launch_pack(const DevLayer& L, const dim3& gp, int chunk_len,
                               const int* dpos, const float* mask, float* part, const int* dnkv,
                               int nh, int q_s2, int b_s2, float scale,
                               const unsigned char* qbuf, const float* qscales) {
    const size_t sm = (size_t)(4 * 3 * 128 + 4 * 3 * 2 + 4 * 3 * 16) * 4;
    inkwell_attn_kvq_f4e_pack<8><<<gp, dim3(128), sm>>>(
        L.pack, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

static inline void launch_vec2(const DevLayer& L, const dim3& gp, int chunk_len,
                               const int* dpos, const float* mask, float* part, const int* dnkv,
                               int nh, int q_s2, int b_s2, float scale,
                               const unsigned char* qbuf, const float* qscales) {
    const size_t sm = (size_t)(4 * 3 * 128 + 4 * 3 * 2 + 4 * 3 * 16) * 4;
    inkwell_attn_kvq_f4e_vec2<<<gp, dim3(128), sm>>>(
        L.vec, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    const int n_kv = argc > 3 ? atoi(argv[3]) : 65536, nh = 24, hkv = 8, hd = 128;
    const int q_s2 = 128, b_s2 = 1025;
    const float scale = 1.0f / sqrtf(128.f);
    const char* mode = argc > 1 ? argv[1] : "real";
    const int chunk_len = argc > 2 ? atoi(argv[2]) : 256;
    const int reps = getenv("F4E_REPS") ? atoi(getenv("F4E_REPS")) : 7;
    const bool uniform = strcmp(mode, "uniform") == 0;
    const std::string dumproot = getenv("F4E_DUMP") ? getenv("F4E_DUMP") :
        "/home/ash/.cache/inkwell-intkv/attn4k-v26-nocache/4096";
    const char* names[F4E_NLAYER] = {"l5p0", "l11p0", "l11p1", "l11p2", "l17p0", "l17p1", "l17p2", "l23p0"};
    if (chunk_len < 32 || chunk_len > 1024 || (chunk_len & 15) != 0 || (n_kv & 15) != 0) {
        fprintf(stderr, "chunk_len must be a multiple of 16 in [32,1024], n_kv multiple of 16\n");
        return 2;
    }

    std::vector<DevLayer> layers(F4E_NLAYER);
    const size_t kbytes = (size_t)n_kv * 1024, vbytes = (size_t)n_kv * 512, mbytes = (size_t)n_kv * 96;
    const size_t m16bytes = (size_t)n_kv * 128;
    const size_t packbytes = kbytes + vbytes + mbytes;
    const float tau = 1.f + 0.1f * logf((float)n_kv / 8192.f);
    for (int li = 0; li < F4E_NLAYER; ++li) {
        DevLayer& L = layers[li];
        cuda_check(cudaMalloc(&L.ckq, kbytes), "malloc ckq");
        cuda_check(cudaMalloc(&L.cvq, vbytes), "malloc cvq");
        cuda_check(cudaMalloc(&L.cmeta, mbytes), "malloc cmeta");
        cuda_check(cudaMalloc(&L.hkq, kbytes), "malloc hkq");
        cuda_check(cudaMalloc(&L.hvq, vbytes), "malloc hvq");
        cuda_check(cudaMalloc(&L.hmeta, mbytes), "malloc hmeta");
        cuda_check(cudaMalloc(&L.hmeta16, m16bytes), "malloc hmeta16");
        cudaMemset(L.hmeta16, 0, m16bytes);
        cuda_check(cudaMalloc(&L.packbuf, packbytes), "malloc pack");
        cuda_check(cudaMalloc(&L.vkq, kbytes), "malloc vkq");
        cuda_check(cudaMalloc(&L.vvq, vbytes), "malloc vvq");
        cuda_check(cudaMalloc(&L.vmeta, mbytes), "malloc vmeta");
        cuda_check(cudaMalloc(&L.q, nh * hd * 4), "malloc q");
        cuda_check(cudaMalloc(&L.bias, nh * b_s2 * 4), "malloc bias");
        if (uniform) {
            cudaMemset(L.ckq, 0, kbytes);
            cudaMemset(L.cvq, 0, vbytes);
            cudaMemset(L.cmeta, 0, mbytes);
            cudaMemset(L.q, 0, nh * hd * 4);
            cudaMemset(L.bias, 0, nh * b_s2 * 4);
        } else {
            const std::string dir = dumproot + "/" + names[li];
            std::vector<unsigned char> kb, vb, qb, bb;
            if (!read_file(dir + "/k.f16", kb) || !read_file(dir + "/v.f16", vb) ||
                !read_file(dir + "/q.f32", qb) || !read_file(dir + "/bias.f32", bb)) {
                fprintf(stderr, "missing realistic dump in %s\n", dir.c_str());
                return 2;
            }
            __half *dk = nullptr, *dv = nullptr;
            cuda_check(cudaMalloc(&dk, kb.size()), "malloc dump k");
            cuda_check(cudaMalloc(&dv, vb.size()), "malloc dump v");
            cudaMemcpy(dk, kb.data(), kb.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dv, vb.data(), vb.size(), cudaMemcpyHostToDevice);
            InkKvQSlot s{};
            s.kq = L.ckq; s.vq = L.cvq; s.meta = L.cmeta;
            s.cap = n_kv; s.hp_cap = 0; s.swa = 1024; s.sinks = 0; s.local = 0; s.is_swa = 0;
            s.fmt = INK_KVQ_K8V4; s.krow = 1024; s.vrow = 512; s.mrow = 96;
            dim3 qg((unsigned)((n_kv + 7) / 8), hkv);
            ink_f4e_quant_from_dump<<<qg, 128>>>(dk, dv, s, F4E_SRC_N);
            cuda_check(cudaGetLastError(), "quant launch");
            cuda_check(cudaDeviceSynchronize(), "quant sync");
            cudaFree(dk);
            cudaFree(dv);

            std::vector<float> hq((const float*)qb.data(), (const float*)qb.data() + nh * hd);
            for (float& x : hq) x *= tau;
            cudaMemcpy(L.q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
            const float* bd = (const float*)bb.data();
            std::vector<float> table(nh * b_s2, 0.f);
            const int pos4 = F4E_SRC_N - 1;
            for (int h = 0; h < nh; ++h) {
                for (int rp = 0; rp < b_s2; ++rp) {
                    const int j = pos4 - rp;
                    table[h * b_s2 + rp] = bd[h * F4E_SRC_N + j] * tau;
                }
            }
            cudaMemcpy(L.bias, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
        }

        const int trthreads = 256;
        const int trblocks = (int)((kbytes + trthreads - 1) / trthreads);
        ink_f4e_transpose_hm<<<trblocks, trthreads>>>(L.ckq, L.cvq, L.cmeta, L.hkq, L.hvq, L.hmeta, n_kv, 12);
        ink_f4e_transpose_hm<<<trblocks, trthreads>>>(L.ckq, L.cvq, L.cmeta, L.hkq, L.hvq, L.hmeta16, n_kv, 16);
        ink_f4e_transpose_pack<<<trblocks, trthreads>>>(L.ckq, L.cvq, L.cmeta, L.packbuf, n_kv);
        ink_f4e_transpose_vec2<<<trblocks, trthreads>>>(L.ckq, L.cvq, L.cmeta, L.vkq, L.vvq, L.vmeta, n_kv);
        cuda_check(cudaGetLastError(), "transpose launch");

        auto fill_slot = [&](InkKvQSlot& s, unsigned char* k, unsigned char* v, unsigned char* m) {
            s.kq = k; s.vq = v; s.meta = m;
            s.cap = n_kv; s.hp_cap = 0; s.swa = 1024; s.sinks = 0; s.local = 0; s.is_swa = 0;
            s.fmt = INK_KVQ_K8V4; s.krow = 1024; s.vrow = 512; s.mrow = 96;
        };
        fill_slot(L.ctl, L.ckq, L.cvq, L.cmeta);
        fill_slot(L.hm, L.hkq, L.hvq, L.hmeta);
        fill_slot(L.hm16, L.hkq, L.hvq, L.hmeta16);
        fill_slot(L.pack, L.packbuf, nullptr, nullptr);
        fill_slot(L.vec, L.vkq, L.vvq, L.vmeta);
    }
    cuda_check(cudaDeviceSynchronize(), "transpose sync");

    float *part = nullptr, *out = nullptr, *mask = nullptr, *qscales = nullptr;
    unsigned char* qbuf = nullptr;
    int *dpos = nullptr, *dnkv = nullptr;
    unsigned int* xout = nullptr;
    cuda_check(cudaMalloc(&part, (size_t)4096 * nh * (hd + 2) * 4), "malloc part");
    cuda_check(cudaMalloc(&out, nh * hd * 4), "malloc out");
    cuda_check(cudaMalloc(&mask, (n_kv + 1) * 4), "malloc mask");
    cudaMemset(mask, 0, (n_kv + 1) * 4);
    cuda_check(cudaMalloc(&qbuf, 8192), "malloc qbuf");
    qscales = (float*)(qbuf + 7168);
    cuda_check(cudaMalloc(&dpos, 4), "malloc pos");
    cuda_check(cudaMalloc(&dnkv, 4), "malloc nkv");
    cuda_check(cudaMalloc(&xout, 4096 * 8 * 4 * 4), "malloc xout");
    cudaMemset(xout, 0, 4096 * 8 * 4 * 4);
    const int hpos = n_kv - 1;
    cudaMemcpy(dpos, &hpos, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dnkv, &n_kv, 4, cudaMemcpyHostToDevice);

    const int max_chunks = (n_kv + chunk_len - 1) / chunk_len;
    const dim3 gp_gqa((unsigned)max_chunks, hkv);

    auto quant_layer = [&](int li) {
        inkwell_kvq_quantq<<<dim3((unsigned)nh), 32>>>(layers[li].q, q_s2, nh, qbuf, 2);
    };
    auto launch_ctl = [&](int li) { launch_ctl_row<8>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hm64 = [&](int li) { launch_hm<4, 12>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hm128 = [&](int li) { launch_hm<8, 12>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hm256 = [&](int li) { launch_hm<16, 12>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hmm16 = [&](int li) { launch_hm<8, 16>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hp = [&](int li) { launch_pack(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hv = [&](int li) { launch_vec2(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_hmd1 = [&](int li) { launch_hm<8, 12, false>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_mem_tm = [&](int li) {
        ink_f4e_memrows<0><<<gp_gqa, dim3(128)>>>(layers[li].ctl, dnkv, chunk_len, xout);
    };
    auto launch_mem_hm = [&](int li) {
        ink_f4e_memrows<1><<<gp_gqa, dim3(128)>>>(layers[li].hm, dnkv, chunk_len, xout);
    };
    auto launch_mem_vec = [&](int li) {
        ink_f4e_memrows<2><<<gp_gqa, dim3(128)>>>(layers[li].vec, dnkv, chunk_len, xout);
    };
    auto launch_seq = [&](int li) {
        const int threads = 256;
        const int blocks = 1024;
        ink_f4e_seqread<<<blocks, threads>>>((const uint4*)layers[li].ckq, kbytes / 16, xout);
        ink_f4e_seqread<<<blocks, threads>>>((const uint4*)layers[li].cvq, vbytes / 16, xout);
        ink_f4e_seqread<<<blocks, threads>>>((const uint4*)layers[li].cmeta, mbytes / 16, xout);
    };
    auto reduce = [&]() {
        inkwell_attn_decode_reduce_dyn<<<dim3((unsigned)nh), 128>>>(part, mask, out, nh, dnkv, hd, chunk_len);
    };

    constexpr int NV = 8;
    std::vector<std::function<void(int)>> ilaunch = {
        launch_ctl, launch_hm64, launch_hm128, launch_hm256, launch_hmm16, launch_hp, launch_hv, launch_hmd1
    };
    const char* inames[NV] = {"ctl_row128", "hm_row64", "hm_row128", "hm_row256", "hm_m16", "hm_pack", "hm_vec2", "hm_dp4a1"};
    double max_abs[NV] = {}, sumsq[NV] = {}, refsq = 0.0;
    std::vector<float> hnew(nh * hd);
    std::vector<std::vector<float>> refs(F4E_NLAYER);
    for (int li = 0; li < F4E_NLAYER; ++li) {
        quant_layer(li);
        launch_ctl(li);
        reduce();
        cuda_check(cudaDeviceSynchronize(), "control correctness sync");
        refs[li].resize(nh * hd);
        cudaMemcpy(refs[li].data(), out, nh * hd * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < nh * hd; ++i) refsq += (double)refs[li][i] * refs[li][i];
        for (int v = 1; v < NV; ++v) {
            ilaunch[v](li);
            reduce();
            cuda_check(cudaDeviceSynchronize(), "candidate correctness sync");
            cudaMemcpy(hnew.data(), out, nh * hd * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < nh * hd; ++i) {
                const double df = (double)hnew[i] - refs[li][i];
                max_abs[v] = std::max(max_abs[v], fabs(df));
                sumsq[v] += df * df;
            }
        }
    }
    cuda_check(cudaGetLastError(), "correctness launch");

    constexpr int NB = 12;
    std::vector<std::function<void()>> bench = {
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_ctl(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hm64(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hm128(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hm256(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hmm16(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hp(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hv(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_hmd1(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_mem_tm(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_mem_hm(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_mem_vec(li); },
        [&] { for (int li = 0; li < F4E_NLAYER; ++li) launch_seq(li); },
    };
    const char* bnames[NB] = {"ctl_row128", "hm_row64", "hm_row128", "hm_row256", "hm_m16", "hm_pack", "hm_vec2", "hm_dp4a1", "mem_tm", "mem_hm", "mem_vec2", "mem_seqread"};
    for (auto& fn : bench) fn();
    cudaDeviceSynchronize();
    float bms[NB] = {};
    for (int round = 0; round < 2; ++round) {
        for (int v = 0; v < NB; ++v) {
            const float t = time_kernel(bench, v, reps);
            bms[v] = bms[v] == 0.f ? t : std::min(bms[v], t);
        }
    }

    printf("mode=%s chunk=%d layers=%d nkv=%d\n", mode, chunk_len, F4E_NLAYER, n_kv);
    for (int v = 0; v < NV; ++v) {
        const double rel = v ? sqrt(sumsq[v] / std::max(1e-30, refsq)) : 0.0;
        printf("%s ms=%.3f maxabs=%.3g relL2=%.3g delta=%+.3f\n", inames[v], bms[v], max_abs[v], rel, bms[v] - bms[0]);
    }
    for (int v = NV; v < NB; ++v) {
        printf("%s ms=%.3f delta=%+.3f\n", bnames[v], bms[v], bms[v] - bms[0]);
    }
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const double total_bytes = (double)F4E_NLAYER * (double)(kbytes + vbytes + mbytes);
    printf("bytes=%.0f attention_bw_ctl=%.1fGBs memhm_bw=%.1fGBs seq_bw=%.1fGBs\n",
           total_bytes, total_bytes / gib / (bms[0] / 1000.0), total_bytes / gib / (bms[9] / 1000.0),
           total_bytes / gib / (bms[11] / 1000.0));
    bool ok = max_abs[2] == 0.0 && max_abs[4] == 0.0 && max_abs[5] == 0.0 && max_abs[6] == 0.0;
    return ok ? 0 : 1;
}
