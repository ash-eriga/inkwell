#include "kernels.cu"
extern "C" const unsigned char ink_ch2nopless_cubin_start[1] = {0};
extern "C" const unsigned char ink_ch2nopless_cubin_end[1] = {0};

enum InkF3eQMode { F3E_QREG = 0, F3E_QSM = 1, F3E_QSMASM = 2, F3E_QLDG = 3 };
enum InkF3eVMode { F3E_VBASE = 0, F3E_VU8 = 1 };
enum InkF3ePFMode { F3E_PF1 = 1, F3E_PF2 = 2 };

__device__ __forceinline__ uint2 ink_f3e_lds_u2(const uint2* p) {
    uint2 v;
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
    return v;
}

__device__ __forceinline__ float ink_f3e_u8f(unsigned int x) {
    float f;
    asm volatile("cvt.rn.f32.u8 %0, %1;" : "=f"(f) : "r"(x));
    return f;
}

template<int QMODE>
__device__ __forceinline__ uint2 ink_f3e_qget(const uint2* qa, const unsigned char* qpacks,
                                              int hq0, int d0, int g, int sl, bool low) {
    if constexpr (QMODE == F3E_QREG) {
        return qa[g];
    } else if constexpr (QMODE == F3E_QSM) {
        return qa[g * 16 + sl];
    } else if constexpr (QMODE == F3E_QSMASM) {
        return ink_f3e_lds_u2(qa + g * 16 + sl);
    } else {
        return __ldg((const uint2*)(qpacks + (low ? 3072 : 0) + (hq0 + g) * 128 + d0));
    }
}

template<int FMT, int KV>
__device__ __forceinline__ InkF3dRawKV ink_f3e_load_kv(const InkKvqPtr8<FMT>& p, int sl) {
    InkF3dRawKV r{};
    if constexpr (KV == 0) return ink_f3d_load_kv8<FMT>(p, sl);
    else {
        r.kb = __ldg((const uint2*)p.kp);
        if constexpr (KV == 1) r.vb = __ldg((const unsigned int*)p.vp);
        return r;
    }
}

template<int FMT, int NST, bool CONST_BIAS, int QMODE, int VMODE, int PF, int KV = 0>
__device__ __forceinline__ void ink_f3e_rows(
        InkKvqPtr8<FMT> qp, const int j0, const int j1, const int stream, const int sl,
        const float* __restrict__ bias, const float* __restrict__ mask,
        const int hq0, const int b_s2, const int pos, const int n_kv,
        const int bias_mode, const int swa, const int rel_extent, const int mask_zero,
        const uint2* __restrict__ qpack, const uint2* __restrict__ qpackl, const float* __restrict__ qscale_s,
        const unsigned char* __restrict__ qpacks, const int d0,
        float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    InkF3dRawKV qcur{}, qnext{};
    const int first = j0 + stream;
    if constexpr (PF == F3E_PF2) {
        if (first < j1) {
            qcur = ink_f3e_load_kv<FMT, KV>(qp, sl);
        }
    }
    for (int jb = j0; jb < j1; jb += NST) {
        const int j = jb + stream;
        const bool valid = j < j1;
        if constexpr (PF == F3E_PF2) {
            const int jn = jb + NST + stream;
            ink_f3d_step_kv8<FMT, NST>(qp);
            if (jn < j1) {
                qnext = ink_f3e_load_kv<FMT, KV>(qp, sl);
            }
        } else {
            if (valid) {
                qcur = ink_f3e_load_kv<FMT, KV>(qp, sl);
            }
            ink_f3d_step_kv8<FMT, NST>(qp);
        }

        unsigned int kscw = qcur.kscw;
        unsigned int vwv = qcur.vw;
        unsigned int vbv = qcur.vb;
        if constexpr (KV != 0) {
            kscw = 0;
            vwv = 0;
            if constexpr (KV == 2) vbv = 0;
            if (valid) {
                constexpr unsigned int MSTEP = (unsigned int)NST * ((FMT == INK_KVQ_K8V4G32) ? 192u : 96u);
                constexpr unsigned int VSTEP = (unsigned int)NST * 512u;
                const unsigned char* mpc = qp.mp - MSTEP;
                const int kgrp = sl >> 3;
                const int vgrp = (FMT == INK_KVQ_K8V4G32) ? (sl >> 2) : kgrp;
                kscw = __ldg((const unsigned short*)mpc + kgrp);
                vwv = __ldg((const unsigned int*)(mpc + 4) + vgrp);
                if constexpr (KV == 2) vbv = __ldg((const unsigned int*)(qp.vp - VSTEP));
            }
        }
        const float kscv = ink_kvq_half_as_float(kscw);
        int isum[3];
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            const uint2 qpg = ink_f3e_qget<QMODE>(qpack, qpacks, hq0, d0, g, sl, false);
            const uint2 qplg = ink_f3e_qget<QMODE>(qpackl, qpacks, hq0, d0, g, sl, true);
            const int hx = __dp4a((int)qpg.x, (int)qcur.kb.x, 0);
            const int hy = __dp4a((int)qpg.y, (int)qcur.kb.y, 0);
            int ls = __dp4a((int)qplg.x, (int)qcur.kb.x, 0);
            ls = __dp4a((int)qplg.y, (int)qcur.kb.y, ls);
            isum[g] = ((hx + hy) << 7) + ls;
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
            for (int g = 0; g < 3; ++g) {
                sc[g] += ink_attn_bias_value(bias, j, hq0 + g, b_s2, pos, n_kv, bias_mode, swa, rel_extent) +
                         (mask_zero ? 0.f : mask[j]);
            }
        }

        const float vsc = ink_kvq_half_as_float(vwv & 0xffffu);
        const float vmn = ink_kvq_half_as_float(vwv >> 16);
        const unsigned int lo4 = vbv & 0x0f0f0f0fu;
        const unsigned int hi4 = (vbv >> 4) & 0x0f0f0f0fu;
        if constexpr (VMODE == F3E_VBASE) {
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
        } else if (valid) {
            float cw[3], ww[3];
            #pragma unroll
            for (int g = 0; g < 3; ++g) {
                const float mold = mst[g];
                if (sc[g] > mold) {
                    const float c = __expf(mold - sc[g]);
                    mst[g] = sc[g];
                    lst[g] = fmaf(lst[g], c, 1.f);
                    bacc[g] = fmaf(bacc[g], c, vmn);
                    cw[g] = c;
                    ww[g] = vsc;
                } else {
                    const float p = __expf(sc[g] - mold);
                    lst[g] += p;
                    bacc[g] = fmaf(p, vmn, bacc[g]);
                    cw[g] = 1.f;
                    ww[g] = p * vsc;
                }
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float q0 = ink_f3e_u8f(lo4 >> (8 * k));
                const float q1 = ink_f3e_u8f(hi4 >> (8 * k));
                #pragma unroll
                for (int g = 0; g < 3; ++g) {
                    acc[g][2 * k] = fmaf(ww[g], q0, acc[g][2 * k] * cw[g]);
                    acc[g][2 * k + 1] = fmaf(ww[g], q1, acc[g][2 * k + 1] * cw[g]);
                }
            }
        }
        if constexpr (PF == F3E_PF2) {
            qcur = qnext;
        }
    }
}

template<int FMT, int NST, int QMODE, int VMODE, int PF, int LB = 0, int KV = 0>
__global__ void __launch_bounds__(NST * 16, LB ? 1024 / (NST * 16) : 1) inkwell_attn_kvq_gqa3_v2_row_f3e(InkKvQSlot slot, const int* __restrict__ posp,
        const float* __restrict__ Q, const float* __restrict__ bias,
        const float* __restrict__ mask, float* __restrict__ part,
        int nh, const int* __restrict__ d_nkv, int q_s2, int b_s2, float scale,
        int chunk_len, int mask_zero, int bias_mode, int swa, int rel_extent,
        const unsigned char* __restrict__ qpacks, const float* __restrict__ qscales) {
    constexpr int NW = NST / 2;
    extern __shared__ float sw[];
    float* sacc = sw;
    float* smm = sacc + NW * 3 * 128;
    float* sml = smm + NW * 3;
    float* smb = sml + NW * 3;

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
    const uint2* qpp = qpack;
    const uint2* qplp = qpackl;
    const float* qsp = qscale_s;
    if constexpr (QMODE == F3E_QSM || QMODE == F3E_QSMASM) {
        __shared__ uint2 sh_qp[3][16], sh_qpl[3][16];
        __shared__ float sh_qs[3];
        if (tid < 48) {
            const int g = tid >> 4;
            const int ss = tid & 15;
            const int h = hq0 + g;
            const int dd = ss * 8;
            sh_qp[g][ss] = __ldg((const uint2*)(qpacks + h * 128 + dd));
            sh_qpl[g][ss] = __ldg((const uint2*)(qpacks + 3072 + h * 128 + dd));
        }
        if (tid < 3) sh_qs[tid] = __ldg(qscales + hq0 + tid) * scale * 0.0078125f;
        __syncthreads();
        qpp = &sh_qp[0][0];
        qplp = &sh_qpl[0][0];
        qsp = sh_qs;
    } else {
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            const int h = hq0 + g;
            if constexpr (QMODE == F3E_QREG) {
                qpack[g] = __ldg((const uint2*)(qpacks + h * 128 + d0));
                qpackl[g] = __ldg((const uint2*)(qpacks + 3072 + h * 128 + d0));
            }
            qscale_s[g] = __ldg(qscales + h) * scale * 0.0078125f;
        }
    }

    float acc[3][8] = {};
    float bacc[3] = {}, mst[3], lst[3] = {};
    #pragma unroll
    for (int g = 0; g < 3; ++g) mst[g] = -1e30f;

    InkKvqPtr8<FMT> qp = ink_kvq_ptr8<FMT>(slot, j0 + stream, hk, sl);
    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;
    if (const_bias) {
        ink_f3e_rows<FMT, NST, true, QMODE, VMODE, PF, KV>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                                     bias_mode, swa, rel_extent, mask_zero, qpp, qplp, qsp, qpacks, d0,
                                     acc, bacc, mst, lst);
    } else {
        ink_f3e_rows<FMT, NST, false, QMODE, VMODE, PF, KV>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                                      bias_mode, swa, rel_extent, mask_zero, qpp, qplp, qsp, qpacks, d0,
                                      acc, bacc, mst, lst);
    }

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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>



#define F3D_SRC_N 4096
#define F3D_NLAYER 8

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        exit(2);
    }
}

__global__ void ink_f3d_quant_from_dump(const __half* __restrict__ kd, const __half* __restrict__ vd,
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
    InkKvQSlot slot{};
    float* q = nullptr;
    float* bias = nullptr;
    unsigned char* kq = nullptr;
    unsigned char* vq = nullptr;
    unsigned char* meta = nullptr;
};

template<int NST, int QMODE, int VMODE, int PF, int LB = 0, int KV = 0>
static inline void launch_f3e(const DevLayer& L, const dim3& gp, int chunk_len,
                              const int* dpos, const float* mask, float* part, const int* dnkv,
                              int nh, int q_s2, int b_s2, float scale,
                              const unsigned char* qbuf, const float* qscales) {
    constexpr int NW = NST / 2;
    const size_t sm = (size_t)(NW * 3 * 128 + NW * 3 * 2 + NW * 3 * 16) * 4;
    inkwell_attn_kvq_gqa3_v2_row_f3e<INK_KVQ_K8V4, NST, QMODE, VMODE, PF, LB, KV><<<gp, dim3(NST * 16), sm>>>(
        L.slot, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

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
static inline void launch_rowv2(const DevLayer& L, const dim3& gp, int chunk_len,
                                const int* dpos, const float* mask, float* part, const int* dnkv,
                                int nh, int q_s2, int b_s2, float scale,
                                const unsigned char* qbuf, const float* qscales) {
    constexpr int NW = NST / 2;
    const size_t sm = (size_t)(NW * 3 * 128 + NW * 3 * 2 + NW * 3 * 16) * 4;
    inkwell_attn_kvq_gqa3_v2_row<INK_KVQ_K8V4, NST><<<gp, dim3(NST * 16), sm>>>(
        L.slot, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    const int n_kv = argc > 3 ? atoi(argv[3]) : 65536, nh = 24, hkv = 8, hd = 128;
    const int q_s2 = 128, b_s2 = 1025;
    const float scale = 1.0f / sqrtf(128.f);
    const char* mode = argc > 1 ? argv[1] : "real";
    const int chunk_len = argc > 2 ? atoi(argv[2]) : 256;
    const int reps = getenv("F3D_REPS") ? atoi(getenv("F3D_REPS")) : 7;
    const bool uniform = strcmp(mode, "uniform") == 0;
    const std::string dumproot = getenv("F3D_DUMP") ? getenv("F3D_DUMP") :
        "/home/ash/.cache/inkwell-intkv/attn4k-v26-nocache/4096";
    const char* names[F3D_NLAYER] = {"l5p0", "l11p0", "l11p1", "l11p2", "l17p0", "l17p1", "l17p2", "l23p0"};
    if (chunk_len < 32 || chunk_len > 1024) {
        fprintf(stderr, "chunk_len must be in [32,1024]\n");
        return 2;
    }

    std::vector<DevLayer> layers(F3D_NLAYER);
    const size_t kbytes = (size_t)n_kv * 1024, vbytes = (size_t)n_kv * 512, mbytes = (size_t)n_kv * 96;
    const float tau = 1.f + 0.1f * logf((float)n_kv / 8192.f);
    for (int li = 0; li < F3D_NLAYER; ++li) {
        DevLayer& L = layers[li];
        cuda_check(cudaMalloc(&L.kq, kbytes), "malloc kq");
        cuda_check(cudaMalloc(&L.vq, vbytes), "malloc vq");
        cuda_check(cudaMalloc(&L.meta, mbytes), "malloc meta");
        cuda_check(cudaMalloc(&L.q, nh * hd * 4), "malloc q");
        cuda_check(cudaMalloc(&L.bias, nh * b_s2 * 4), "malloc bias");
        if (uniform) {
            cudaMemset(L.kq, 0, kbytes);
            cudaMemset(L.vq, 0, vbytes);
            cudaMemset(L.meta, 0, mbytes);
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
            s.kq = L.kq; s.vq = L.vq; s.meta = L.meta;
            s.cap = n_kv; s.hp_cap = 0; s.swa = 1024; s.sinks = 0; s.local = 0; s.is_swa = 0;
            s.fmt = INK_KVQ_K8V4; s.krow = 1024; s.vrow = 512; s.mrow = 96;
            dim3 qg((unsigned)((n_kv + 7) / 8), hkv);
            ink_f3d_quant_from_dump<<<qg, 128>>>(dk, dv, s, F3D_SRC_N);
            cuda_check(cudaGetLastError(), "quant launch");
            cuda_check(cudaDeviceSynchronize(), "quant sync");
            cudaFree(dk);
            cudaFree(dv);

            std::vector<float> hq((const float*)qb.data(), (const float*)qb.data() + nh * hd);
            for (float& x : hq) x *= tau;
            cudaMemcpy(L.q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
            const float* bd = (const float*)bb.data();
            std::vector<float> table(nh * b_s2, 0.f);
            const int pos4 = F3D_SRC_N - 1;
            for (int h = 0; h < nh; ++h) {
                for (int rp = 0; rp < b_s2; ++rp) {
                    const int j = pos4 - rp;
                    table[h * b_s2 + rp] = bd[h * F3D_SRC_N + j] * tau;
                }
            }
            cudaMemcpy(L.bias, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
        }
        L.slot.kq = L.kq;
        L.slot.vq = L.vq;
        L.slot.meta = L.meta;
        L.slot.cap = n_kv;
        L.slot.hp_cap = 0;
        L.slot.swa = 1024;
        L.slot.fmt = INK_KVQ_K8V4;
        L.slot.krow = 1024;
        L.slot.vrow = 512;
        L.slot.mrow = 96;
    }

    float *part = nullptr, *out = nullptr, *mask = nullptr, *qscales = nullptr;
    unsigned char* qbuf = nullptr;
    int *dpos = nullptr, *dnkv = nullptr;
    cuda_check(cudaMalloc(&part, (size_t)4096 * nh * (hd + 2) * 4), "malloc part");
    cuda_check(cudaMalloc(&out, nh * hd * 4), "malloc out");
    cuda_check(cudaMalloc(&mask, (n_kv + 1) * 4), "malloc mask");
    cudaMemset(mask, 0, (n_kv + 1) * 4);
    cuda_check(cudaMalloc(&qbuf, 8192), "malloc qbuf");
    qscales = (float*)(qbuf + 7168);
    cuda_check(cudaMalloc(&dpos, 4), "malloc pos");
    cuda_check(cudaMalloc(&dnkv, 4), "malloc nkv");
    const int hpos = n_kv - 1;
    cudaMemcpy(dpos, &hpos, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dnkv, &n_kv, 4, cudaMemcpyHostToDevice);

    const int max_chunks = (n_kv + chunk_len - 1) / chunk_len;
    const dim3 gp_split((unsigned)(max_chunks * 3), hkv);
    const dim3 gp_gqa((unsigned)max_chunks, hkv);
    const size_t smw = (3 * 128 + 2 * 3) * 4;

    auto quant_layer = [&](int li) {
        inkwell_kvq_quantq<<<dim3((unsigned)nh), 32>>>(layers[li].q, q_s2, nh, qbuf, 2);
    };
    auto launch_base = [&](int li) {
        inkwell_attn_kvq_splitw<INK_KVQ_K8V4, false, 2><<<gp_split, dim3(64), smw>>>(
            layers[li].slot, dpos, layers[li].q, layers[li].bias, mask, part,
            nh, dnkv, q_s2, b_s2, scale, chunk_len, 1, 3, 1,
            0, 1024, 1024, qbuf, qscales);
    };
    auto launch_a128 = [&](int li) { launch_rowv2<8>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_vu8 = [&](int li) { launch_f3e<8, F3E_QREG, F3E_VU8, F3E_PF2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_sm = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_smasm = [&](int li) { launch_f3e<8, F3E_QSMASM, F3E_VU8, F3E_PF2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_ldg = [&](int li) { launch_f3e<8, F3E_QLDG, F3E_VU8, F3E_PF2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_sm1 = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF1>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_ldg1 = [&](int li) { launch_f3e<8, F3E_QLDG, F3E_VU8, F3E_PF1>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_sm64 = [&](int li) { launch_f3e<4, F3E_QSM, F3E_VU8, F3E_PF2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_sm256 = [&](int li) { launch_f3e<16, F3E_QSM, F3E_VU8, F3E_PF2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_smlb = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF2, 1>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_sm1lb = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF1, 1>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_smk = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF2, 0, 1>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_smkk = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF2, 0, 2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_sm1kk = [&](int li) { launch_f3e<8, F3E_QSM, F3E_VU8, F3E_PF1, 0, 2>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto reduce = [&]() {
        inkwell_attn_decode_reduce_dyn<<<dim3((unsigned)nh), 128>>>(part, mask, out, nh, dnkv, hd, chunk_len);
    };

    std::vector<std::function<void(int)>> ilaunch = {launch_base, launch_a128, launch_vu8, launch_sm, launch_smasm, launch_ldg, launch_sm1, launch_ldg1, launch_sm64, launch_sm256, launch_smlb, launch_sm1lb, launch_smk, launch_smkk, launch_sm1kk};
    constexpr int F3E_NV = 15;
    const char* inames[F3E_NV] = {"splitw", "row128", "vu8", "sm", "smasm", "ldg", "sm1", "ldg1", "sm64", "sm256", "smlb", "sm1lb", "smk", "smkk", "sm1kk"};
    double max_abs[F3E_NV] = {}, sumsq[F3E_NV] = {}, refsq = 0.0;
    std::vector<float> hnew(nh * hd);
    std::vector<std::vector<float>> refs(F3D_NLAYER);
    for (int li = 0; li < F3D_NLAYER; ++li) {
        quant_layer(li);
        launch_base(li);
        reduce();
        cuda_check(cudaDeviceSynchronize(), "baseline correctness sync");
        refs[li].resize(nh * hd);
        cudaMemcpy(refs[li].data(), out, nh * hd * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < nh * hd; ++i) refsq += (double)refs[li][i] * refs[li][i];
        for (int v = 1; v < F3E_NV; ++v) {
            ilaunch[v](li);
            reduce();
            cuda_check(cudaDeviceSynchronize(), "candidate correctness sync");
            cudaMemcpy(hnew.data(), out, nh * hd * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < nh * hd; ++i) {
                const double df = (double)hnew[i] - refs[li][i];
                max_abs[v] = std::max(max_abs[v], fabs(df));
                sumsq[v] += df * df;
            }
            if (li == 0 && v == 1 && getenv("F3D_DEBUG")) {
                int mi = 0;
                for (int i = 1; i < nh * hd; ++i) if (fabsf(hnew[i] - refs[li][i]) > fabsf(hnew[mi] - refs[li][mi])) mi = i;
                printf("debug maxi=%d h=%d d=%d ref=%.8g new=%.8g\n", mi, mi / hd, mi % hd, refs[li][mi], hnew[mi]);
                for (int h = 0; h < 3; ++h) {
                    printf("debug h%d ref:", h);
                    for (int d = 0; d < 8; ++d) printf(" %.6g", refs[li][h * hd + d]);
                    printf(" new:");
                    for (int d = 0; d < 8; ++d) printf(" %.6g", hnew[h * hd + d]);
                    printf("\n");
                }
            }
        }
    }
    cuda_check(cudaGetLastError(), "correctness launch");

    std::vector<std::function<void()>> bench = {
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_base(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_a128(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_vu8(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_sm(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_smasm(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_ldg(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_sm1(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_ldg1(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_sm64(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_sm256(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_smlb(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_sm1lb(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_smk(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_smkk(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_sm1kk(li); },
    };
    for (auto& fn : bench) fn();
    cudaDeviceSynchronize();
    float bms[F3E_NV] = {};
    for (int round = 0; round < 2; ++round) {
        for (int v = 0; v < F3E_NV; ++v) {
            const float t = time_kernel(bench, v, reps);
            bms[v] = bms[v] == 0.f ? t : std::min(bms[v], t);
        }
    }

    printf("mode=%s chunk=%d layers=%d nkv=%d\n", mode, chunk_len, F3D_NLAYER, n_kv);
    for (int v = 0; v < F3E_NV; ++v) {
        const double rel = v ? sqrt(sumsq[v] / std::max(1e-30, refsq)) : 0.0;
        printf("%s ms=%.3f maxabs=%.3g relL2=%.3g speedup=%.3fx\n", inames[v], bms[v], max_abs[v], rel, bms[0] / bms[v]);
    }
    bool ok = true;
    for (int v = 1; v < F3E_NV; ++v) ok = ok && max_abs[v] <= 1e-5;
    return ok ? 0 : 1;
}
