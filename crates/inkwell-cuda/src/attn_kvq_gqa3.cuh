struct InkF3dRawKV {
    uint2 kb;
    unsigned int vb, kscw, vw;
};

template<int FMT>
__device__ __forceinline__ InkF3dRawKV ink_f3d_load_kv8(const InkKvqPtr8<FMT>& p, int sl) {
    InkF3dRawKV r;
    const int kgrp = sl >> 3;
    const int vgrp = (FMT == INK_KVQ_K8V4G32) ? (sl >> 2) : kgrp;
    r.kscw = __ldg((const unsigned short*)p.mp + kgrp);
    r.vw = __ldg((const unsigned int*)(p.mp + 4) + vgrp);
    r.kb = __ldg((const uint2*)p.kp);
    r.vb = __ldg((const unsigned int*)p.vp);
    return r;
}

template<int FMT, int NST>
__device__ __forceinline__ void ink_f3d_step_kv8(InkKvqPtr8<FMT>& p) {
    p.kp += (size_t)NST * 1024;
    p.vp += (size_t)NST * 512;
    p.mp += (size_t)NST * ((FMT == INK_KVQ_K8V4G32) ? 192 : 96);
}

template<int FMT, int NST, bool CONST_BIAS>
__device__ __forceinline__ void ink_f3d_rows(
        InkKvqPtr8<FMT> qp, const int j0, const int j1, const int stream, const int sl,
        const float* __restrict__ bias, const float* __restrict__ mask,
        const int hq0, const int b_s2, const int pos, const int n_kv,
        const int bias_mode, const int swa, const int rel_extent, const int mask_zero,
        const uint2 qpack[3], const uint2 qpackl[3], const float qscale_s[3],
        float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    InkF3dRawKV qcur{}, qnext{};
    float bcur[3] = {}, bnext[3] = {}, mcur = 0.f, mnext = 0.f;
    const int first = j0 + stream;
    if (first < j1) {
        qcur = ink_f3d_load_kv8<FMT>(qp, sl);
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
        ink_f3d_step_kv8<FMT, NST>(qp);
        if (jn < j1) {
            qnext = ink_f3d_load_kv8<FMT>(qp, sl);
            if constexpr (!CONST_BIAS) {
                #pragma unroll
                for (int g = 0; g < 3; ++g) bnext[g] = ink_attn_bias_value(bias, jn, hq0 + g, b_s2, pos, n_kv, bias_mode, swa, rel_extent);
                mnext = mask_zero ? 0.f : mask[jn];
            }
        }

        const float kscv = ink_kvq_half_as_float(qcur.kscw);
        int isum[3];
        #pragma unroll
        for (int g = 0; g < 3; ++g) {
            const int hx = __dp4a((int)qpack[g].x, (int)qcur.kb.x, 0);
            const int hy = __dp4a((int)qpack[g].y, (int)qcur.kb.y, 0);
            int ls = __dp4a((int)qpackl[g].x, (int)qcur.kb.x, 0);
            ls = __dp4a((int)qpackl[g].y, (int)qcur.kb.y, ls);
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
        qcur = qnext;
        if constexpr (!CONST_BIAS) {
            #pragma unroll
            for (int g = 0; g < 3; ++g) bcur[g] = bnext[g];
            mcur = mnext;
        }
    }
}

struct InkGqaVecPtr {
    const unsigned char* kp;
    const unsigned char* vp;
    const unsigned char* mp;
};

template<bool LOW, bool CONST_BIAS>
__device__ __forceinline__ void ink_gqa_process_token(
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

struct InkGqaPairRaw {
    InkF3dRawKV a;
    InkF3dRawKV b;
};

__device__ __forceinline__ InkGqaPairRaw ink_gqa_load_pair(const InkGqaVecPtr& p, int sl) {
    InkGqaPairRaw r{};
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
__device__ __forceinline__ void ink_gqa_rows_vec2(
        InkGqaVecPtr qp, const int j0, const int j1, const int stream, const int sl,
        const float* __restrict__ bias, const float* __restrict__ mask,
        const int hq0, const int b_s2, const int pos, const int n_kv,
        const int bias_mode, const int swa, const int rel_extent, const int mask_zero,
        const uint2 qpack[3], const uint2 qpackl[3], const float qscale_s[3],
        float acc[3][8], float bacc[3], float mst[3], float lst[3]) {
    constexpr int NST = 8;
    InkGqaPairRaw qcur{}, qnext{};
    float bcur[2][3] = {}, bnext[2][3] = {}, mcur[2] = {}, mnext[2] = {};
    const int first = j0 + stream;
    if (first < j1) {
        qcur = ink_gqa_load_pair(qp, sl);
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
            qnext = ink_gqa_load_pair(qp, sl);
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
        ink_gqa_process_token<true, CONST_BIAS>(qcur.a, ja < j1, bcur[0], mcur[0], qpack, qpackl, qscale_s,
                                                acc, bacc, mst, lst);
        ink_gqa_process_token<true, CONST_BIAS>(qcur.b, jbv < j1, bcur[1], mcur[1], qpack, qpackl, qscale_s,
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


template<int FMT, int NST>
__global__ void inkwell_attn_kvq_gqa3_v2_row(InkKvQSlot slot, const int* __restrict__ posp,
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

    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;
    if constexpr (FMT == INK_KVQ_K8V4 && NST == 8) {
        const int row = j0 + stream;
        const int tile = row >> 4;
        const int rs = row & 7;
        const int ntiles = slot.cap >> 4;
        InkGqaVecPtr qp;
        qp.kp = slot.kq + ((size_t)hk * ntiles + tile) * 2048 + rs * 256 + sl * 16;
        qp.vp = slot.vq + ((size_t)hk * ntiles + tile) * 1024 + rs * 128 + sl * 8;
        qp.mp = slot.meta + ((size_t)hk * ntiles + tile) * 192 + rs * 24;
        if (const_bias) {
            ink_gqa_rows_vec2<true>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
        } else {
            ink_gqa_rows_vec2<false>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s, acc, bacc, mst, lst);
        }
    } else {
        InkKvqPtr8<FMT> qp = ink_kvq_ptr8<FMT>(slot, j0 + stream, hk, sl);
        if (const_bias) {
            ink_f3d_rows<FMT, NST, true>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                                         bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s,
                                         acc, bacc, mst, lst);
        } else {
            ink_f3d_rows<FMT, NST, false>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                                          bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s,
                                          acc, bacc, mst, lst);
        }
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
