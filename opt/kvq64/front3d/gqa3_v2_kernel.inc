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

    InkKvqPtr8<FMT> qp = ink_kvq_ptr8<FMT>(slot, j0 + stream, hk, sl);
    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;
    if (const_bias) {
        ink_f3d_rows<FMT, NST, true>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                                     bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s,
                                     acc, bacc, mst, lst);
    } else {
        ink_f3d_rows<FMT, NST, false>(qp, j0, j1, stream, sl, bias, mask, hq0, b_s2, pos, n_kv,
                                      bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s,
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

struct InkF3dSRow {
    uint2 kb[16];
    unsigned int vb[16];
    unsigned short ksc[2];
    unsigned int vw[2];
};

struct InkF3dLdRow {
    uint2 kb;
    unsigned int vb, kscw, vw;
};

template<int FMT>
__device__ __forceinline__ InkF3dLdRow ink_f3d_ld_coop(const InkKvQSlot& slot, int row, int hk, int sl) {
    InkF3dLdRow r{};
    const unsigned char* krow = slot.kq + (size_t)row * slot.krow + hk * 128 + sl * 8;
    const unsigned char* vrow = slot.vq + (size_t)row * slot.vrow + hk * 64 + sl * 4;
    r.kb = __ldg((const uint2*)krow);
    r.vb = __ldg((const unsigned int*)vrow);
    if (sl == 0 || sl == 8) {
        const int grp = sl >> 3;
        const int vgrp = (FMT == INK_KVQ_K8V4G32) ? 0 : grp;
        const unsigned char* mp = slot.meta + (size_t)row * slot.mrow + hk * ((FMT == INK_KVQ_K8V4G32) ? 24 : 12);
        r.kscw = __ldg((const unsigned short*)mp + grp);
        r.vw = __ldg((const unsigned int*)(mp + 4) + vgrp);
    }
    return r;
}

template<int FMT>
__device__ __forceinline__ void ink_f3d_store_coop(InkF3dSRow& sr, const InkF3dLdRow& r, int sl) {
    sr.kb[sl] = r.kb;
    sr.vb[sl] = r.vb;
    if (sl == 0 || sl == 8) {
        sr.ksc[sl >> 3] = (unsigned short)r.kscw;
        sr.vw[sl >> 3] = r.vw;
    }
}

template<int FMT, bool CONST_BIAS>
__device__ __forceinline__ void ink_f3d_b_row(const InkF3dSRow& sr, int j, int sl,
        const float* __restrict__ bias, const float* __restrict__ mask,
        int h, int b_s2, int pos, int n_kv, int bias_mode, int swa, int rel_extent, int mask_zero,
        uint2 qpack, uint2 qpackl, float qscale_s,
        float acc[8], float& bacc, float& mst, float& lst) {
    const float kscv = ink_kvq_half_as_float(sr.ksc[sl >> 3]);
    int isum;
    {
        const int hx = __dp4a((int)qpack.x, (int)sr.kb[sl].x, 0);
        const int hy = __dp4a((int)qpack.y, (int)sr.kb[sl].y, 0);
        int ls = __dp4a((int)qpackl.x, (int)sr.kb[sl].x, 0);
        ls = __dp4a((int)qpackl.y, (int)sr.kb[sl].y, ls);
        isum = ((hx + hy) << 7) + ls;
    }
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) isum += __shfl_xor_sync(0xffffffffu, isum, o);
    float sc = (float)isum * kscv;
    sc += __shfl_xor_sync(0xffffffffu, sc, 8);
    sc *= qscale_s;
    if constexpr (!CONST_BIAS) {
        sc += ink_attn_bias_value(bias, j, h, b_s2, pos, n_kv, bias_mode, swa, rel_extent) +
              (mask_zero ? 0.f : mask[j]);
    }

    const unsigned int vwd = sr.vw[sl >> 3];
    const float vsc = ink_kvq_half_as_float(vwd & 0xffffu);
    const float vmn = ink_kvq_half_as_float(vwd >> 16);
    const unsigned int lo4 = sr.vb[sl] & 0x0f0f0f0fu;
    const unsigned int hi4 = (sr.vb[sl] >> 4) & 0x0f0f0f0fu;
    float qf[8];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        qf[2 * k] = (float)((lo4 >> (8 * k)) & 0xffu);
        qf[2 * k + 1] = (float)((hi4 >> (8 * k)) & 0xffu);
    }
    const float mold = mst;
    if (sc > mold) {
        const float c = __expf(mold - sc);
        mst = sc;
        lst = fmaf(lst, c, 1.f);
        bacc = fmaf(bacc, c, vmn);
        #pragma unroll
        for (int f = 0; f < 8; ++f) acc[f] = fmaf(vsc, qf[f], acc[f] * c);
    } else {
        const float p = __expf(sc - mold);
        lst += p;
        bacc = fmaf(p, vmn, bacc);
        const float wv = p * vsc;
        #pragma unroll
        for (int f = 0; f < 8; ++f) acc[f] = fmaf(wv, qf[f], acc[f]);
    }
}

template<int FMT>
__global__ void inkwell_attn_kvq_gqa3_v2_smem3(InkKvQSlot slot, const int* __restrict__ posp,
        const float* __restrict__ Q, const float* __restrict__ bias,
        const float* __restrict__ mask, float* __restrict__ part,
        int nh, const int* __restrict__ d_nkv, int q_s2, int b_s2, float scale,
        int chunk_len, int mask_zero, int bias_mode, int swa, int rel_extent,
        const unsigned char* __restrict__ qpacks, const float* __restrict__ qscales) {
    extern __shared__ float sw[];
    InkF3dSRow (*rows)[6] = (InkF3dSRow (*)[6])sw;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int sl = lane & 15;
    const int rw = lane >> 4;
    const int g = tid >> 5;
    const int lstream = tid >> 4;
    const int chunk = blockIdx.x;
    const int hk = blockIdx.y;
    const int ratio = nh / 8;
    if (ratio != 3 || g >= 3) return;
    const int h = hk * ratio + g;
    const int hq0 = hk * ratio;
    const int n_kv = min(d_nkv[0], slot.cap);
    const int j0 = chunk * chunk_len;
    if (j0 >= n_kv) return;
    const int j1 = min(n_kv, j0 + chunk_len);
    const int d0 = sl * 8;
    const int pos = posp[0];

    const uint2 qpack = __ldg((const uint2*)(qpacks + h * 128 + d0));
    const uint2 qpackl = __ldg((const uint2*)(qpacks + 3072 + h * 128 + d0));
    const float qscale_s = __ldg(qscales + h) * scale * 0.0078125f;
    float acc[8] = {};
    float bacc = 0.f, mst = -1e30f, lst = 0.f;
    const bool const_bias = mask_zero && bias_mode == 0 && (pos - (j1 - 1)) >= rel_extent;

    int row0 = j0 + lstream;
    if (row0 < j1) {
        InkF3dLdRow lr = ink_f3d_ld_coop<FMT>(slot, row0, hk, sl);
        ink_f3d_store_coop<FMT>(rows[0][lstream], lr, sl);
    }
    __syncthreads();

    int cur = 0;
    for (int base = j0; base < j1; base += 6) {
        const int nb = base + 6;
        const int nrow = nb + lstream;
        InkF3dLdRow nl{};
        bool nvalid = nrow < j1;
        if (nvalid) nl = ink_f3d_ld_coop<FMT>(slot, nrow, hk, sl);

        #pragma unroll
        for (int rr = 0; rr < 3; ++rr) {
            const int sloti = rr * 2 + rw;
            const int j = base + sloti;
            if (j < j1) {
                if (const_bias) {
                    ink_f3d_b_row<FMT, true>(rows[cur][sloti], j, sl, bias, mask, h, b_s2, pos, n_kv,
                                             bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s,
                                             acc, bacc, mst, lst);
                } else {
                    ink_f3d_b_row<FMT, false>(rows[cur][sloti], j, sl, bias, mask, h, b_s2, pos, n_kv,
                                              bias_mode, swa, rel_extent, mask_zero, qpack, qpackl, qscale_s,
                                              acc, bacc, mst, lst);
                }
            }
        }
        if (!nvalid && nb >= j1) break;
        if (nvalid) ink_f3d_store_coop<FMT>(rows[cur ^ 1][lstream], nl, sl);
        __syncthreads();
        cur ^= 1;
    }

    {
        const float mo = __shfl_xor_sync(0xffffffffu, mst, 16);
        const float lo = __shfl_xor_sync(0xffffffffu, lst, 16);
        const float bo = __shfl_xor_sync(0xffffffffu, bacc, 16);
        const float mx = fmaxf(mst, mo);
        const float wc = __expf(mst - mx);
        const float wo = __expf(mo - mx);
        mst = mx;
        lst = lst * wc + lo * wo;
        bacc = bacc * wc + bo * wo;
        #pragma unroll
        for (int f = 0; f < 8; ++f) {
            const float ao = __shfl_xor_sync(0xffffffffu, acc[f], 16);
            acc[f] = acc[f] * wc + ao * wo;
        }
    }
    if (rw == 0) {
        float* pb = part + ((size_t)chunk * nh + h) * (128 + 2);
        if (const_bias && sl == 0) mst += __ldg(bias + rel_extent + h * b_s2);
        if (sl == 0) { pb[0] = mst; pb[1] = lst; }
        #pragma unroll
        for (int f = 0; f < 8; ++f) pb[2 + d0 + f] = acc[f] + bacc;
    }
}
