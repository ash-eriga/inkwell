// exl3_gemv_v2.cuh — issue-lean EXL3 trellis GEMVs for sm_61 (Pascal).
//
// Why these exist (measured; opt/ledger.md "custo real da decodificacao"): the
// EXL3 GEMVs are bound by instructions issued (~3.7 of 4 slots/cycle), so the
// lever is fewer instructions per weight:
//   * a lane owns four CONSECUTIVE ring windows (rows 2m,2m+1,2m+8,2m+9 of one
//     column): they are consecutive bit ranges of one ring word pair, so 2 LDS +
//     1 funnel shift + 4 masks decode 4 weights (was 4x LDS,LDS,SHF,LOP3);
//   * sum(x) of the affine decode is hoisted out of the inner loop (was 7 FADD
//     per 8 weights per row); row loads use constant offsets (no address adds);
//   * consumer loops are unrolled at compile time (no runtime-count loop with
//     dynamically indexed accumulators, no integer division in the batch loop);
//   * weight tiles stream through a register queue DEPTH batches deep and a
//     2-stage smem ring with one __syncthreads per batch (a host-pinned expert
//     needs far more bytes in flight than a VRAM-resident tile).
// The products of a row are the same as the v1 kernels; they are summed in a
// different order (the 16 rows of a column are split differently over lanes).
#pragma once
#include <cstdint>
#include <type_traits>
#include <cuda_fp16.h>

namespace exl3v2 {

constexpr int NTOK_MAX = 8;   // widest row count the ntok kernels are built for

__host__ __device__ constexpr int words_of(int ka, int halfk) {
    return (halfk ? 256 * ka + 128 : 256 * ka) / 32;
}

// true when four consecutive windows fit one 32-bit extraction (ka <= 5)
__host__ __device__ constexpr bool pairable_ka(int ka, int halfk) {
    return 3 * ka + (halfk ? 1 : 0) + 16 <= 32 && words_of(ka, halfk) % 4 == 0;
}

// bytesum(w16 * M) as float; decode = KINV*bytesum + KOFF is applied around the dot product
__device__ __forceinline__ float bs_f32(uint32_t w16) {
    return (float)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
}

__device__ __forceinline__ void affine_consts(float& kinv, float& koff) {
    kinv = __half2float(__ushort_as_half(0x1eee));
    koff = 1024.f * kinv + __half2float(__ushort_as_half(0xc931));
}

// Ring position i = 8t + j of a tile maps to weight (row r, col c) with
// r = (t%4)*2 + (j&1) + ((j&2) ? 8 : 0), c = t/4 + ((j&4) ? 8 : 0). Lane (c, m)
// owns the four positions i0..i0+3, i0 = 32*(c&7) + 8*m + 4*(c>>3): same column,
// rows {2m, 2m+1, 2m+8, 2m+9}, and their 16-bit windows start s1, s2, s3 bits
// after window 0 — all inside the 32 ring bits that start at bit `d` of word m0.
struct WinGeo { int m0, m1, d, s1, s2, s3; };

__device__ __forceinline__ WinGeo win_geo(int c, int m, int ka, int halfk) {
    const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
    const int nw = r_bits >> 5;
    const int i0 = 32 * (c & 7) + 8 * m + 4 * (c >> 3);
    const auto bstart = [&](int i) {
        const int s_i = (i + 1) * ka + (halfk ? (i + 1) >> 1 : 0);   // inclusive prefix end
        return s_i - 16 + (s_i < 16 ? r_bits : 0);
    };
    WinGeo g;
    const int b = bstart(i0);
    g.m0 = b >> 5;
    g.m1 = (g.m0 + 1 == nw) ? 0 : g.m0 + 1;
    g.d = b & 31;
    // a window that wraps the ring end gets its own +r_bits, hence the modulo
    g.s1 = (bstart(i0 + 1) - b + r_bits) % r_bits;
    g.s2 = (bstart(i0 + 2) - b + r_bits) % r_bits;
    g.s3 = (bstart(i0 + 3) - b + r_bits) % r_bits;
    return g;
}

// ---- shared pipeline pieces -------------------------------------------------
// Each thread owns NS uint4 slots of a batch: uint4 index u = tid + 256*j of
// the batch's 4 tile-groups x TB rows x (w/4) uint4s. Slot geometry depends on
// the runtime w, so it is resolved once (integer division) at kernel start.
template <int TB, int WM>
struct Slots {
    static constexpr int NS = (4 * TB * (WM / 4) + 255) / 256;
    int g[NS], r[NS], q[NS];
    __device__ void init(int tid, int w) {
        const int w4 = w >> 2, per_tile = TB * w4, total = 4 * per_tile;
#pragma unroll
        for (int j = 0; j < NS; ++j) {
            const int u = tid + 256 * j;
            g[j] = -1; r[j] = 0; q[j] = 0;
            if (u < total) {
                g[j] = u / per_tile;
                const int v = u - g[j] * per_tile;
                r[j] = v / w4;
                q[j] = v - r[j] * w4;
            }
        }
    }
};

// 2-stage smem ring of tile words
template <int TB, int WM>
struct Ring {
    uint32_t rg[2][4][TB][WM];
    template <int NS>
    __device__ void store(int st, const Slots<TB, WM>& sl, const uint4 (&src)[NS]) {
#pragma unroll
        for (int j = 0; j < NS; ++j)
            if (sl.g[j] >= 0) *(uint4*)&rg[st][sl.g[j]][sl.r[j]][sl.q[j] * 4] = src[j];
    }
    __device__ const uint32_t* row(int st, int g, int r) const { return &rg[st][g][r][0]; }
};

// Cooperative sum: out[tt] = sum over `n4` float4s at x[tt*xstride/4 + base4 ..]
// (red = 8*NTOK floats of scratch; two __syncthreads).
template <int NTOK>
__device__ __forceinline__ void block_sum_rows(const float* __restrict__ x, long long xstride,
                                               int base4, int n4, float* red_s, float (&out)[NTOK]) {
    const int tid = threadIdx.x;
    float r[NTOK];
#pragma unroll
    for (int tt = 0; tt < NTOK; ++tt) r[tt] = 0.f;
    for (int i = tid; i < n4; i += 256) {
#pragma unroll
        for (int tt = 0; tt < NTOK; ++tt) {
            const float4 v = *(const float4*)(x + (size_t)tt * xstride + base4 + i * 4);
            r[tt] += (v.x + v.y) + (v.z + v.w);
        }
    }
#pragma unroll
    for (int tt = 0; tt < NTOK; ++tt) {
#pragma unroll
        for (int o = 16; o; o >>= 1) r[tt] += __shfl_xor_sync(0xffffffffu, r[tt], o);
        if ((tid & 31) == 0) red_s[(tid >> 5) * NTOK + tt] = r[tt];
    }
    __syncthreads();
#pragma unroll
    for (int tt = 0; tt < NTOK; ++tt) {
        float a = 0.f;
#pragma unroll
        for (int wp = 0; wp < 8; ++wp) a += red_s[wp * NTOK + tt];
        out[tt] = a;
    }
    __syncthreads();
}

// One tile row for NC rows of x: 4 weights from one funnel shift, 2 loads per
// row (rows 2m,2m+1 | 2m+8,2m+9 at constant offsets from xb[tt]).
template <int NC>
__device__ __forceinline__ void tile_row(const uint32_t* __restrict__ ring, const WinGeo& geo,
                                         const float* const (&xb)[NC], int roff, float (&s)[NC]) {
    // Ring bits are numbered MSB-first inside each word (v1 convention), so the
    // 32 bits from bit `d` of word m0 are a LEFT funnel shift, and window k sits
    // s_k bits below the top: w_k = X >> (16 - s_k).
    const uint32_t X = __funnelshift_l(ring[geo.m1], ring[geo.m0], geo.d);
    const float f0 = bs_f32(X >> 16);
    const float f1 = bs_f32((X >> (16 - geo.s1)) & 0xffffu);
    const float f2 = bs_f32((X >> (16 - geo.s2)) & 0xffffu);
    const float f3 = bs_f32((X >> (16 - geo.s3)) & 0xffffu);
#pragma unroll
    for (int tt = 0; tt < NC; ++tt) {
        const float2 a = *(const float2*)(xb[tt] + roff);
        const float2 b = *(const float2*)(xb[tt] + roff + 8);
        s[tt] += (a.x * f0 + a.y * f1) + (b.x * f2 + b.y * f3);
    }
}

// DP4A variant: x pre-quantized to int8 per-16-group AND pre-packed so lane m's
// four elements {2m, 2m+1, 2m+8, 2m+9} sit in one u32. Weight = the four
// plane-bytes of p = w16*K (native int8); dp4a([x,x,x,x], p) == x*w exactly.
// Loop cost ~5.75 instr/weight vs ~6.75 fp32, plus no float x traffic.
template <int NC>
__device__ __forceinline__ void tile_row_d4(const uint32_t* __restrict__ ring, const WinGeo& geo,
                                            const uint32_t* const (&xq)[NC], const float* const (&xqs)[NC],
                                            int roff, float (&s)[NC]) {
    const uint32_t X = __funnelshift_l(ring[geo.m1], ring[geo.m0], geo.d);
    const uint32_t p0 = (X >> 16) * 0x83DCD12Du;
    const uint32_t p1 = ((X >> (16 - geo.s1)) & 0xffffu) * 0x83DCD12Du;
    const uint32_t p2 = ((X >> (16 - geo.s2)) & 0xffffu) * 0x83DCD12Du;
    const uint32_t p3 = ((X >> (16 - geo.s3)) & 0xffffu) * 0x83DCD12Du;
    const int gidx = roff >> 4;
#pragma unroll
    for (int tt = 0; tt < NC; ++tt) {
        const uint4 xp4 = ((const uint4*)xq[tt])[roff >> 2 & ~3u];   // uint4 idx = g*4 (xqb has +m)
        // dp4a.s32.u32: signed int8 splat x UNSIGNED plane bytes (exact weight bytesum)
        int isum = 0;
        asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "+r"(isum) : "r"(xp4.x), "r"(p0), "r"(0));
        asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "+r"(isum) : "r"(xp4.y), "r"(p1), "r"(isum));
        asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "+r"(isum) : "r"(xp4.z), "r"(p2), "r"(isum));
        asm volatile("dp4a.s32.u32 %0, %1, %2, %3;" : "+r"(isum) : "r"(xp4.w), "r"(p3), "r"(isum));
        s[tt] = fmaf((float)isum, xqs[tt][gidx], s[tt]);
    }
}

template <int NC, int TB, int WM>
__device__ __forceinline__ void batch_full_d4(const Ring<TB, WM>& ring, int st, int g, const WinGeo& geo,
                                              const uint32_t* const (&xq)[NC], const float* const (&xqs)[NC],
                                              int xbase, float (&s)[NC]) {
#pragma unroll
    for (int r = 0; r < TB; ++r) tile_row_d4<NC>(ring.row(st, g, r), geo, xq, xqs, xbase + r * 16, s);
}

template <int NC, int TB, int WM>
__device__ __forceinline__ void batch_tail_d4(const Ring<TB, WM>& ring, int st, int g, const WinGeo& geo,
                                              const uint32_t* const (&xq)[NC], const float* const (&xqs)[NC],
                                              int xbase, float (&s)[NC], int nrows) {
    for (int r = 0; r < nrows; ++r) tile_row_d4<NC>(ring.row(st, g, r), geo, xq, xqs, xbase + r * 16, s);
}

// A full batch of TB tile rows (roff is a compile-time multiple of 16).
template <int NC, int TB, int WM>
__device__ __forceinline__ void batch_full(const Ring<TB, WM>& ring, int st, int g, const WinGeo& geo,
                                           const float* const (&xb)[NC], int xbase, float (&s)[NC]) {
#pragma unroll
    for (int r = 0; r < TB; ++r) tile_row<NC>(ring.row(st, g, r), geo, xb, xbase + r * 16, s);
}

// Last batch with `nrows` valid rows: the rest are zero-filled, their x reads
// are clamped to the last valid row (finite garbage times zero).
template <int NC, int TB, int WM>
__device__ __forceinline__ void batch_tail(const Ring<TB, WM>& ring, int st, int g, const WinGeo& geo,
                                           const float* const (&xb)[NC], int xbase, float (&s)[NC], int nrows) {
    for (int r = 0; r < nrows; ++r) tile_row<NC>(ring.row(st, g, r), geo, xb, xbase + r * 16, s);
}

// 4-way lane reduce (lanes with equal c differ in m = lane & 3) + affine epilogue
template <int NC>
__device__ __forceinline__ void finish(float (&s)[NC], const float (&sx)[NC],
                                       const float& kinv, const float& koff) {
#pragma unroll
    for (int tt = 0; tt < NC; ++tt) {
        s[tt] += __shfl_xor_sync(0xffffffffu, s[tt], 1);
        s[tt] += __shfl_xor_sync(0xffffffffu, s[tt], 2);
        s[tt] = fmaf(kinv, s[tt], koff * sx[tt]);
    }
}

// ---- dense GEMV, NTOK rows --------------------------------------------------
// t[z][tt][n] (n = tn*16) = affine( sum_k xh[tt][k] * bytesum(w[k][n]) ), xh
// rows of tk*16 floats; the K range is split over gridDim.y (z) and had_out
// sums the z slices in order. Grid (ceil(tn/4), nky), block 256 = 4 tiles x
// 64 lanes. Weights: tk x tn tiles of `w` u32 (w <= WM, pairable ka).
template <int NTOK>
__global__ void __launch_bounds__(256) gemv_n(const uint32_t* __restrict__ trel,
                                              const float* __restrict__ xh,
                                              float* __restrict__ t,
                                              int tk, int tn, int ka, int halfk) {
    constexpr int TB = 8, WM = 48, DEPTH = 2;
    using SL = Slots<TB, WM>;
    __shared__ Ring<TB, WM> ring;
    __shared__ float red[8 * NTOK];
    const int tid = threadIdx.x;
    const int g = tid >> 6, c = (tid & 63) >> 2, m = tid & 3;
    const int tile = blockIdx.x * 4 + g;
    const bool live = tile < tn;
    const int w = words_of(ka, halfk);
    const int tkc = (tk + gridDim.y - 1) / gridDim.y;
    const int tk_lo = blockIdx.y * tkc, tk_hi = min(tk_lo + tkc, tk);
    const WinGeo geo = win_geo(c, m, ka, halfk);
    float kinv, koff; affine_consts(kinv, koff);
    SL sl; sl.init(tid, w);

    float sx[NTOK];
    block_sum_rows<NTOK>(xh, (long long)tk * 16, tk_lo * 16, (tk_hi - tk_lo) * 4, red, sx);

    float s[NTOK];
    const float* xb[NTOK];
#pragma unroll
    for (int tt = 0; tt < NTOK; ++tt) {
        s[tt] = 0.f;
        xb[tt] = xh + (size_t)tt * tk * 16 + tk_lo * 16 + 2 * m;
    }
    const int nb = (tk_hi - tk_lo + TB - 1) / TB;
    uint4 pf[DEPTH][SL::NS];
    const auto load = [&](int bi, uint4 (&dst)[SL::NS]) {
#pragma unroll
        for (int j = 0; j < SL::NS; ++j) {
            dst[j] = make_uint4(0u, 0u, 0u, 0u);
            if (sl.g[j] >= 0) {
                const int row = tk_lo + bi * TB + sl.r[j];
                if (live && row < tk_hi)
                    dst[j] = *(const uint4*)(trel + ((size_t)row * tn + blockIdx.x * 4 + sl.g[j]) * (size_t)w + sl.q[j] * 4);
            }
        }
    };
    // Batch b lives in pf[b % DEPTH]. At iteration bi its slot is free (batch bi
    // is already in smem), so batch bi + DEPTH is loaded there and lands DEPTH-1
    // compute iterations before its store — all with compile-time slot indices.
#pragma unroll
    for (int jj = 0; jj < DEPTH; ++jj) if (jj < nb) load(jj, pf[jj]);
    ring.store<SL::NS>(0, sl, pf[0]);
    __syncthreads();
    for (int b0 = 0; b0 < nb; b0 += DEPTH) {
#pragma unroll
        for (int jj = 0; jj < DEPTH; ++jj) {
            const int bi = b0 + jj;
            if (bi >= nb) break;
            if (bi + DEPTH < nb) load(bi + DEPTH, pf[jj]);
            if (live) {
                const int nrows = tk_hi - tk_lo - bi * TB;
                if (nrows >= TB) batch_full<NTOK, TB, WM>(ring, bi & 1, g, geo, xb, bi * TB * 16, s);
                else batch_tail<NTOK, TB, WM>(ring, bi & 1, g, geo, xb, bi * TB * 16, s, nrows);
            }
            if (bi + 1 < nb) {
                ring.store<SL::NS>((bi + 1) & 1, sl, pf[(jj + 1) % DEPTH]);
                __syncthreads();
            }
        }
    }
    finish<NTOK>(s, sx, kinv, koff);
    if (live && m == 0) {
#pragma unroll
        for (int tt = 0; tt < NTOK; ++tt)
            t[((size_t)blockIdx.y * NTOK + tt) * (size_t)(tn * 16) + (size_t)tile * 16 + c] = s[tt];
    }
}

// ---- union MoE GEMV (spec-decode verify) ------------------------------------
// One block per (4 output tiles, union expert u, projection zsel); the
// expert's consumer rows are the expanded slots estart[u]..estart[u+1]-1.
// Grid (ceil(tn/4), 64, descs2 ? 2 : 1); blocks with u >= *pnexp exit.
// t[slot * n + zsel * tzoff + col]; x rows: (zsel ? xh2 : xh)[(estart[u]+tt) * (zsel ? xh2s : tk*16)].
template <int DEPTH>
__global__ void __launch_bounds__(256) gemv_moe_n(const uint64_t* __restrict__ descs,
                                                  const uint64_t* __restrict__ descs2,
                                                  const int* __restrict__ uids,
                                                  const int* __restrict__ estart,
                                                  const float* __restrict__ xh,
                                                  const float* __restrict__ xh2, int xh2s,
                                                  float* __restrict__ t, int n, int tzoff,
                                                  int tk, int tn,
                                                  const int* __restrict__ pnexp,
                                                  const int* __restrict__ missf = nullptr,
                                                  int skipmode = 0,
                                                  const uint32_t* __restrict__ xq = nullptr,
                                                  const float* __restrict__ xqs = nullptr,
                                                  const uint32_t* __restrict__ xq2 = nullptr,
                                                  const float* __restrict__ xqs2 = nullptr) {
    constexpr int TB = 8, WM = 32;
    using SL = Slots<TB, WM>;
    __shared__ Ring<TB, WM> ring;
    __shared__ float red[8 * NTOK_MAX];
    const int u = blockIdx.y;
    if (pnexp && u >= *pnexp) return;
    if (skipmode && ((missf[u] != 0) == (skipmode == 1))) return;
    const int nsel = descs2 ? 2 : 1;
    const int zsel = blockIdx.z % nsel;
    const uint64_t* d = (zsel ? descs2 : descs) + uids[u] * 4;
    const uint32_t* trel = (const uint32_t*)(uintptr_t)d[0];
    const int ka = (int)(d[3] & 0xff);
    const int halfk = (int)((d[3] >> 8) & 1);
    const int w = words_of(ka, halfk);
    const int sx0 = estart[u];
    const int nc = estart[u + 1] - sx0;
    const int xstride = zsel ? xh2s : tk * 16;
    const float* xg = (zsel ? xh2 : xh) + (size_t)sx0 * xstride;
    const int tid = threadIdx.x;
    const int g = tid >> 6, c = (tid & 63) >> 2, m = tid & 3;
    const int tile = blockIdx.x * 4 + g;
    const bool live = tile < tn;
    const WinGeo geo = win_geo(c, m, ka, halfk);
    float kinv, koff; affine_consts(kinv, koff);
    SL sl; sl.init(tid, w);

    const auto body = [&](auto ncc) {
        constexpr int NC = decltype(ncc)::value;
        float sx[NC], s[NC];
        const float* xb[NC];
        const uint32_t* xqu = zsel ? xq2 : xq;
        const float* xqsu = zsel ? xqs2 : xqs;
        const uint32_t* xqb[NC];
        const float* xqsb[NC];
        block_sum_rows<NC>(xg, xstride, 0, tk * 4, red, sx);
#pragma unroll
        for (int tt = 0; tt < NC; ++tt) { s[tt] = 0.f; xb[tt] = xg + (size_t)tt * xstride + 2 * m; }
        if (xqu) {
            // xq = uint4 splats at [slot*tk*4 + g*4 + m]; base +m picks lane pack
            const uint32_t* xqrow = xqu + (size_t)sx0 * tk * 16 + m * 4;
            const float* xqsrow = xqsu + (size_t)sx0 * tk;
#pragma unroll
            for (int tt = 0; tt < NC; ++tt) {
                xqb[tt] = xqrow + (size_t)tt * tk * 16;
                xqsb[tt] = xqsrow + (size_t)tt * tk;
            }
        }
        const int nb = (tk + TB - 1) / TB;
        uint4 pf[DEPTH][SL::NS];
        const auto load = [&](int bi, uint4 (&dst)[SL::NS]) {
#pragma unroll
            for (int j = 0; j < SL::NS; ++j) {
                dst[j] = make_uint4(0u, 0u, 0u, 0u);
                if (sl.g[j] >= 0) {
                    const int row = bi * TB + sl.r[j];
                    if (live && row < tk)
                        dst[j] = *(const uint4*)(trel + ((size_t)row * tn + blockIdx.x * 4 + sl.g[j]) * (size_t)w + sl.q[j] * 4);
                }
            }
        };
#pragma unroll
        for (int jj = 0; jj < DEPTH; ++jj) if (jj < nb) load(jj, pf[jj]);
        ring.store<SL::NS>(0, sl, pf[0]);
        __syncthreads();
        for (int b0 = 0; b0 < nb; b0 += DEPTH) {
#pragma unroll
            for (int jj = 0; jj < DEPTH; ++jj) {
                const int bi = b0 + jj;
                if (bi >= nb) break;
                if (bi + DEPTH < nb) load(bi + DEPTH, pf[jj]);   // slot jj is free (see gemv_n)
                if (live) {
                    const int nrows = tk - bi * TB;
                    if (xqu) {
                        if (nrows >= TB) batch_full_d4<NC, TB, WM>(ring, bi & 1, g, geo, xqb, xqsb, bi * TB * 16, s);
                        else batch_tail_d4<NC, TB, WM>(ring, bi & 1, g, geo, xqb, xqsb, bi * TB * 16, s, nrows);
                    } else {
                        if (nrows >= TB) batch_full<NC, TB, WM>(ring, bi & 1, g, geo, xb, bi * TB * 16, s);
                        else batch_tail<NC, TB, WM>(ring, bi & 1, g, geo, xb, bi * TB * 16, s, nrows);
                    }
                }
                if (bi + 1 < nb) {
                    ring.store<SL::NS>((bi + 1) & 1, sl, pf[(jj + 1) % DEPTH]);
                    __syncthreads();
                }
            }
        }
        finish<NC>(s, sx, kinv, koff);
        if (live && m == 0) {
#pragma unroll
            for (int tt = 0; tt < NC; ++tt)
                t[(size_t)(sx0 + tt) * (size_t)n + (size_t)zsel * tzoff + (size_t)tile * 16 + c] = s[tt];
        }
    };
    switch (nc) {
        case 1: body(std::integral_constant<int, 1>{}); break;
        case 2: body(std::integral_constant<int, 2>{}); break;
        case 3: body(std::integral_constant<int, 3>{}); break;
        case 4: body(std::integral_constant<int, 4>{}); break;
        case 5: body(std::integral_constant<int, 5>{}); break;
        case 6: body(std::integral_constant<int, 6>{}); break;
        case 7: body(std::integral_constant<int, 7>{}); break;
        default: body(std::integral_constant<int, NTOK_MAX>{}); break;
    }
}

}  // namespace exl3v2
