/* Challenge 3b candidate: whole-loop-optimal decode ("direct-window" schedule).
 *
 * Same proven-optimal core as shipped (2 XMAD + IDP.4A + I2F + FFMA per weight),
 * but window extraction is re-scheduled to the lead16-verified minimum:
 *   - each lane loads its 2 (K!=3.5/5: span<=64b) or 3 (K=3.5/5) ring words
 *     DIRECTLY via immediate-offset LDG.32s from a per-lane base register
 *     (no donor SHFLs at all);
 *   - window j sits at stream bit p = o_t + D_j; if p % 16 == 0 it is a free
 *     half-word operand (XMAD .H0/.H1 selector, zero instructions);
 *   - otherwise ONE SHF.L.W funnel from the raw bracketing word pair yields the
 *     window in the top half (no base-alignment funnels);
 *   - K=4: windows (j, j+4) for j=0..2 are exactly 16 bits apart, so one funnel
 *     per pair yields both (top half = j, bottom half = j+4): 3 SHF/lane/tile.
 *
 * x delivery: 4 floats/lane/tile via one LDS.128 from a shared-memory x buffer
 * permuted at block start (lane t's group r0 = 2*(t&3) pre-swizzled to be
 * conflict-free); this is value-identical to reading xh directly, which is what
 * this CPU model does. Weight tensor layout is untouched (cand_has_relayout=0).
 */

#include "exl3_ref.h"

#include <stddef.h>

const char *cand_name(void) {
    return "ch3b-direct-window-shf-min";
}

int cand_has_scalar_decode(void) {
    return 1;
}

int cand_has_relayout(void) {
    return 0;
}

void cand_relayout_matrix(const uint32_t *src, uint32_t *dst,
                          int tk, int tn, int ka, int halfk) {
    (void)src; (void)dst; (void)tk; (void)tn; (void)ka; (void)halfk;
}

/* Exact Pascal SHF.L.W (funnel shift left, high 32 bits, shift mod 32). */
static inline uint32_t shf_l_w(uint32_t lo, uint32_t hi, int s) {
    const uint32_t sh = (uint32_t)s & 31u;
    const uint64_t v = (((uint64_t)hi << 32) | (uint64_t)lo) << sh;
    return (uint32_t)(v >> 32);
}

/* Exact patched 2-XMAD high-half product (identical to shipped core). */
static inline uint32_t ch2_mul16_hi_patched(uint32_t U) {
    const uint32_t w = U >> 16;
    const uint32_t p0 = w * EXL3_MAGIC_LO;
    const uint32_t p1 = p0 + ((w * EXL3_MAGIC_HI) << 16);
    return p1;
}

/* Exact IDP.4A.U8.U8(p, 0x01010101, 0) + I2F.F32.S32. */
static inline float ch2_bs_f32(uint32_t p) {
    const uint32_t s = (p & 0xffu)
                     + ((p >> 8) & 0xffu)
                     + ((p >> 16) & 0xffu)
                     + ((p >> 24) & 0xffu);
    return (float)(int32_t)s;
}

static inline int d_of(int j, int ka, int h) { return j * ka + h * ((j + 1) >> 1); }
static inline int s_of(int i, int ka, int h) { return (i + 1) * ka + h * ((i + 1) >> 1); }

/* Faithful PRMT (default mode): pool = {a.b0..a.b3, b.b0..b.b3} (b0 = LSB). */
static inline uint32_t prmt_bytes(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d = 0;
    for (int r = 0; r < 4; ++r) {
        const int idx = (int)((sel >> (4 * r)) & 7u);
        const uint32_t byte = (idx < 4) ? (a >> (8 * idx)) : (b >> (8 * (idx - 4)));
        d |= (byte & 0xffu) << (8 * r);
    }
    return d;
}

/* Window j of lane t from the lane's raw words W[0..2] at stream bit p = o + D_j.
 *   p % 16 == 0 : free half-word operand (XMAD .H0/.H1 selector, zero instr);
 *   p %  8 == 0 : ONE PRMT (own half-rate pipe, dual-issues outside N_ALU+Shift);
 *   else        : ONE SHF.L.W funnel from the raw bracketing word pair.
 * Warp-uniform PRMT cases (lead18-verified): K=2 j=3; K=4 j=1 and j=5 -- and for
 * K=4 the pair (j=1, j=5) at p = 24, 40 comes from a SINGLE PRMT sel=0x0765 whose
 * .H1/.H0 halves feed the two XMAD hash chains (2 windows, 1 instruction). */
static inline uint32_t window_u32(const uint32_t W[3], int o, int D) {
    const int p = o + D;
    const int wi = p >> 5, off = p & 31;
    if (off == 0)  return W[wi];              /* top half free: U = W[wi]         */
    if (off == 16) return W[wi] << 16;        /* low half free: U = W[wi] << 16   */
    if ((off & 7) == 0) {                     /* one PRMT, window in top half     */
        const int k = p >> 3;                 /* first stream byte (MSB-first)    */
        const int m = k >> 2;
        const uint32_t a = W[m], b = W[m + 1 > 2 ? 2 : m + 1];
        const int i0 = 3 - (k - 4 * m);                       /* pool idx byte k   */
        const int i1 = ((k + 1) & 3) == 0 ? 7 : 3 - ((k + 1) & 3); /* pool idx k+1  */
        const uint32_t sel = ((uint32_t)i0 << 12) | ((uint32_t)i1 << 8);
        return prmt_bytes(a, b, sel);
    }
    return shf_l_w(W[wi + 1], W[wi], off);    /* one funnel, window in top half   */
}

float cand_decode_state_f32(uint16_t w) {
    const uint32_t U = ((uint32_t)w << 16) | 0xA55Au; /* low half don't-care */
    return ch2_bs_f32(ch2_mul16_hi_patched(U));
}

void cand_decode_tile_f32(const uint32_t *ring, int ka, int halfk, float f_out[256]) {
    const int nw = exl3_nw(ka, halfk);
    const int R = 32 * nw;

    for (int t = 0; t < 32; ++t) {
        int B = s_of(8 * t, ka, halfk) - 16;
        B %= R; if (B < 0) B += R;
        const int o = B & 31, w0 = B >> 5;

        /* Direct immediate-offset loads: 2 words when o + D_7 + 16 <= 64
         * (K = 1.5, 2, 2.5, 3, 4), else 3 words (K = 3.5, 5). lead16-verified. */
        uint32_t W[3];
        W[0] = ring[w0];
        W[1] = ring[(w0 + 1) % nw];
        W[2] = ring[(w0 + 2) % nw]; /* fetched but unused for span <= 64 */

        for (int j = 0; j < 8; ++j) {
            const uint32_t U = window_u32(W, o, d_of(j, ka, halfk));
            f_out[8 * t + j] = ch2_bs_f32(ch2_mul16_hi_patched(U));
        }
    }
}

void cand_gemv_tile_col(const uint32_t *tb, int tk_lo, int tk_hi,
                        int tn, int tn_i, int ka, int halfk,
                        const float *xh, float out16[16]) {
    const int nw = exl3_nw(ka, halfk);
    const size_t stride = (size_t)tn * (size_t)nw;
    const uint32_t *col_base = tb + (size_t)tn_i * (size_t)nw;

    float a0[32], a1[32];
    for (int t = 0; t < 32; ++t) {
        a0[t] = 0.0f;
        a1[t] = 0.0f;
    }

    for (int a = tk_lo; a < tk_hi; ++a) {
        const uint32_t *ring = col_base + (size_t)a * stride;
        const float *xt = xh + a * 16;
        float f_tile[256];
        cand_decode_tile_f32(ring, ka, halfk, f_tile);
        for (int t = 0; t < 32; ++t) {
            const int r0 = 2 * (t & 3);
            const float *f = f_tile + 8 * t;
            a0[t] = exl3_ffma_ftz(xt[r0 + 0], f[0], a0[t]);
            a0[t] = exl3_ffma_ftz(xt[r0 + 1], f[1], a0[t]);
            a0[t] = exl3_ffma_ftz(xt[r0 + 8], f[2], a0[t]);
            a0[t] = exl3_ffma_ftz(xt[r0 + 9], f[3], a0[t]);
            a1[t] = exl3_ffma_ftz(xt[r0 + 0], f[4], a1[t]);
            a1[t] = exl3_ffma_ftz(xt[r0 + 1], f[5], a1[t]);
            a1[t] = exl3_ffma_ftz(xt[r0 + 8], f[6], a1[t]);
            a1[t] = exl3_ffma_ftz(xt[r0 + 9], f[7], a1[t]);
        }
    }

    const float sx = exl3_compute_sx_ref(xh, tk_lo, tk_hi);
    const float kinv = exl3_u2f(EXL3_KINV_BITS);
    const float koff = exl3_u2f(EXL3_KOFF_BITS);
    const float off = exl3_fmul_ftz(koff, sx);

    for (int c = 0; c < 8; ++c) {
        const int base = c * 4;
        const float A0 = exl3_fadd_ftz(
            exl3_fadd_ftz(a0[base + 0], a0[base + 1]),
            exl3_fadd_ftz(a0[base + 2], a0[base + 3]));
        const float A1 = exl3_fadd_ftz(
            exl3_fadd_ftz(a1[base + 0], a1[base + 1]),
            exl3_fadd_ftz(a1[base + 2], a1[base + 3]));
        out16[c]     = exl3_ffma_ftz(kinv, A0, off);
        out16[c + 8] = exl3_ffma_ftz(kinv, A1, off);
    }
}
