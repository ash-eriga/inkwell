// Exhaustive CPU verification for EXL3 mul1 decode math reformulations.
// Checks ALL 65,536 window values, ALL 4,063,232 incremental transitions across k=1..5,
// ALL 7 K geometries for rotated-ring + paired extraction, and end-to-end GEMV accuracy
// (Hadamard-like zero-mean x with outliers AND adversarial all-positive x).
//
// Build & run (CPU only):
//   g++ -O3 -std=c++17 opt/kbench/exl3_math/test_cpu.cpp -o opt/kbench/exl3_math/test_cpu -lm && opt/kbench/exl3_math/test_cpu

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr uint32_t MUL1_C  = 0x83DCD12Du;
static constexpr uint32_t MUL1_CL = 0xD12Du;
static constexpr uint32_t MUL1_CH = 0x83DCu;

static float h2f(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    float v = e ? ldexpf(1.f + m / 1024.f, e - 15) : ldexpf(m / 1024.f, -14);
    return (h & 0x8000) ? -v : v;
}

static inline uint32_t bits_as_u32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static inline float u32_as_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static inline uint32_t dp4a_u32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t s = c;
    for (int i = 0; i < 4; ++i)
        s += ((a >> (8 * i)) & 0xFFu) * ((b >> (8 * i)) & 0xFFu);
    return s;
}

static inline int32_t dp4a_s32_u32(uint32_t a, uint32_t b, int32_t c) {
    int32_t s = c;
    for (int i = 0; i < 4; ++i) {
        int8_t ai = (int8_t)((a >> (8 * i)) & 0xFFu);
        uint8_t bi = (uint8_t)((b >> (8 * i)) & 0xFFu);
        s += (int32_t)ai * (int32_t)bi;
    }
    return s;
}

static inline int32_t dp2a_lo_s32_u32(uint32_t a, uint32_t b, int32_t c) {
    int16_t a0 = (int16_t)(a & 0xFFFFu), a1 = (int16_t)(a >> 16);
    uint8_t b0 = (uint8_t)(b & 0xFFu), b1 = (uint8_t)((b >> 8) & 0xFFu);
    return c + (int32_t)a0 * (int32_t)b0 + (int32_t)a1 * (int32_t)b1;
}

static inline int32_t dp2a_hi_s32_u32(uint32_t a, uint32_t b, int32_t c) {
    int16_t a0 = (int16_t)(a & 0xFFFFu), a1 = (int16_t)(a >> 16);
    uint8_t b2 = (uint8_t)((b >> 16) & 0xFFu), b3 = (uint8_t)((b >> 24) & 0xFFu);
    return c + (int32_t)a0 * (int32_t)b2 + (int32_t)a1 * (int32_t)b3;
}

static inline uint32_t ref_prod(uint16_t h) {
    return (uint32_t)h * MUL1_C;
}

static inline uint32_t ref_bs(uint16_t h) {
    uint32_t p = ref_prod(h);
    return (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
}

// Reference window extractor (exact match to kernels.cu:exl3_window / sgemv_bench.cu:ref_gemv)
static inline uint16_t ref_window(const uint32_t* ring, int i, int ka, int hk) {
    const int nw = 8 * ka + 4 * hk;
    const int R = nw * 32;
    const int s = hk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
    const int b0 = s - 16 + (s < 16 ? R : 0);
    const int i0 = b0 >> 5;
    const int s0 = 48 - (b0 & 31);
    const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[i1];
    return (uint16_t)((v >> s0) & 0xffffu);
}

// Load-time lossless ring rotation: rotate cyclic bitstream left by shift_bits
// so that window i=0 starts at bit 0 (MSB-first of word 0).
static void rotate_ring_left(const uint32_t* src, uint32_t* dst, int nw, int shift_bits) {
    const int R = nw * 32;
    shift_bits = ((shift_bits % R) + R) % R;
    const int ws = shift_bits >> 5;
    const int bs = shift_bits & 31;
    for (int w = 0; w < nw; ++w) {
        int i0 = (w + ws) % nw;
        int i1 = (i0 + 1) % nw;
        if (bs == 0) dst[w] = src[i0];
        else dst[w] = (src[i0] << bs) | (src[i1] >> (32 - bs));
    }
}

// Extract 32-bit word starting at bit offset D (MSB-first) from cyclic word array W[0..nw-1]
static inline uint32_t stream_word32(const uint32_t* W, int nw, int D) {
    const int R = nw * 32;
    D = ((D % R) + R) % R;
    int w0 = D >> 5, b = D & 31;
    int w1 = (w0 + 1 == nw) ? 0 : w0 + 1;
    if (b == 0) return W[w0];
    return (W[w0] << b) | (W[w1] >> (32 - b));
}

// Extract 16-bit window at offset D using optimal A1+A2+A3 rules (aligned / paired SHF / (D-1)+IADD)
static void extract_chunk_opt(const uint32_t* rot_ring, int nw, int base_bit, int ka, int hk, int M, uint16_t* out) {
    // base_bit is the starting bit of window 0 of this chunk in rot_ring.
    // For M=64 (all K) or M=32 (integer K), base_bit % 32 == 0!
    // For M=32 (half-K), base_bit % 32 is 0 (even lane) or 16 (odd lane).
    // Align a local window of words starting at base_bit (0 SHF when base_bit%32==0).
    int offs[64];
    for (int j = 0; j < M; ++j) offs[j] = j * ka + hk * ((j + 1) >> 1);
    const int span = offs[M - 1] + 16;
    const int nloc = (span + 31) / 32 + 1;
    uint32_t W[16];
    for (int k = 0; k < nloc; ++k) W[k] = stream_word32(rot_ring, nw, base_bit + 32 * k);

    auto shf_l = [&](int d) -> uint32_t {
        int w = d >> 5, b = d & 31;
        return b == 0 ? W[w] : ((W[w] << b) | (W[w + 1] >> (32 - b)));
    };

    bool done[64] = {};
    // 1. Aligned (D % 16 == 0): direct .H1 / .H0
    for (int j = 0; j < M; ++j) {
        if ((offs[j] & 15) == 0) {
            uint32_t word = W[offs[j] >> 5];
            out[j] = (offs[j] & 16) ? (uint16_t)(word & 0xFFFFu) : (uint16_t)(word >> 16);
            done[j] = true;
        }
    }
    // Map offset -> j for quick pairing
    int idx_of[512];
    std::memset(idx_of, -1, sizeof(idx_of));
    for (int j = 0; j < M; ++j) idx_of[offs[j]] = j;

    // 2. Paired (D, D+16) via 1 SHF -> .H1 and .H0 (+ optional IADD U+U for D+1!)
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        int d = offs[j];
        int j2 = (d + 16 < 512) ? idx_of[d + 16] : -1;
        if (j2 >= 0 && !done[j2]) {
            uint32_t U = shf_l(d);
            out[j]  = (uint16_t)(U >> 16);        // U.H1
            out[j2] = (uint16_t)(U & 0xFFFFu);    // U.H0
            done[j] = done[j2] = true;
            // Check if d+1 is also needed -> IADD (U + U).H1!
            int j3 = (d + 1 < 512) ? idx_of[d + 1] : -1;
            if (j3 >= 0 && !done[j3]) {
                uint32_t V = U + U;
                out[j3] = (uint16_t)(V >> 16);    // V.H1
                done[j3] = true;
            }
        }
    }
    // 3. Check if any remaining d has d == 32*k + 1 -> IADD (W[k] + W[k]).H1
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        int d = offs[j];
        if ((d & 31) == 1) {
            uint32_t V = W[d >> 5] + W[d >> 5];
            out[j] = (uint16_t)(V >> 16);
            done[j] = true;
        }
    }
    // 4. Paired (D, D+15) via 1 SHF(D-1) + 1 IADD(U+U): U.H0 = D+15, (U+U).H1 = D
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        int d = offs[j];
        int j2 = (d + 15 < 512) ? idx_of[d + 15] : -1;
        if (d >= 1 && j2 >= 0 && !done[j2]) {
            uint32_t U = shf_l(d - 1);
            uint32_t V = U + U;
            out[j]  = (uint16_t)(V >> 16);        // (U+U).H1 = D
            out[j2] = (uint16_t)(U & 0xFFFFu);    // U.H0     = D+15
            done[j] = done[j2] = true;
        }
    }
    // 5. Remaining singletons
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        uint32_t U = shf_l(offs[j]);
        out[j] = (uint16_t)(U >> 16);
        done[j] = true;
    }
}

static uint32_t rng_s = 12345u;
static uint32_t rnd() {
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 17;
    rng_s ^= rng_s << 5;
    return rng_s;
}

int main() {
    printf("=== EXL3 mul1 Mathematical Reformulation Exhaustive CPU Tests ===\n\n");

    // -------------------------------------------------------------------------
    // TEST 1: All 65,536 window values (codebook & float conversion tricks)
    // -------------------------------------------------------------------------
    printf("[TEST 1] Exhaustive codebook & conversion checks (all 65,536 windows h=0..65535)\n");
    int pass_b1 = 0, fail_b1 = 0;
    int pass_b1_h1 = 0, fail_b1_h1 = 0;
    int pass_b3 = 0, fail_b3 = 0;
    int pass_c1 = 0, fail_c1 = 0;
    int pass_c3 = 0, fail_c3 = 0;
    int pass_c4 = 0, fail_c4 = 0;
    int pass_a4 = 0, fail_a4 = 0;

    uint32_t Q_byte[256];
    for (uint32_t b = 0; b < 256; ++b) Q_byte[b] = b * MUL1_C;

    const float KINV = h2f(0x1eee);
    const float KC   = h2f(0xc931);
    const float KOFF = 1024.f * KINV + KC;
    const double KINV_d = KINV, KOFF_d = KOFF;

    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t pref = ref_prod((uint16_t)h);
        const uint32_t bsref = ref_bs((uint16_t)h);

        // B1: 2-XMAD 16x32 split: (h * CL) + ((h * CH) << 16)
        uint32_t p_b1 = (h * MUL1_CL) + ((h * MUL1_CH) << 16);
        if (p_b1 == pref) pass_b1++; else fail_b1++;

        // B1_H1: direct .H1 / .H0 from packed 32-bit U = (h << 16) | (h ^ 0xA55A)
        uint32_t U = (h << 16) | ((h ^ 0xA55Au) & 0xFFFFu);
        uint32_t h1 = U >> 16;
        uint32_t p_h1 = (h1 * MUL1_CL) + ((h1 * MUL1_CH) << 16);
        if (p_h1 == pref) pass_b1_h1++; else fail_b1_h1++;

        // B3: Byte-shared decomposition h = 256*A + B -> (Q[A] << 8) + Q[B]
        uint32_t p_b3 = (Q_byte[h >> 8] << 8) + Q_byte[h & 0xFFu];
        if (p_b3 == pref) pass_b3++; else fail_b3++;

        // C1: Magic float 0x4B000000 - 2^23 (exact IEEE-754)
        uint32_t mag1 = dp4a_u32(pref, 0x01010101u, 0x4B000000u);
        float f_c1 = u32_as_f32(mag1) - 8388608.f;
        // C1c: Zero-centered exact magic: 0x4B000000 - (2^23 + 510) = exact (float)(bs - 510)
        float f_c1c = u32_as_f32(mag1) - 8389118.f;
        if (f_c1 == (float)bsref && bits_as_u32(f_c1) == bits_as_u32((float)bsref) &&
            f_c1c == (float)((int32_t)bsref - 510))
            pass_c1++;
        else
            fail_c1++;

        // C3: Max-scale folded magic 0x3F800000 + 255 * bs (exact IEEE-754 identity)
        uint32_t mag3 = dp4a_u32(pref, 0xFFFFFFFFu, 0x3F800000u);
        float f_c3 = u32_as_f32(mag3);
        double exact_c3 = 1.0 + (255.0 / 8388608.0) * (double)bsref;
        if ((double)f_c3 == exact_c3) pass_c3++; else fail_c3++;

        // C4: Zero-centered bs_c = dp4a(P, 0x01010101, -510) in [-510, +510]
        int32_t bs_c = (int32_t)dp4a_u32(pref, 0x01010101u, (uint32_t)(-510));
        bool ok_c4 = (bs_c == (int32_t)bsref - 510) && (bs_c >= -510 && bs_c <= 510);
        double w_ref = KINV_d * (double)bsref + KOFF_d;
        double w_c4  = KINV_d * (double)bs_c + (KOFF_d + 510.0 * KINV_d);
        if (ok_c4 && fabs(w_ref - w_c4) < 1e-14) pass_c4++; else fail_c4++;

        // A4: XMAD-based shift extraction for D=1..15 (check all 65536 hi_1 with paired hi_0)
        uint32_t hi = (h << 16) | ((h * 2654435761u) >> 16);
        bool ok_a4 = true;
        for (int D = 1; D <= 15; ++D) {
            uint32_t ref_w = (hi >> (16 - D)) & 0xFFFFu;
            uint32_t t = (hi & 0xFFFFu) * (1u << D);          // XMAD hi.H0, (1<<D), RZ
            uint32_t u = ((hi >> 16) * (1u << D)) + (t >> 16); // XMAD.CHI hi.H1, (1<<D), t
            if ((u & 0xFFFFu) != ref_w) ok_a4 = false;
        }
        if (ok_a4) pass_a4++; else fail_a4++;
    }
    printf("  B1  (16x32 2-XMAD split)          : PASS=%d FAIL=%d\n", pass_b1, fail_b1);
    printf("  B1H (packed U.H1 direct 2-XMAD)   : PASS=%d FAIL=%d\n", pass_b1_h1, fail_b1_h1);
    printf("  B3  (byte-shared (Q[A]<<8)+Q[B])  : PASS=%d FAIL=%d\n", pass_b3, fail_b3);
    printf("  C1  (0x4B000000 - 2^23 exact f32) : PASS=%d FAIL=%d\n", pass_c1, fail_c1);
    printf("  C3  (0x3F800000 + 255*bs exact)   : PASS=%d FAIL=%d\n", pass_c3, fail_c3);
    printf("  C4  (zero-centered bs-510 exact)  : PASS=%d FAIL=%d\n", pass_c4, fail_c4);
    printf("  A4  (XMAD shift D=1..15 x 65536)  : PASS=%d FAIL=%d\n\n", pass_a4, fail_a4);

    // -------------------------------------------------------------------------
    // TEST 2: Incremental update P_{i+1} from P_i — ALL transitions for k=1..5
    // -------------------------------------------------------------------------
    printf("[TEST 2] Exhaustive incremental update P_{i+1} = (P_i << k) + v*C - u*(CL<<16) mod 2^32\n");
    const uint32_t CL_HI = MUL1_CL << 16;
    for (int k = 1; k <= 5; ++k) {
        int pass = 0, fail = 0;
        const uint32_t nv = 1u << k;
        for (uint32_t h = 0; h < 65536; ++h) {
            const uint32_t P_i = ref_prod((uint16_t)h);
            const uint32_t u = h >> (16 - k);
            for (uint32_t v = 0; v < nv; ++v) {
                const uint16_t h_next = (uint16_t)(((h << k) & 0xFFFFu) | v);
                const uint32_t P_ref = ref_prod(h_next);
                const uint32_t P_inc = (P_i << k) + v * MUL1_C - u * CL_HI;
                if (P_inc == P_ref) pass++; else fail++;
            }
        }
        printf("  B2  (step k=%d, %7d transitions) : PASS=%d FAIL=%d\n", k, 65536 * (int)nv, pass, fail);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // TEST 3: Load-time ring rotation + paired/IADD extraction (all 7 K)
    // -------------------------------------------------------------------------
    printf("[TEST 3] Rotated-ring + paired (D,D+16) + (D-1)+IADD extraction (all 7 K, 1000 tiles each)\n");
    struct KH { int ka, hk; };
    const KH khs[7] = {{1,1}, {2,0}, {2,1}, {3,0}, {3,1}, {4,0}, {5,0}};
    for (int ki = 0; ki < 7; ++ki) {
        const int ka = khs[ki].ka, hk = khs[ki].hk;
        const int nw = 8 * ka + 4 * hk;
        const int R = nw * 32;
        // Start bit of window i=0 in original ring:
        const int s0 = hk ? (1 * ka + (1 >> 1)) : (1 * ka); // = ka
        const int b0_init = s0 - 16 + (s0 < 16 ? R : 0);
        int pass32 = 0, fail32 = 0, pass64 = 0, fail64 = 0;
        uint32_t ring[40], rot[40];
        for (int tile = 0; tile < 1000; ++tile) {
            for (int w = 0; w < nw; ++w) ring[w] = rnd();
            rotate_ring_left(ring, rot, nw, b0_init);
            // M = 32 (8 lanes per tile)
            for (int lane = 0; lane < 8; ++lane) {
                const int i_base = lane * 32;
                const int base_bit = i_base * ka + hk * (i_base >> 1);
                uint16_t got[32];
                extract_chunk_opt(rot, nw, base_bit, ka, hk, 32, got);
                for (int j = 0; j < 32; ++j) {
                    if (got[j] == ref_window(ring, i_base + j, ka, hk)) pass32++; else fail32++;
                }
            }
            // M = 64 (4 lanes per tile)
            for (int lane = 0; lane < 4; ++lane) {
                const int i_base = lane * 64;
                const int base_bit = i_base * ka + hk * (i_base >> 1);
                uint16_t got[64];
                extract_chunk_opt(rot, nw, base_bit, ka, hk, 64, got);
                for (int j = 0; j < 64; ++j) {
                    if (got[j] == ref_window(ring, i_base + j, ka, hk)) pass64++; else fail64++;
                }
            }
        }
        printf("  A1-A3 K=%.1f (M=32): PASS=%d FAIL=%d | (M=64): PASS=%d FAIL=%d\n",
               ka + 0.5 * hk, pass32, fail32, pass64, fail64);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // TEST 4: End-to-End GEMV Accuracy (K=2304, N=1152 -> tk=144, tn=72, K=2.0)
    // Compare all MAC formulations on (a) Hadamard-like zero-mean x w/ 6x outliers,
    // and (b) adversarial all-positive x (nonzero mean).
    // -------------------------------------------------------------------------
    printf("[TEST 4] End-to-End GEMV Accuracy vs FP64 ref (K=2304, N=1152, K=2.0)\n");
    const int TK = 144, TN = 72, KDIM = TK * 16, NDIM = TN * 16;
    const int NW = 16;
    std::vector<uint32_t> tr((size_t)TK * TN * NW);
    for (auto& v : tr) v = rnd();

    // Decode all tiles' bs[tk][tn][r][c] once for fast multi-method comparison
    std::vector<uint16_t> bs_mat((size_t)KDIM * NDIM);
    std::vector<uint32_t> p_mat((size_t)KDIM * NDIM);
    for (int a = 0; a < TK; ++a) {
        for (int b = 0; b < TN; ++b) {
            const uint32_t* ring = &tr[((size_t)a * TN + b) * NW];
            for (int i = 0; i < 256; ++i) {
                uint16_t h = ref_window(ring, i, 2, 0);
                uint32_t p = ref_prod(h);
                uint32_t bs = (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
                int tt = i >> 3, j = i & 7;
                int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
                int cc = (tt >> 2) + ((j & 4) ? 8 : 0);
                bs_mat[(size_t)(a * 16 + r) * NDIM + (b * 16 + cc)] = (uint16_t)bs;
                p_mat[(size_t)(a * 16 + r) * NDIM + (b * 16 + cc)]  = p;
            }
        }
    }

    for (int mode_x = 0; mode_x < 2; ++mode_x) {
        std::vector<float> x(KDIM);
        if (mode_x == 0) {
            // Hadamard-like zero-mean with 6x outliers (same distribution as sgemv_bench.cu)
            for (int i = 0; i < KDIM; ++i) {
                float g = 0;
                for (int t = 0; t < 4; ++t) g += (int)(rnd() % 2001 - 1000) / 1000.f;
                x[i] = g * 0.5f;
            }
            for (int b = 0; b < KDIM / 128; ++b) x[b * 128 + (rnd() % 128)] *= 6.f;
            printf("\n  --- Distribution A: Hadamard-like zero-mean x with 6x outliers ---\n");
        } else {
            // Adversarial all-positive x in [0.5, 1.5]
            for (int i = 0; i < KDIM; ++i) x[i] = 0.5f + (rnd() % 1000) / 1000.f;
            printf("\n  --- Distribution B: Adversarial all-positive x in [0.5, 1.5] ---\n");
        }

        // FP64 reference
        std::vector<double> yref(NDIM, 0.0);
        for (int c = 0; c < NDIM; ++c) {
            double s = 0.0;
            for (int r = 0; r < KDIM; ++r)
                s += (double)x[r] * ((1024.0 + bs_mat[(size_t)r * NDIM + c]) * KINV_d + (double)KC);
            yref[c] = s;
        }
        double ymax = 0.0, yrms = 0.0;
        for (double v : yref) { ymax = std::max(ymax, fabs(v)); yrms += v * v; }
        yrms = sqrt(yrms / NDIM);

        float sx_exact = 0.f;
        for (float v : x) sx_exact += v;

        auto report = [&](const char* name, const std::vector<float>& y) {
            double emax = 0.0, erms = 0.0;
            for (int c = 0; c < NDIM; ++c) {
                double e = fabs((double)y[c] - yref[c]);
                emax = std::max(emax, e);
                erms += e * e;
            }
            erms = sqrt(erms / NDIM);
            printf("  %-44s max_rel=%.3e  rms_rel=%.3e\n", name, emax / ymax, erms / yrms);
        };

        // C0: Baseline FP32 I2F + FFMA
        {
            std::vector<float> y(NDIM);
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int r = 0; r < KDIM; ++r) a = fmaf(x[r], (float)bs_mat[(size_t)r * NDIM + c], a);
                y[c] = fmaf(KINV, a, KOFF * sx_exact);
            }
            report("C0  Baseline FP32 I2F + FFMA", y);
        }
        // C1: Magic 0x4B000000 - 2^23 + FFMA
        {
            std::vector<float> y(NDIM);
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int r = 0; r < KDIM; ++r) {
                    float f = u32_as_f32(dp4a_u32(p_mat[(size_t)r * NDIM + c], 0x01010101u, 0x4B000000u)) - 8388608.f;
                    a = fmaf(x[r], f, a);
                }
                y[c] = fmaf(KINV, a, KOFF * sx_exact);
            }
            report("C1  Exact Magic (0x4B000000 - 2^23) + FFMA", y);
        }
        // C1c: Zero-Centered Exact Magic (0x4B000000 - (2^23+510)) + FFMA (same 1 FADD, zero cancel!)
        {
            std::vector<float> y(NDIM);
            const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int r = 0; r < KDIM; ++r) {
                    float f = u32_as_f32(dp4a_u32(p_mat[(size_t)r * NDIM + c], 0x01010101u, 0x4B000000u)) - 8389118.f;
                    a = fmaf(x[r], f, a);
                }
                y[c] = fmaf(KINV, a, KOFF_c * sx_exact);
            }
            report("C1c Zero-Centered Magic (-8389118.f) + FFMA", y);
        }
        // C2: Naive Folded 0x4B000000 (no FADD)
        {
            std::vector<float> y(NDIM);
            const float KOFF_c2 = KOFF - 8388608.f * KINV;
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int r = 0; r < KDIM; ++r) {
                    float f = u32_as_f32(dp4a_u32(p_mat[(size_t)r * NDIM + c], 0x01010101u, 0x4B000000u));
                    a = fmaf(x[r], f, a);
                }
                y[c] = fmaf(KINV, a, KOFF_c2 * sx_exact);
            }
            report("C2  Naive Folded 0x4B000000 (no FADD) [FAIL]", y);
        }
        // C3a: Max-Scale Folded 0x3F800000 + 255*bs (global fold, 0 FADD)
        {
            std::vector<float> y(NDIM);
            const float KINV_s = (float)((double)KINV * (8388608.0 / 255.0));
            const float KOFF_s = (float)((double)KOFF - (double)KINV * (8388608.0 / 255.0));
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int r = 0; r < KDIM; ++r) {
                    float f = u32_as_f32(dp4a_u32(p_mat[(size_t)r * NDIM + c], 0xFFFFFFFFu, 0x3F800000u));
                    a = fmaf(x[r], f, a);
                }
                y[c] = fmaf(KINV_s, a, KOFF_s * sx_exact);
            }
            report("C3a Folded 0x3F800000+255*bs (global fold)", y);
        }
        // C3b: Max-Scale Folded 0x3F800000 + 255*bs (per-tile cancel, 16 elems/tile)
        {
            std::vector<float> y(NDIM);
            const double alpha = 8388608.0 / 255.0;
            const float KINV_s = (float)(KINV_d * alpha);
            // w = KINV_s * (f - beta) where beta = 1.0 - KOFF / (KINV * alpha)
            const float beta = (float)(1.0 - KOFF_d / (KINV_d * alpha));
            std::vector<float> sxt(TK);
            for (int tk = 0; tk < TK; ++tk) {
                double s = 0.0;
                for (int r = 0; r < 16; ++r) s += (double)x[tk * 16 + r];
                sxt[tk] = (float)(-(double)beta * s);
            }
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int tk = 0; tk < TK; ++tk) {
                    float t = sxt[tk];
                    for (int r = 0; r < 16; ++r) {
                        int idx = tk * 16 + r;
                        float f = u32_as_f32(dp4a_u32(p_mat[(size_t)idx * NDIM + c], 0xFFFFFFFFu, 0x3F800000u));
                        t = fmaf(x[idx], f, t);
                    }
                    a += t;
                }
                y[c] = KINV_s * a;
            }
            report("C3b Folded 0x3F800000+255*bs (per-tile)", y);
        }
        // C4: Zero-Centered IDP4A(-510) + XMAD.S16.S16 (per-128 and per-16 scale)
        for (int blk : {128, 16}) {
            std::vector<int16_t> xq(KDIM);
            std::vector<float> qs(KDIM / blk);
            double sx_q = 0.0;
            for (int b = 0; b < KDIM / blk; ++b) {
                float am = 1e-30f;
                for (int i = 0; i < blk; ++i) am = fmaxf(am, fabsf(x[b * blk + i]));
                qs[b] = am / 32767.f;
                for (int i = 0; i < blk; ++i) {
                    xq[b * blk + i] = (int16_t)lrintf(x[b * blk + i] / qs[b]);
                    sx_q += (double)xq[b * blk + i] * qs[b];
                }
            }
            std::vector<float> y(NDIM);
            const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d); // -0.0018158!
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int b = 0; b < KDIM / blk; ++b) {
                    int32_t acc = 0;
                    for (int i = 0; i < blk; ++i) {
                        int r = b * blk + i;
                        int32_t bs_c = (int32_t)dp4a_u32(p_mat[(size_t)r * NDIM + c], 0x01010101u, (uint32_t)(-510));
                        acc += (int32_t)xq[r] * bs_c;
                    }
                    a = fmaf(qs[b], (float)acc, a);
                }
                y[c] = fmaf(KINV, a, KOFF_c * (float)sx_q);
            }
            char buf[64];
            snprintf(buf, sizeof(buf), "C4  IDP4A(-510)+XMAD.S16 (blk=%d)", blk);
            report(buf, y);
        }
        // C5: dp2a int16 (original unmatched sx blk=128 vs fixed matched sx_q blk=16)
        for (int fixed = 0; fixed <= 1; ++fixed) {
            const int blk = fixed ? 16 : 128;
            std::vector<uint32_t> x16(KDIM);
            std::vector<float> qs(KDIM / blk);
            double sx_q = 0.0;
            for (int b = 0; b < KDIM / blk; ++b) {
                float am = 1e-30f;
                for (int i = 0; i < blk; ++i) am = fmaxf(am, fabsf(x[b * blk + i]));
                qs[b] = am / 32767.f;
                for (int i = 0; i < blk; ++i) {
                    int q = (int)lrintf(x[b * blk + i] / qs[b]);
                    x16[b * blk + i] = (uint32_t)(q & 0xFFFFu) * 0x00010001u;
                    sx_q += (double)q * qs[b];
                }
            }
            std::vector<float> y(NDIM);
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int b = 0; b < KDIM / blk; ++b) {
                    int32_t i0 = 0, i2 = 0;
                    for (int i = 0; i < blk; ++i) {
                        int r = b * blk + i;
                        i0 = dp2a_lo_s32_u32(x16[r], p_mat[(size_t)r * NDIM + c], i0);
                        i2 = dp2a_hi_s32_u32(x16[r], p_mat[(size_t)r * NDIM + c], i2);
                    }
                    a = fmaf(qs[b], (float)(i0 + i2), a);
                }
                float sx_use = fixed ? (float)sx_q : sx_exact;
                y[c] = fmaf(KINV, a, KOFF * sx_use);
            }
            report(fixed ? "C5  dp2a-16 FIXED (matched sx_q, blk=16)" : "C5  dp2a-16 orig (unmatched sx, blk=128)", y);
        }
        // C6: dp4a int8 (original blk=128 vs fixed matched sx_q blk=16)
        for (int fixed = 0; fixed <= 1; ++fixed) {
            const int blk = fixed ? 16 : 128;
            std::vector<uint32_t> x8(KDIM);
            std::vector<float> qs(KDIM / blk);
            double sx_q = 0.0;
            for (int b = 0; b < KDIM / blk; ++b) {
                float am = 1e-30f;
                for (int i = 0; i < blk; ++i) am = fmaxf(am, fabsf(x[b * blk + i]));
                qs[b] = am / 127.f;
                for (int i = 0; i < blk; ++i) {
                    int q = (int)lrintf(x[b * blk + i] / qs[b]);
                    x8[b * blk + i] = (uint32_t)(q & 0xFFu) * 0x01010101u;
                    sx_q += (double)q * qs[b];
                }
            }
            std::vector<float> y(NDIM);
            for (int c = 0; c < NDIM; ++c) {
                float a = 0.f;
                for (int b = 0; b < KDIM / blk; ++b) {
                    int32_t acc = 0;
                    for (int i = 0; i < blk; ++i) {
                        int r = b * blk + i;
                        acc = dp4a_s32_u32(x8[r], p_mat[(size_t)r * NDIM + c], acc);
                    }
                    a = fmaf(qs[b], (float)acc, a);
                }
                float sx_use = fixed ? (float)sx_q : sx_exact;
                y[c] = fmaf(KINV, a, KOFF * sx_use);
            }
            report(fixed ? "C6  dp4a-8 FIXED (matched sx_q, blk=16) [FAIL]" : "C6  dp4a-8 orig (unmatched sx, blk=128) [FAIL]", y);
        }
    }
    return 0;
}
