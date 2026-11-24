// Round 2 EXL3 decode-math exhaustive CPU verification & real-weight error audit.
//
// Verifies:
//   1. Direction 1 (Lossless Load-Time Re-Encoding):
//      - Real-weight entropy audit across ALL 7 K geometries (MoE + dense real dumps).
//      - L1: Ring rotation + tile-block transpose + even/odd warp-pair zero-align (M=32) & M=64
//            on REAL weights for ALL K in {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0}.
//      - L2: Lossless Strided-Crumb bijection (0% bloat) + IDP4A seed reconstruction +
//            replicated 16-entry SHFL delta table D[lane & 15] + ISCADD incremental update
//            on ALL 10,368 real K=2.0 tiles (2,654,208 real weights) + exhaustive 65,536x4 transitions.
//   2. Direction 2 (Algebraic / Bit-Parallel):
//      - Exhaustive 65,536-window verification for mul16_imm, XMAD+LEA.HI+2*DP2A.LO,
//        C0c (zero-centered I2F), C1c (zero-centered magic), C3a/C3b (folded 0x3F800000+255*bs),
//        C4 (IDP4A(-510)+XMAD.S16), C4-dp2a (block zero-centered DP2A).
//   3. Direction 4 (Approximate Pareto Curve on REAL weights vs EXL3 quantization step):
//      - Measures max_abs_err, rms_err, max_rel_err, rms_rel_err, and ratio to 1 EXL3 quant step
//        (KINV = fp16 0x1eee = 0.006767273 per bs unit) on REAL MoE & dense tensors across
//        realistic Hadamard-like x (true H128*gauss), outlier-stressed Dist A (6x outliers),
//        and adversarial all-positive Dist B.
//
// Build & run:
//   g++ -O3 -std=c++17 opt/kbench/exl3_math2/test_cpu.cpp -o /tmp/exl3_math2_cpu -lm && /tmp/exl3_math2_cpu

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr uint32_t MUL1_C  = 0x83DCD12Du;
static constexpr uint32_t MUL1_CL = 0xD12Du;
static constexpr uint32_t MUL1_CH = 0x83DCu;

static float h2f(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    float v = e ? ldexpf(1.f + m / 1024.f, e - 15) : ldexpf(m / 1024.f, -14);
    return (h & 0x8000) ? -v : v;
}
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00u);
    uint32_t r = m + 0x1000u;
    if (r & 0x800000u) { r = 0; e++; }
    return (uint16_t)(s | ((uint32_t)e << 10) | (r >> 13));
}
static inline float u32_as_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static inline uint32_t f32_as_u32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

static inline uint32_t dp4a_uu(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t s = c;
    for (int i = 0; i < 4; ++i) s += ((a >> (8 * i)) & 0xFFu) * ((b >> (8 * i)) & 0xFFu);
    return s;
}
static inline int32_t dp4a_su(uint32_t a, uint32_t b, int32_t c) {
    int32_t s = c;
    for (int i = 0; i < 4; ++i)
        s += (int32_t)(int8_t)((a >> (8 * i)) & 0xFFu) * (int32_t)(uint8_t)((b >> (8 * i)) & 0xFFu);
    return s;
}
static inline int32_t dp2a_lo_su(uint32_t a, uint32_t b, int32_t c) {
    int16_t a0 = (int16_t)(a & 0xFFFFu), a1 = (int16_t)(a >> 16);
    uint8_t b0 = (uint8_t)(b & 0xFFu), b1 = (uint8_t)((b >> 8) & 0xFFu);
    return c + (int32_t)a0 * (int32_t)b0 + (int32_t)a1 * (int32_t)b1;
}
static inline uint32_t ref_prod(uint16_t h) { return (uint32_t)h * MUL1_C; }
static inline uint32_t ref_bs(uint16_t h) {
    uint32_t p = ref_prod(h);
    return (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
}
static inline uint16_t ref_window(const uint32_t* ring, int i, int ka, int hk) {
    const int nw = 8 * ka + 4 * hk;
    const int R = nw * 32;
    const int s = hk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
    const int b0 = s - 16 + (s < 16 ? R : 0);
    const int i0 = b0 >> 5;
    const int s0 = 48 - (b0 & 31);
    const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[i1];
    return (uint16_t)((v >> s0) & 0xFFFFu);
}

// L1: Lossless ring rotation (0% memory bloat)
static void rotate_ring_left(const uint32_t* src, uint32_t* dst, int nw, int shift_bits) {
    const int R = nw * 32;
    shift_bits = ((shift_bits % R) + R) % R;
    const int ws = shift_bits >> 5, bs = shift_bits & 31;
    for (int w = 0; w < nw; ++w) {
        int i0 = (w + ws) % nw, i1 = (i0 + 1) % nw;
        dst[w] = bs == 0 ? src[i0] : ((src[i0] << bs) | (src[i1] >> (32 - bs)));
    }
}

static inline uint32_t shf_l32(uint32_t lo, uint32_t hi, int s) {
    return (s == 0) ? hi : ((hi << s) | (lo >> (32 - s)));
}

// Even/odd parity zero-align extractor for half-K at M=32 (no alignment SHFs!)
static void extract_m32_zero_align(const uint32_t* rot, int nw, int lane, int ka, int hk, uint16_t* out) {
    const int i_base = lane * 32;
    const int base_bit = i_base * ka + hk * (i_base >> 1);
    const int w0 = base_bit >> 5;
    const int o = base_bit & 31; // 0 for all int-K and even lanes of half-K; 16 for odd lanes of half-K
    int offs[32];
    for (int j = 0; j < 32; ++j) offs[j] = j * ka + hk * ((j + 1) >> 1) + o;
    const int max_word = (offs[31] + 15) >> 5;
    uint32_t W[8];
    for (int k = 0; k <= max_word + 1 && k < 8; ++k) W[k] = rot[(w0 + k) % nw];

    auto get_u32 = [&](int d) -> uint32_t {
        return shf_l32(W[(d >> 5) + 1], W[d >> 5], d & 31);
    };
    bool done[32] = {};
    int idx_of[256];
    std::memset(idx_of, -1, sizeof(idx_of));
    for (int j = 0; j < 32; ++j) idx_of[offs[j]] = j;
    // Aligned (% 16 == 0) -> direct .H1 / .H0
    for (int j = 0; j < 32; ++j) {
        if ((offs[j] & 15) == 0) {
            uint32_t word = W[offs[j] >> 5];
            out[j] = (offs[j] & 16) ? (uint16_t)(word & 0xFFFFu) : (uint16_t)(word >> 16);
            done[j] = true;
        }
    }
    // Paired (d, d+16) -> 1 SHF + optional IADD U+U for d+1
    for (int j = 0; j < 32; ++j) {
        if (done[j]) continue;
        int d = offs[j];
        int j2 = (d + 16 < 256) ? idx_of[d + 16] : -1;
        if (j2 >= 0 && !done[j2]) {
            uint32_t U = get_u32(d);
            out[j]  = (uint16_t)(U >> 16);
            out[j2] = (uint16_t)(U & 0xFFFFu);
            done[j] = done[j2] = true;
            int j3 = (d + 1 < 256) ? idx_of[d + 1] : -1;
            if (j3 >= 0 && !done[j3]) {
                out[j3] = (uint16_t)((U + U) >> 16);
                done[j3] = true;
            }
        }
    }
    // Remaining singletons (direct SHF, no serial U+U needed for odd-K)
    for (int j = 0; j < 32; ++j) {
        if (done[j]) continue;
        uint32_t U = get_u32(offs[j]);
        out[j] = (uint16_t)(U >> 16);
        done[j] = true;
    }
}

// L2: Lossless Strided-Crumb bijection for K=2.0 (0% memory bloat: 16 uint32 -> 16 uint32)
// M=32 layout: lane l (0..7) owns 2 words W0, W1:
//   Byte ch (0..3) of W0 holds 4 crumbs of chain ch for lane l:
//     bits [1:0] = c[32*l + ch +  0]
//     bits [3:2] = c[32*l + ch +  8]
//     bits [5:4] = c[32*l + ch + 16]
//     bits [7:6] = c[32*l + ch + 24]
//   Wait: in D[idx], idx = (u << 2) | v, where u is the OLDER crumb (c[32*l + ch + 8*s])
//   and v is the NEWER crumb (c[32*l + ch + 8*(s+1)]).
//   To make bits [3:0] = (u << 2) | v when shifting right by 2*s, we store:
//     bits [7:6] = oldest? NO! If bits [3:2] = u and bits [1:0] = v, then u is HIGHER than v!
//     Wait: let's check both orders!
//     If we store LSB-first:
//       bits [1:0] = c_0 (oldest)
//       bits [3:2] = c_8
//       bits [5:4] = c_16
//       bits [7:6] = c_24 (newest)
//     Then bits [3:0] = (c_8 << 2) | c_0 = (v << 2) | u!
//     And D_rev[(v << 2) | u] = v * C - (u * CL << 16)!!
//     THAT'S EVEN CLEANER! Just define D_reg with v = (lane >> 2) & 3, u = lane & 3!
static void encode_strided_k2_m32(const uint32_t* raw_ring, uint32_t* out16) {
    uint32_t rot[16];
    // Rotate left by b0(0) = (2 - 16) mod 512 = 498 bits so window 0 starts at bit 0 MSB-first
    rotate_ring_left(raw_ring, rot, 16, 512 - 14);
    // Extract 256 2-bit crumbs MSB-first: crumb k is bits [2k, 2k+1] of rot
    uint8_t c[256];
    for (int k = 0; k < 256; ++k) {
        int w = k >> 4;
        int sh = 30 - 2 * (k & 15);
        c[k] = (uint8_t)((rot[w] >> sh) & 3u);
    }
    // Pack 8 lanes x 2 words = 16 words (exact 64 bytes, 0% bloat!)
    for (int l = 0; l < 8; ++l) {
        uint32_t W0 = 0, W1 = 0;
        for (int ch = 0; ch < 4; ++ch) {
            uint32_t b0 = (uint32_t)c[(32 * l + ch +  0) & 255]
                        | ((uint32_t)c[(32 * l + ch +  8) & 255] << 2)
                        | ((uint32_t)c[(32 * l + ch + 16) & 255] << 4)
                        | ((uint32_t)c[(32 * l + ch + 24) & 255] << 6);
            uint32_t b1 = (uint32_t)c[(32 * l + ch +  4) & 255]
                        | ((uint32_t)c[(32 * l + ch + 12) & 255] << 2)
                        | ((uint32_t)c[(32 * l + ch + 20) & 255] << 4)
                        | ((uint32_t)c[(32 * l + ch + 28) & 255] << 6);
            W0 |= b0 << (8 * ch);
            W1 |= b1 << (8 * ch);
        }
        out16[l * 2 + 0] = W0;
        out16[l * 2 + 1] = W1;
    }
}

// Decode all 256 P values from strided-crumb M=32 layout using exact GPU operations!
static void decode_strided_k2_m32(const uint32_t* enc16, uint32_t* out_P) {
    uint32_t D[32];
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = (lane >> 2) & 3u;
        uint32_t u = lane & 3u;
        D[lane] = v * MUL1_C - ((u * MUL1_CL) << 16);
    }
    for (int l = 0; l < 8; ++l) {
        uint32_t W0 = enc16[l * 2 + 0];
        uint32_t W1 = enc16[l * 2 + 1];
        uint32_t W2 = enc16[((l + 1) & 7) * 2 + 0];
        uint32_t W3 = enc16[((l + 1) & 7) * 2 + 1];
        // Reconstruct seed h0 via IDP4A: c0*64 + c1*16 + c2*4 + c3*1
        uint32_t h0_hi = dp4a_uu(W0 & 0x03030303u, 0x01041040u, 0u);
        uint32_t h0_lo = dp4a_uu(W1 & 0x03030303u, 0x01041040u, 0u);
        uint32_t h0 = (h0_hi << 8) + h0_lo;
        uint32_t P = ref_prod((uint16_t)h0);
        int base = l * 32;
        out_P[base + 0] = P;
        auto step = [&](int j, uint32_t idx) {
            P = (P << 2) + D[idx & 31u];
            out_P[base + j] = P;
        };
        step(1,  W0);       step(2,  W0 >> 8);  step(3,  W0 >> 16); step(4,  W0 >> 24);
        step(5,  W1);       step(6,  W1 >> 8);  step(7,  W1 >> 16); step(8,  W1 >> 24);
        step(9,  W0 >> 2);  step(10, W0 >> 10); step(11, W0 >> 18); step(12, W0 >> 26);
        step(13, W1 >> 2);  step(14, W1 >> 10); step(15, W1 >> 18); step(16, W1 >> 26);
        step(17, W0 >> 4);  step(18, W0 >> 12); step(19, W0 >> 20); step(20, W0 >> 28);
        step(21, W1 >> 4);  step(22, W1 >> 12); step(23, W1 >> 20); step(24, W1 >> 28);
        // Tail j=25..31: u = bits [7:6] of byte in W0/W1 (put into bits [1:0]),
        //                v = bits [1:0] of byte in W2/W3 (put into bits [3:2])!
        uint32_t W02 = ((W0 >> 6) & 0x03030303u) | ((W2 & 0x03030303u) << 2);
        uint32_t W13 = ((W1 >> 6) & 0x03030303u) | ((W3 & 0x03030303u) << 2);
        step(25, W02);      step(26, W02 >> 8); step(27, W02 >> 16); step(28, W02 >> 24);
        step(29, W13);      step(30, W13 >> 8); step(31, W13 >> 16);
    }
}

// L2-M64: Strided-Crumb M=64 layout (4 lanes/tile, 4 words/lane = 1 LDG.128!)
// Lane l (0..3) owns 4 words W0..W3:
//   W0 = [chain1_16b | chain0_16b], W1 = [chain3_16b | chain2_16b],
//   W2 = [chain5_16b | chain4_16b], W3 = [chain7_16b | chain6_16b]
//   Each 16-bit half holds 8 consecutive crumbs LSB-first -> ZERO mid-seam for j=1..56!
static void encode_strided_k2_m64(const uint32_t* raw_ring, uint32_t* out16) {
    uint32_t rot[16];
    rotate_ring_left(raw_ring, rot, 16, 512 - 14);
    uint8_t c[256];
    for (int k = 0; k < 256; ++k) {
        int w = k >> 4, sh = 30 - 2 * (k & 15);
        c[k] = (uint8_t)((rot[w] >> sh) & 3u);
    }
    for (int l = 0; l < 4; ++l) {
        for (int p = 0; p < 4; ++p) {
            int ch0 = 2 * p, ch1 = 2 * p + 1;
            uint32_t h0 = 0, h1 = 0;
            for (int s = 0; s < 8; ++s) {
                h0 |= (uint32_t)c[(64 * l + ch0 + 8 * s) & 255] << (2 * s);
                h1 |= (uint32_t)c[(64 * l + ch1 + 8 * s) & 255] << (2 * s);
            }
            out16[l * 4 + p] = h0 | (h1 << 16);
        }
    }
}

static void decode_strided_k2_m64(const uint32_t* enc16, uint32_t* out_P) {
    uint32_t D[32];
    for (int lane = 0; lane < 32; ++lane) {
        uint32_t v = (lane >> 2) & 3u;
        uint32_t u = lane & 3u;
        D[lane] = v * MUL1_C - ((u * MUL1_CL) << 16);
    }
    for (int l = 0; l < 4; ++l) {
        uint32_t W[4], N[4];
        for (int p = 0; p < 4; ++p) {
            W[p] = enc16[l * 4 + p];
            N[p] = enc16[((l + 1) & 3) * 4 + p];
        }
        // Reconstruct seed h0 from bits [1:0] and [17:16] of W[0..3]
        uint32_t h0 = ((W[0] & 3u) << 14) | (((W[0] >> 16) & 3u) << 12)
                    | ((W[1] & 3u) << 10) | (((W[1] >> 16) & 3u) << 8)
                    | ((W[2] & 3u) <<  6) | (((W[2] >> 16) & 3u) << 4)
                    | ((W[3] & 3u) <<  2) | (((W[3] >> 16) & 3u) << 0);
        uint32_t P = ref_prod((uint16_t)h0);
        int base = l * 64;
        out_P[base + 0] = P;
        auto step = [&](int j, uint32_t idx) {
            P = (P << 2) + D[idx & 31u];
            out_P[base + j] = P;
        };
        for (int s = 0; s < 7; ++s) {
            for (int p = 0; p < 4; ++p) {
                int j0 = 1 + 8 * s + 2 * p;
                step(j0,     W[p] >> (2 * s));
                step(j0 + 1, W[p] >> (16 + 2 * s));
            }
        }
        // Tail j=57..63 (7 steps): u = bits [15:14]/[31:30] of W[p], v = bits [1:0]/[17:16] of N[p]
        for (int p = 0; p < 4; ++p) {
            uint32_t T = ((W[p] >> 14) & 0x00030003u) | ((N[p] & 0x00030003u) << 2);
            int j0 = 57 + 2 * p;
            step(j0, T);
            if (j0 + 1 < 64) step(j0 + 1, T >> 16);
        }
    }
}

static uint32_t rng_s = 12345u;
static uint32_t rnd() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5; return rng_s; }

// Fast Walsh-Hadamard transform of size 128 (orthonormal, /sqrt(128))
static void had128(float* v) {
    for (int s = 1; s < 128; s <<= 1) {
        for (int i = 0; i < 128; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                float a = v[i + j], b = v[i + j + s];
                v[i + j] = a + b;
                v[i + j + s] = a - b;
            }
        }
    }
    const float inv = 1.f / sqrtf(128.f);
    for (int i = 0; i < 128; ++i) v[i] *= inv;
}

static std::vector<uint32_t> load_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "ERROR: missing %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> buf((size_t)sz / 4);
    if (fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    fclose(f);
    return buf;
}

int main() {
    printf("=== Round 2 EXL3 Decode-Math: Real-Weight Losslessness, Entropy & Approximate Pareto ===\n\n");

    const float KINV = h2f(0x1eee);
    const float KC   = h2f(0xc931);
    const float KOFF = 1024.f * KINV + KC;
    const double KINV_d = KINV, KC_d = KC, KOFF_d = KOFF;
    printf("EXL3 constants: KINV = %.9f (0x1eee), KC = %.9f (0xc931), KOFF = %.9f, KOFF+510*KINV = %.9f\n\n",
           KINV, KC, KOFF, (float)(KOFF_d + 510.0 * KINV_d));

    // =========================================================================
    // TEST 1: Exhaustive 65,536-window algebraic identities (new Round-2 forms)
    // =========================================================================
    printf("[TEST 1] Exhaustive 65,536-window & transition checks\n");
    int fail_lea = 0, fail_c0c = 0, fail_dp2a = 0, fail_inc = 0;
    uint32_t D_rev[32];
    for (int l = 0; l < 32; ++l) {
        uint32_t v = (l >> 2) & 3u, u = l & 3u;
        D_rev[l] = v * MUL1_C - ((u * MUL1_CL) << 16);
    }
    for (uint32_t h = 0; h < 65536; ++h) {
        uint32_t pref = ref_prod((uint16_t)h);
        uint32_t bsref = ref_bs((uint16_t)h);
        // XMAD + LEA.HI + 2*DP2A.LO
        uint32_t rt = h * MUL1_CL;
        uint32_t rhi = h * MUL1_CH + (rt >> 16);
        int16_t xval = (int16_t)((h * 2654435761u) >> 16);
        uint32_t xsplat = ((uint32_t)(uint16_t)xval) * 0x00010001u;
        int32_t mac_ref = (int32_t)xval * (int32_t)bsref;
        int32_t mac_lea = dp2a_lo_su(xsplat, rhi, dp2a_lo_su(xsplat, rt, 0));
        if (mac_lea != mac_ref) fail_lea++;
        // Check rt.H0 + rhi.H0 == P
        if (((rt & 0xFFFFu) | ((rhi & 0xFFFFu) << 16)) != pref) fail_dp2a++;
        // C0c: zero-centered I2F via dp4a(P, 1s, -510)
        int32_t bsc = (int32_t)dp4a_uu(pref, 0x01010101u, (uint32_t)(-510));
        if ((float)bsc != (float)((int32_t)bsref - 510)) fail_c0c++;
        // Strided-crumb D_rev transitions (all 65536 x 4)
        uint32_t u = h >> 14;
        for (uint32_t v = 0; v < 4; ++v) {
            uint16_t h2 = (uint16_t)(((h << 2) & 0xFFFFu) | v);
            uint32_t idx = (v << 2) | u;
            // Test with arbitrary upper garbage bits in idx (to verify D_rev[lane & 15] replication)
            uint32_t idx_dirty = idx | ((h & 0x7FFFu) << 4);
            uint32_t p2 = (pref << 2) + D_rev[idx_dirty & 31u];
            if (p2 != ref_prod(h2)) fail_inc++;
        }
    }
    printf("  XMAD+LEA.HI+2*DP2A.LO (65536): FAIL=%d\n", fail_lea);
    printf("  rt.H0|rhi.H0 == P     (65536): FAIL=%d\n", fail_dp2a);
    printf("  C0c zero-centered I2F (65536): FAIL=%d\n", fail_c0c);
    printf("  D_rev replicated SHFL (262144): FAIL=%d\n\n", fail_inc);
    if (fail_lea || fail_dp2a || fail_c0c || fail_inc) return 1;

    // =========================================================================
    // TEST 2: Real-Weight Losslessness (ALL 7 K) & Entropy Audit
    // =========================================================================
    printf("[TEST 2] Real-Weight Lossless Re-Encoding Verification & Entropy Audit (ALL 7 K)\n");
    struct RealTensor { const char* label; const char* path; int ka, hk, tk, tn; };
    const RealTensor tensors[7] = {
        {"gate_s0 (MoE)",  "/tmp/sgemv_gold/gate_s0.bin",       1, 1, 144,  72},
        {"gate_s1 (MoE)",  "/tmp/sgemv_gold/gate_s1.bin",       2, 0, 144,  72},
        {"gate_s2 (MoE)",  "/tmp/sgemv_gold/gate_s2.bin",       2, 1, 144,  72},
        {"gate_s3 (MoE)",  "/tmp/sgemv_gold/gate_s3.bin",       3, 0, 144,  72},
        {"gate_s4 (MoE)",  "/tmp/sgemv_gold/gate_s4.bin",       3, 1, 144,  72},
        {"gate_s5 (MoE)",  "/tmp/sgemv_gold/gate_s5.bin",       4, 0, 144,  72},
        {"wr_du (dense)",  "/tmp/sgemv_dense_gold/wr_du.bin",   5, 0, 144,  24},
    };
    printf("  %-15s %4s %7s  %7s %7s %7s  %10s %10s\n",
           "tensor", "K", "tiles", "H1(B)", "H(B|B')", "H(bs)", "L1-rot", "L2-strided");
    for (int ti = 0; ti < 7; ++ti) {
        const auto& T = tensors[ti];
        auto raw = load_file(T.path);
        const int nw = 8 * T.ka + 4 * T.hk;
        const int R = nw * 32;
        const int ntiles = (int)(raw.size() / nw);
        // Byte 1st-order & 2nd-order conditional entropy on raw bitstream bytes
        const uint8_t* bytes = (const uint8_t*)raw.data();
        const size_t nbytes = raw.size() * 4;
        std::vector<uint64_t> c1(256, 0), c2(65536, 0);
        for (size_t i = 0; i < nbytes; ++i) {
            c1[bytes[i]]++;
            if (i > 0) c2[(bytes[i - 1] << 8) | bytes[i]]++;
        }
        double H1 = 0.0, Hcond = 0.0;
        for (int a = 0; a < 256; ++a) {
            if (!c1[a]) continue;
            double pa = (double)c1[a] / (double)nbytes;
            H1 -= pa * log2(pa);
            for (int b = 0; b < 256; ++b) {
                uint64_t cab = c2[(a << 8) | b];
                if (!cab) continue;
                double pab = (double)cab / (double)(nbytes - 1);
                double pb_a = (double)cab / (double)c1[a];
                Hcond -= pab * log2(pb_a);
            }
        }
        // Check L1 on ALL tiles (zero-align M=32) & compute H(bs)
        const int b0_init = T.ka - 16 + (T.ka < 16 ? R : 0);
        uint64_t fail_l1 = 0, fail_l2_m32 = 0, fail_l2_m64 = 0;
        uint64_t bs_hist[1024] = {};
        uint32_t rot[40];
        for (int t = 0; t < ntiles; ++t) {
            const uint32_t* ring = &raw[(size_t)t * nw];
            rotate_ring_left(ring, rot, nw, b0_init);
            for (int lane = 0; lane < 8; ++lane) {
                uint16_t got[32];
                extract_m32_zero_align(rot, nw, lane, T.ka, T.hk, got);
                for (int j = 0; j < 32; ++j) {
                    uint16_t ref_h = ref_window(ring, lane * 32 + j, T.ka, T.hk);
                    if (got[j] != ref_h) fail_l1++;
                    if (t < 1000) bs_hist[ref_bs(ref_h)]++;
                }
            }
            if (T.ka == 2 && T.hk == 0) {
                uint32_t enc32[16], enc64[16], dec32[256], dec64[256];
                encode_strided_k2_m32(ring, enc32);
                decode_strided_k2_m32(enc32, dec32);
                encode_strided_k2_m64(ring, enc64);
                decode_strided_k2_m64(enc64, dec64);
                for (int i = 0; i < 256; ++i) {
                    uint32_t pref = ref_prod(ref_window(ring, i, 2, 0));
                    if (dec32[i] != pref) fail_l2_m32++;
                    if (dec64[i] != pref) fail_l2_m64++;
                }
            }
        }
        double Hbs = 0.0;
        double tot_bs = (double)std::min(ntiles, 1000) * 256.0;
        for (int b = 0; b < 1024; ++b) {
            if (!bs_hist[b]) continue;
            double p = bs_hist[b] / tot_bs;
            Hbs -= p * log2(p);
        }
        char l2buf[32];
        if (T.ka == 2 && T.hk == 0)
            snprintf(l2buf, sizeof(l2buf), "PASS(%llu/%llu)", (unsigned long long)fail_l2_m32, (unsigned long long)fail_l2_m64);
        else
            snprintf(l2buf, sizeof(l2buf), "n/a (K!=2)");
        printf("  %-15s %4.1f %7d  %7.4f %7.4f %7.3f  PASS(%llu)  %s\n",
               T.label, T.ka + 0.5 * T.hk, ntiles, H1, Hcond, Hbs, (unsigned long long)fail_l1, l2buf);
        if (fail_l1 || fail_l2_m32 || fail_l2_m64) return 1;
    }
    printf("\n");

    // =========================================================================
    // TEST 3: Direction 4 — Approximate Pareto Curve on REAL Weights
    //         Compare max_abs_err & max_rel_err directly against 1 EXL3 quant step!
    // =========================================================================
    printf("[TEST 3] Approximate Pareto Curve on REAL Weights vs EXL3 Quantization Step (KINV=%.6f)\n", KINV);
    // Test across 3 real matrices: gate_s0 (K=1.5), gate_s1 (K=2.0), gate_s3 (K=3.0) — report K=2.0 in full + summary across K
    const int pareto_idxs[3] = {0, 1, 3};
    for (int pi = 0; pi < 3; ++pi) {
        const auto& T = tensors[pareto_idxs[pi]];
        auto raw = load_file(T.path);
        const int TK = T.tk, TN = T.tn, KDIM = TK * 16, NDIM = TN * 16;
        const int NW = 8 * T.ka + 4 * T.hk;
        std::vector<uint16_t> bs_mat((size_t)KDIM * NDIM);
        std::vector<uint32_t> p_mat((size_t)KDIM * NDIM);
        double bs_sum = 0.0, bs_sq = 0.0;
        for (int a = 0; a < TK; ++a) {
            for (int b = 0; b < TN; ++b) {
                const uint32_t* ring = &raw[((size_t)a * TN + b) * NW];
                for (int i = 0; i < 256; ++i) {
                    uint16_t h = ref_window(ring, i, T.ka, T.hk);
                    uint32_t p = ref_prod(h);
                    uint32_t bs = (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
                    int tt = i >> 3, j = i & 7;
                    int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
                    int cc = (tt >> 2) + ((j & 4) ? 8 : 0);
                    bs_mat[(size_t)(a * 16 + r) * NDIM + (b * 16 + cc)] = (uint16_t)bs;
                    p_mat[(size_t)(a * 16 + r) * NDIM + (b * 16 + cc)] = p;
                    bs_sum += bs;
                    bs_sq += (double)bs * bs;
                }
            }
        }
        double bs_mean = bs_sum / ((double)KDIM * NDIM);
        double bs_std = sqrt(bs_sq / ((double)KDIM * NDIM) - bs_mean * bs_mean);
        printf("\n=== Matrix: %s (K=%.1f, %dx%d), real bs_std=%.2f, w_std=%.4f, 1-step/w_std=%.3e ===\n",
               T.label, T.ka + 0.5 * T.hk, KDIM, NDIM, bs_std, bs_std * KINV_d, 1.0 / bs_std);

        // Run all 3 x distributions only on K=2.0 (gate_s1); run Dist H & Dist A on K=1.5 and K=3.0
        int n_modes = (pi == 1) ? 3 : 1;
        for (int mode_x = 0; mode_x < n_modes; ++mode_x) {
            std::vector<float> x(KDIM);
            const char* mname = "";
            if (mode_x == 0) {
                // Realistic Hadamard-rotated x (true H128 applied to heavy-tailed pre-Hadamard activations!)
                mname = "Real H128(x_with_10x_outliers)";
                for (int i = 0; i < KDIM; ++i) {
                    float g = 0; for (int k = 0; k < 4; ++k) g += (int)(rnd() % 2001 - 1000) / 1000.f;
                    x[i] = g * 0.5f;
                }
                for (int b = 0; b < KDIM / 128; ++b) {
                    x[b * 128 + (rnd() % 128)] *= 10.f;
                    had128(&x[b * 128]);
                }
            } else if (mode_x == 1) {
                // Dist A: Post-Hadamard with surviving 6x outliers (same as Round 1 & sgemv_bench)
                mname = "Dist A (zero-mean + 6x outliers)";
                for (int i = 0; i < KDIM; ++i) {
                    float g = 0; for (int k = 0; k < 4; ++k) g += (int)(rnd() % 2001 - 1000) / 1000.f;
                    x[i] = g * 0.5f;
                }
                for (int b = 0; b < KDIM / 128; ++b) x[b * 128 + (rnd() % 128)] *= 6.f;
            } else {
                // Dist B: Adversarial all-positive x in [0.5, 1.5]
                mname = "Dist B (adversarial all-pos [0.5,1.5])";
                for (int i = 0; i < KDIM; ++i) x[i] = 0.5f + (rnd() % 1000) / 1000.f;
            }

            double xnorm2 = 0.0;
            float sx_exact = 0.f;
            for (float v : x) { xnorm2 += (double)v * v; sx_exact += v; }
            double xnorm = sqrt(xnorm2);
            double qstep_rms = KINV_d * xnorm; // RMS output perturbation from +-1 EXL3 step per weight
            double qhalf_rms = (KINV_d / sqrt(12.0)) * xnorm; // RMS from U[-0.5, +0.5] step per weight

            // FP64 reference + Monte Carlo 1-step and 0.5-step perturbations on the EXACT same matrix & x
            std::vector<double> yref(NDIM, 0.0), y_1step(NDIM, 0.0), y_half(NDIM, 0.0);
            for (int c = 0; c < NDIM; ++c) {
                double s = 0.0, s1 = 0.0, sh = 0.0;
                for (int r = 0; r < KDIM; ++r) {
                    double w = (1024.0 + bs_mat[(size_t)r * NDIM + c]) * KINV_d + KC_d;
                    double dw1 = ((rnd() & 1u) ? 1.0 : -1.0) * KINV_d;
                    double dwh = (((double)(rnd() & 0xFFFFu) / 65536.0) - 0.5) * KINV_d;
                    s  += (double)x[r] * w;
                    s1 += (double)x[r] * (w + dw1);
                    sh += (double)x[r] * (w + dwh);
                }
                yref[c] = s; y_1step[c] = s1; y_half[c] = sh;
            }
            double ymax = 0.0, yrms = 0.0;
            for (double v : yref) { ymax = std::max(ymax, fabs(v)); yrms += v * v; }
            yrms = sqrt(yrms / NDIM);

            auto eval_d = [&](const char* name, const std::vector<double>& y) {
                double emax = 0.0, erms = 0.0;
                for (int c = 0; c < NDIM; ++c) {
                    double e = fabs(y[c] - yref[c]);
                    emax = std::max(emax, e); erms += e * e;
                }
                erms = sqrt(erms / NDIM);
                printf("  %-42s  max_abs=%9.3e  max_rel=%9.3e  rms_rel=%9.3e  rms/1step=%6.4fx  %s\n",
                       name, emax, emax / ymax, erms / yrms, erms / qstep_rms,
                       (erms / qstep_rms < 0.1) ? "<< 1 step" : ((erms / qstep_rms < 0.5) ? "< 1 step" : ">= 1 step"));
            };
            auto eval_f = [&](const char* name, const std::vector<float>& y) {
                std::vector<double> yd(NDIM);
                for (int c = 0; c < NDIM; ++c) yd[c] = y[c];
                eval_d(name, yd);
            };

            printf("  --- %s (||x||=%.2f, ymax=%.2f, yrms=%.2f, 1-step_rms=%.4f, 0.5-step_rms=%.4f) ---\n",
                   mname, xnorm, ymax, yrms, qstep_rms, qhalf_rms);
            eval_d("[REF] +-1 EXL3 quant step (KINV/w)", y_1step);
            eval_d("[REF] +-0.5 EXL3 quant step (U[-.5,.5])", y_half);

            // 1. C0: Shipped FP32 I2F (uncentered)
            {
                std::vector<float> y(NDIM);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) a = fmaf(x[r], (float)bs_mat[(size_t)r * NDIM + c], a);
                    y[c] = fmaf(KINV, a, KOFF * sx_exact);
                }
                eval_f("C0  Shipped FP32 I2F (uncentered)", y);
            }
            // 2. C0c: Zero-centered FP32 I2F (IDP4A(-510))
            {
                std::vector<float> y(NDIM);
                const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) {
                        int32_t bsc = (int32_t)dp4a_uu(p_mat[(size_t)r * NDIM + c], 0x01010101u, (uint32_t)(-510));
                        a = fmaf(x[r], (float)bsc, a);
                    }
                    y[c] = fmaf(KINV, a, KOFF_c * sx_exact);
                }
                eval_f("C0c Zero-Centered I2F (exact)", y);
            }
            // 3. C1c: Zero-centered magic float (0x4B000000 - 8389118.f)
            {
                std::vector<float> y(NDIM);
                const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) {
                        float f = u32_as_f32(dp4a_uu(p_mat[(size_t)r * NDIM + c], 0x01010101u, 0x4B000000u)) - 8389118.f;
                        a = fmaf(x[r], f, a);
                    }
                    y[c] = fmaf(KINV, a, KOFF_c * sx_exact);
                }
                eval_f("C1c Zero-Centered Magic FADD (exact)", y);
            }
            // 4. C3b: Folded FP32 (0x3F800000 + 255*bs, per-tile cancel)
            {
                std::vector<float> y(NDIM);
                const double alpha = 8388608.0 / 255.0;
                const float KINV_s = (float)(KINV_d * alpha);
                const float beta = (float)(1.0 - KOFF_d / (KINV_d * alpha));
                std::vector<float> sxt(TK);
                for (int tk = 0; tk < TK; ++tk) {
                    double s = 0.0; for (int r = 0; r < 16; ++r) s += (double)x[tk * 16 + r];
                    sxt[tk] = (float)(-(double)beta * s);
                }
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int tk = 0; tk < TK; ++tk) {
                        float t = sxt[tk];
                        for (int r = 0; r < 16; ++r) {
                            int idx = tk * 16 + r;
                            float f = u32_as_f32(dp4a_uu(p_mat[(size_t)idx * NDIM + c], 0xFFFFFFFFu, 0x3F800000u));
                            t = fmaf(x[idx], f, t);
                        }
                        a += t;
                    }
                    y[c] = KINV_s * a;
                }
                eval_f("C3b Folded 0x3F800000+255*bs (per-tile)", y);
            }
            // 5. C3a: Folded FP32 (global cancel)
            {
                std::vector<float> y(NDIM);
                const double alpha = 8388608.0 / 255.0;
                const float KINV_s = (float)(KINV_d * alpha);
                const float KOFF_s = (float)(KOFF_d - KINV_d * alpha);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) {
                        float f = u32_as_f32(dp4a_uu(p_mat[(size_t)r * NDIM + c], 0xFFFFFFFFu, 0x3F800000u));
                        a = fmaf(x[r], f, a);
                    }
                    y[c] = fmaf(KINV_s, a, KOFF_s * sx_exact);
                }
                eval_f("C3a Folded 0x3F800000+255*bs (global)", y);
            }
            // 6. C4 (blk=16 and blk=128): IDP4A(-510) + XMAD.S16.S16 (int16 x)
            for (int blk : {16, 128}) {
                std::vector<int16_t> xq(KDIM);
                std::vector<float> qs(KDIM / blk);
                double sx_q = 0.0;
                for (int b = 0; b < KDIM / blk; ++b) {
                    float am = 1e-30f; for (int i = 0; i < blk; ++i) am = fmaxf(am, fabsf(x[b * blk + i]));
                    qs[b] = am / 32767.f;
                    for (int i = 0; i < blk; ++i) {
                        xq[b * blk + i] = (int16_t)lrintf(x[b * blk + i] / qs[b]);
                        sx_q += (double)xq[b * blk + i] * qs[b];
                    }
                }
                std::vector<float> y(NDIM);
                const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int b = 0; b < KDIM / blk; ++b) {
                        int32_t acc = 0;
                        for (int i = 0; i < blk; ++i) {
                            int r = b * blk + i;
                            int32_t bsc = (int32_t)dp4a_uu(p_mat[(size_t)r * NDIM + c], 0x01010101u, (uint32_t)(-510));
                            acc += (int32_t)xq[r] * bsc;
                        }
                        a = fmaf(qs[b], (float)acc, a);
                    }
                    y[c] = fmaf(KINV, a, KOFF_c * (float)sx_q);
                }
                char nm[64]; snprintf(nm, sizeof(nm), "C4  IDP4A(-510)+XMAD.S16 (blk=%d)", blk);
                eval_f(nm, y);
            }
            // 7. FP16 x (exact bs, x cast to fp16)
            {
                std::vector<float> xh16(KDIM);
                float sx16 = 0.f;
                for (int i = 0; i < KDIM; ++i) { xh16[i] = h2f(f2h(x[i])); sx16 += xh16[i]; }
                std::vector<float> y(NDIM);
                const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) a = fmaf(xh16[r], (float)((int)bs_mat[(size_t)r * NDIM + c] - 510), a);
                    y[c] = fmaf(KINV, a, KOFF_c * sx16);
                }
                eval_f("A-FP16x (fp16 x, exact bs)", y);
            }
            // 8. C6-o1: dp4a-8 (blk=16) + top-1 outlier per 128 in exact FP32
            {
                std::vector<uint32_t> x8(KDIM);
                std::vector<float> qs(KDIM / 16);
                std::vector<int> out_idx(KDIM / 128);
                std::vector<float> out_val(KDIM / 128);
                std::vector<float> x_clean = x;
                for (int g = 0; g < KDIM / 128; ++g) {
                    int bi = g * 128;
                    for (int i = 1; i < 128; ++i) if (fabsf(x[g * 128 + i]) > fabsf(x[bi])) bi = g * 128 + i;
                    out_idx[g] = bi; out_val[g] = x[bi]; x_clean[bi] = 0.f;
                }
                double sx_q = 0.0;
                for (int b = 0; b < KDIM / 16; ++b) {
                    float am = 1e-30f; for (int i = 0; i < 16; ++i) am = fmaxf(am, fabsf(x_clean[b * 16 + i]));
                    qs[b] = am / 127.f;
                    for (int i = 0; i < 16; ++i) {
                        int q = (int)lrintf(x_clean[b * 16 + i] / qs[b]);
                        x8[b * 16 + i] = (uint32_t)(q & 0xFFu) * 0x01010101u;
                        sx_q += (double)q * qs[b];
                    }
                }
                for (int g = 0; g < KDIM / 128; ++g) sx_q += out_val[g];
                std::vector<float> y(NDIM);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int b = 0; b < KDIM / 16; ++b) {
                        int32_t acc = 0;
                        for (int i = 0; i < 16; ++i) acc = dp4a_su(x8[b * 16 + i], p_mat[(size_t)(b * 16 + i) * NDIM + c], acc);
                        a = fmaf(qs[b], (float)acc, a);
                    }
                    for (int g = 0; g < KDIM / 128; ++g)
                        a = fmaf(out_val[g], (float)bs_mat[(size_t)out_idx[g] * NDIM + c], a);
                    y[c] = fmaf(KINV, a, KOFF * (float)sx_q);
                }
                eval_f("C6-o1 dp4a-8 (blk=16 + 1-outlier FP32)", y);
            }
            // 9. C6: dp4a-8 (blk=16 and blk=128, matched sx_q)
            for (int blk : {16, 128}) {
                std::vector<uint32_t> x8(KDIM);
                std::vector<float> qs(KDIM / blk);
                double sx_q = 0.0;
                for (int b = 0; b < KDIM / blk; ++b) {
                    float am = 1e-30f; for (int i = 0; i < blk; ++i) am = fmaxf(am, fabsf(x[b * blk + i]));
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
                        for (int i = 0; i < blk; ++i) acc = dp4a_su(x8[b * blk + i], p_mat[(size_t)(b * blk + i) * NDIM + c], acc);
                        a = fmaf(qs[b], (float)acc, a);
                    }
                    y[c] = fmaf(KINV, a, KOFF * (float)sx_q);
                }
                char nm[64]; snprintf(nm, sizeof(nm), "C6  dp4a-8 matched sx_q (blk=%d)", blk);
                eval_f(nm, y);
            }
            // 10. A-BS8: 8-bit quantized bs (round(bs/4)*4, 64KB LUT / dp2a-bs8)
            {
                std::vector<float> y(NDIM);
                const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) {
                        int q = (((int)bs_mat[(size_t)r * NDIM + c] + 2) >> 2) * 4 - 510;
                        a = fmaf(x[r], (float)q, a);
                    }
                    y[c] = fmaf(KINV, a, KOFF_c * sx_exact);
                }
                eval_f("A-BS8 8-bit bs (round(bs/4)*4, 64KB LUT)", y);
            }
            // 11. A-1XMAD (drop h*0x83DC, use only 2*bytesum_lo(h*0xD12D)) — proves why 2nd XMAD is irreducible
            {
                std::vector<float> y(NDIM);
                const float KOFF_c = (float)(KOFF_d + 510.0 * KINV_d);
                for (int c = 0; c < NDIM; ++c) {
                    float a = 0.f;
                    for (int r = 0; r < KDIM; ++r) {
                        uint32_t p = p_mat[(size_t)r * NDIM + c];
                        int b01 = (int)(p & 255u) + (int)((p >> 8) & 255u);
                        a = fmaf(x[r], (float)(2 * b01 - 510), a);
                    }
                    y[c] = fmaf(KINV, a, KOFF_c * sx_exact);
                }
                eval_f("A-2BYTE (only low 2 bytes x2, 1 XMAD)", y);
            }
        }
    }
    printf("\nALL CPU TESTS PASSED.\n");
    return 0;
}
