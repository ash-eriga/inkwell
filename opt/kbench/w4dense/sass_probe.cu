#include <cstdint>
#include <cuda_fp16.h>

// Helper intrinsics for sm_61 SASS verification
__device__ __forceinline__ uint32_t and_opaque(uint32_t a, uint32_t mask) {
    uint32_t d;
    asm("and.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

__device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); // (a & b) | c
    return d;
}

__device__ __forceinline__ uint32_t lop3_or_and(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xF8;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); // a | (b & c)
    return d;
}

__device__ __forceinline__ float cvt_u8_byte(uint32_t v, int byte_idx) {
    return (float)((v >> (byte_idx * 8)) & 0xFFu);
}

__device__ __forceinline__ float cvt_s8_byte(uint32_t v, int byte_idx) {
    return (float)(int8_t)((v >> (byte_idx * 8)) & 0xFFu);
}

// Explicit PTX test: does byte-union or shift+mask emit native I2F.F32.U8.B{0..3}?
__device__ __forceinline__ float cvt_u8_union(uint32_t v, int byte_idx) {
    union { uint32_t u; uint8_t b[4]; } x;
    x.u = v;
    return (float)x.b[byte_idx];
}

__device__ __forceinline__ int dp2a_lo_su(int a16x2, uint32_t b8x4, int c) {
    int d;
    asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a16x2), "r"(b8x4), "r"(c));
    return d;
}
__device__ __forceinline__ int dp2a_hi_su(int a16x2, uint32_t b8x4, int c) {
    int d;
    asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a16x2), "r"(b8x4), "r"(c));
    return d;
}

// 1. W4 dual-scale unsigned byte I2F (lo = p & 0x0F0F0F0F, hi = p & 0xF0F0F0F0)
__global__ void probe_w4_dual_i2f(const uint4* __restrict__ W, const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        float a_lo = 0.f, a_hi = 0.f;
        uint4 wv = __ldg(W + g);
        uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = ps[k];
            uint32_t lo = and_opaque(p, 0x0F0F0F0Fu);
            uint32_t hi = and_opaque(p, 0xF0F0F0F0u);
            float4 x0 = xs[g * 8 + k * 2 + 0];
            float4 x1 = xs[g * 8 + k * 2 + 1];
            a_lo = fmaf(cvt_u8_byte(lo, 0), x0.x, a_lo);
            a_lo = fmaf(cvt_u8_byte(lo, 1), x0.y, a_lo);
            a_lo = fmaf(cvt_u8_byte(lo, 2), x0.z, a_lo);
            a_lo = fmaf(cvt_u8_byte(lo, 3), x0.w, a_lo);
            a_hi = fmaf(cvt_u8_byte(hi, 0), x1.x, a_hi);
            a_hi = fmaf(cvt_u8_byte(hi, 1), x1.y, a_hi);
            a_hi = fmaf(cvt_u8_byte(hi, 2), x1.z, a_hi);
            a_hi = fmaf(cvt_u8_byte(hi, 3), x1.w, a_hi);
        }
        acc += fmaf(0.0625f, a_hi, a_lo);
    }
    out[threadIdx.x] = acc;
}

// 2. W4 signed int8 byte I2F (lo = (p<<4)&0xF0F0F0F0, hi = p&0xF0F0F0F0 -> I2F.F32.S8)
__global__ void probe_w4_s8_i2f(const uint4* __restrict__ W, const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        float ag = 0.f;
        uint4 wv = __ldg(W + g);
        uint32_t ps[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = ps[k];
            uint32_t lo = and_opaque(p << 4, 0xF0F0F0F0u);
            uint32_t hi = and_opaque(p, 0xF0F0F0F0u);
            float4 x0 = xs[g * 8 + k * 2 + 0];
            float4 x1 = xs[g * 8 + k * 2 + 1];
            ag = fmaf(cvt_s8_byte(lo, 0), x0.x, ag);
            ag = fmaf(cvt_s8_byte(lo, 1), x0.y, ag);
            ag = fmaf(cvt_s8_byte(lo, 2), x0.z, ag);
            ag = fmaf(cvt_s8_byte(lo, 3), x0.w, ag);
            ag = fmaf(cvt_s8_byte(hi, 0), x1.x, ag);
            ag = fmaf(cvt_s8_byte(hi, 1), x1.y, ag);
            ag = fmaf(cvt_s8_byte(hi, 2), x1.z, ag);
            ag = fmaf(cvt_s8_byte(hi, 3), x1.w, ag);
        }
        acc = fmaf(0.0625f, ag, acc);
    }
    out[threadIdx.x] = acc;
}

// 3. W4 BFE + I2F
__global__ void probe_w4_bfe_i2f(const uint32_t* __restrict__ W, const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = __ldg(W + g * 4 + k);
            float4 x0 = xs[g * 8 + k * 2 + 0];
            float4 x1 = xs[g * 8 + k * 2 + 1];
            acc = fmaf((float)(p & 15u), x0.x, acc);
            acc = fmaf((float)((p >> 4) & 15u), x0.y, acc);
            acc = fmaf((float)((p >> 8) & 15u), x0.z, acc);
            acc = fmaf((float)((p >> 12) & 15u), x0.w, acc);
            acc = fmaf((float)((p >> 16) & 15u), x1.x, acc);
            acc = fmaf((float)((p >> 20) & 15u), x1.y, acc);
            acc = fmaf((float)((p >> 24) & 15u), x1.z, acc);
            acc = fmaf((float)(p >> 28), x1.w, acc);
        }
    }
    out[threadIdx.x] = acc;
}

// 4. W4 LOP3 magic 0x4B000000 + FADD(-8388608.f)
__global__ void probe_w4_lop3_magic(const uint32_t* __restrict__ W, const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = __ldg(W + g * 4 + k);
            float4 x0 = xs[g * 8 + k * 2 + 0];
            float4 x1 = xs[g * 8 + k * 2 + 1];
            auto cvt = [](uint32_t v) { return __uint_as_float(lop3_and_or(v, 0xFu, 0x4B000000u)) - 8388608.f; };
            acc = fmaf(cvt(p), x0.x, acc);
            acc = fmaf(cvt(p >> 4), x0.y, acc);
            acc = fmaf(cvt(p >> 8), x0.z, acc);
            acc = fmaf(cvt(p >> 12), x0.w, acc);
            acc = fmaf(cvt(p >> 16), x1.x, acc);
            acc = fmaf(cvt(p >> 20), x1.y, acc);
            acc = fmaf(cvt(p >> 24), x1.z, acc);
            acc = fmaf(cvt(p >> 28), x1.w, acc);
        }
    }
    out[threadIdx.x] = acc;
}

// 5. W4 LOP3 folded 0x3F800000 (1.0 + u/16, no FADD, no I2F)
__global__ void probe_w4_lop3_fold(const uint32_t* __restrict__ W, const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = __ldg(W + g * 4 + k);
            float4 x0 = xs[g * 8 + k * 2 + 0];
            float4 x1 = xs[g * 8 + k * 2 + 1];
            auto cvt = [](uint32_t v) { return __uint_as_float(lop3_and_or(v, 0x00780000u, 0x3F800000u)); };
            acc = fmaf(cvt(p << 19), x0.x, acc);
            acc = fmaf(cvt(p << 15), x0.y, acc);
            acc = fmaf(cvt(p << 11), x0.z, acc);
            acc = fmaf(cvt(p << 7),  x0.w, acc);
            acc = fmaf(cvt(p << 3),  x1.x, acc);
            acc = fmaf(cvt(p >> 1),  x1.y, acc);
            acc = fmaf(cvt(p >> 5),  x1.z, acc);
            acc = fmaf(cvt(p >> 9),  x1.w, acc);
        }
    }
    out[threadIdx.x] = acc;
}

// 6. W4 int16 dp2a
__global__ void probe_w4_dp2a(const uint32_t* __restrict__ W, const int4* __restrict__ xs16, int* out, int ng) {
    int acc = 0;
    for (int g = 0; g < ng; ++g) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = __ldg(W + g * 4 + k);
            uint32_t lo = p & 0x0F0F0F0Fu;
            uint32_t hi = (p >> 4) & 0x0F0F0F0Fu;
            int4 x = xs16[g * 4 + k];
            acc = dp2a_lo_su(x.x, lo, acc);
            acc = dp2a_hi_su(x.y, lo, acc);
            acc = dp2a_lo_su(x.z, hi, acc);
            acc = dp2a_hi_su(x.w, hi, acc);
        }
    }
    out[threadIdx.x] = acc;
}

// 7. W5 4+1 unsigned byte I2F (32 weights = 4 low u32 + 1 high u32)
__global__ void probe_w5_4p1_u8(const uint32_t* __restrict__ Wlo, const uint32_t* __restrict__ Whi,
                                const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        uint32_t H = __ldg(Whi + g);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t p = __ldg(Wlo + g * 4 + k);
            uint32_t m_lo = 2 * k;
            uint32_t m_hi = 2 * k + 1;
            uint32_t h_lo = (m_lo < 4) ? (H << (4 - m_lo)) : ((m_lo == 4) ? H : (H >> (m_lo - 4)));
            uint32_t h_hi = (m_hi < 4) ? (H << (4 - m_hi)) : ((m_hi == 4) ? H : (H >> (m_hi - 4)));
            uint32_t lo = lop3_or_and(p & 0x0F0F0F0Fu, h_lo, 0x10101010u);
            uint32_t hi = lop3_or_and((p >> 4) & 0x0F0F0F0Fu, h_hi, 0x10101010u);
            float4 x0 = xs[g * 8 + k * 2 + 0];
            float4 x1 = xs[g * 8 + k * 2 + 1];
            acc = fmaf(cvt_u8_byte(lo, 0), x0.x, acc);
            acc = fmaf(cvt_u8_byte(lo, 1), x0.y, acc);
            acc = fmaf(cvt_u8_byte(lo, 2), x0.z, acc);
            acc = fmaf(cvt_u8_byte(lo, 3), x0.w, acc);
            acc = fmaf(cvt_u8_byte(hi, 0), x1.x, acc);
            acc = fmaf(cvt_u8_byte(hi, 1), x1.y, acc);
            acc = fmaf(cvt_u8_byte(hi, 2), x1.z, acc);
            acc = fmaf(cvt_u8_byte(hi, 3), x1.w, acc);
        }
    }
    out[threadIdx.x] = acc;
}

// 8. W6 4+2 unsigned byte I2F (32 weights = 4 low u32 + 2 high u32)
__global__ void probe_w6_4p2_u8(const uint32_t* __restrict__ Wlo, const uint32_t* __restrict__ Whi,
                                const float4* __restrict__ xs, float* out, int ng) {
    float acc = 0.f;
    for (int g = 0; g < ng; ++g) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t H = __ldg(Whi + g * 2 + j);
            uint32_t p0 = __ldg(Wlo + g * 4 + j * 2 + 0);
            uint32_t p1 = __ldg(Wlo + g * 4 + j * 2 + 1);
            uint32_t u0 = lop3_or_and(p0 & 0x0F0F0F0Fu, H << 4, 0x30303030u);
            uint32_t u1 = lop3_or_and((p0 >> 4) & 0x0F0F0F0Fu, H << 2, 0x30303030u);
            uint32_t u2 = lop3_or_and(p1 & 0x0F0F0F0Fu, H, 0x30303030u);
            uint32_t u3 = lop3_or_and((p1 >> 4) & 0x0F0F0F0Fu, H >> 2, 0x30303030u);
            float4 x0 = xs[g * 8 + j * 4 + 0];
            float4 x1 = xs[g * 8 + j * 4 + 1];
            float4 x2 = xs[g * 8 + j * 4 + 2];
            float4 x3 = xs[g * 8 + j * 4 + 3];
            acc = fmaf(cvt_u8_byte(u0, 0), x0.x, acc);
            acc = fmaf(cvt_u8_byte(u0, 1), x0.y, acc);
            acc = fmaf(cvt_u8_byte(u0, 2), x0.z, acc);
            acc = fmaf(cvt_u8_byte(u0, 3), x0.w, acc);
            acc = fmaf(cvt_u8_byte(u1, 0), x1.x, acc);
            acc = fmaf(cvt_u8_byte(u1, 1), x1.y, acc);
            acc = fmaf(cvt_u8_byte(u1, 2), x1.z, acc);
            acc = fmaf(cvt_u8_byte(u1, 3), x1.w, acc);
            acc = fmaf(cvt_u8_byte(u2, 0), x2.x, acc);
            acc = fmaf(cvt_u8_byte(u2, 1), x2.y, acc);
            acc = fmaf(cvt_u8_byte(u2, 2), x2.z, acc);
            acc = fmaf(cvt_u8_byte(u2, 3), x2.w, acc);
            acc = fmaf(cvt_u8_byte(u3, 0), x3.x, acc);
            acc = fmaf(cvt_u8_byte(u3, 1), x3.y, acc);
            acc = fmaf(cvt_u8_byte(u3, 2), x3.z, acc);
            acc = fmaf(cvt_u8_byte(u3, 3), x3.w, acc);
        }
    }
    out[threadIdx.x] = acc;
}
