// Focused ptxas emission probe for the exact 2-XMAD EXL3 product core.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__device__ uint32_t ch2_cm = 0x83DCD12Du;

__device__ __forceinline__ float finish(uint32_t p, float x, float acc) {
    const uint32_t bits = __dp4a(p, 0x01010101u, 0x4B000000u);
    const float fs = __uint_as_float(bits) - 8388608.f;
    return fmaf(fs, x, acc);
}

// 0/1: algebraic form used by the shipped core.
__device__ __forceinline__ uint32_t prod_expr0(uint32_t u) {
    const uint32_t h = u & 0xffffu;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}
__device__ __forceinline__ uint32_t prod_expr1(uint32_t u) {
    const uint32_t h = u >> 16;
    return h * 0xD12Du + ((h * 0x83DCu) << 16);
}

// 2/3: two separate mul.wide.u16 asm blocks, cm from opaque device memory.
__device__ __forceinline__ uint32_t prod_split0(uint32_t u, uint32_t cm) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, cl;\n\t}"
        : "=r"(t) : "r"(u), "r"(cm));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, lo, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(u), "r"(cm), "r"(t));
    return p;
}
__device__ __forceinline__ uint32_t prod_split1(uint32_t u, uint32_t cm) {
    uint32_t t, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, cl;\n\t}"
        : "=r"(t) : "r"(u), "r"(cm));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, hi, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(u), "r"(cm), "r"(t));
    return p;
}

// 4/5: one asm block for the whole product, still written in PTX-visible ops.
__device__ __forceinline__ uint32_t prod_one0(uint32_t u, uint32_t cm) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 t, q;\n\t"
        "mov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 t, lo, cl;\n\tmul.wide.u16 q, lo, ch;\n\tmad.lo.u32 %0, q, 65536, t;\n\t}"
        : "=r"(p) : "r"(u), "r"(cm));
    return p;
}
__device__ __forceinline__ uint32_t prod_one1(uint32_t u, uint32_t cm) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 t, q;\n\t"
        "mov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 t, hi, cl;\n\tmul.wide.u16 q, hi, ch;\n\tmad.lo.u32 %0, q, 65536, t;\n\t}"
        : "=r"(p) : "r"(u), "r"(cm));
    return p;
}

// 6/7: immediate constants inside one asm block.
__device__ __forceinline__ uint32_t prod_imm0(uint32_t u) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi;\n\t.reg .u32 t, q;\n\tmov.b32 {lo, hi}, %1;\n\t"
        "mul.wide.u16 t, lo, 0xd12d;\n\tmul.wide.u16 q, lo, 0x83dc;\n\tmad.lo.u32 %0, q, 65536, t;\n\t}"
        : "=r"(p) : "r"(u));
    return p;
}
__device__ __forceinline__ uint32_t prod_imm1(uint32_t u) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi;\n\t.reg .u32 t, q;\n\tmov.b32 {lo, hi}, %1;\n\t"
        "mul.wide.u16 t, hi, 0xd12d;\n\tmul.wide.u16 q, hi, 0x83dc;\n\tmad.lo.u32 %0, q, 65536, t;\n\t}"
        : "=r"(p) : "r"(u));
    return p;
}

// 8/9: separate volatile asm statements.
__device__ __forceinline__ uint32_t prod_vol0(uint32_t u, uint32_t cm) {
    uint32_t t, p;
    asm volatile("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, cl;\n\t}"
        : "=r"(t) : "r"(u), "r"(cm));
    asm volatile("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, lo, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(u), "r"(cm), "r"(t));
    return p;
}
__device__ __forceinline__ uint32_t prod_vol1(uint32_t u, uint32_t cm) {
    uint32_t t, p;
    asm volatile("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, cl;\n\t}"
        : "=r"(t) : "r"(u), "r"(cm));
    asm volatile("{\n\t.reg .u16 lo, hi, cl, ch;\n\t.reg .u32 q;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\t"
        "mul.wide.u16 q, hi, ch;\n\tmad.lo.u32 %0, q, 65536, %3;\n\t}"
        : "=r"(p) : "r"(u), "r"(cm), "r"(t));
    return p;
}

// 10/11: keep the high product as a u16 and use PTX shl+add explicitly.
__device__ __forceinline__ uint32_t prod_shift0(uint32_t u, uint32_t cm) {
    uint32_t t, q, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, cl;\n\t}"
        : "=r"(t) : "r"(u), "r"(cm));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, lo, ch;\n\t}"
        : "=r"(q) : "r"(u), "r"(cm));
    p = (q << 16) + t;
    return p;
}
__device__ __forceinline__ uint32_t prod_shift1(uint32_t u, uint32_t cm) {
    uint32_t t, q, p;
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, cl;\n\t}"
        : "=r"(t) : "r"(u), "r"(cm));
    asm("{\n\t.reg .u16 lo, hi, cl, ch;\n\tmov.b32 {lo, hi}, %1;\n\tmov.b32 {cl, ch}, %2;\n\tmul.wide.u16 %0, hi, ch;\n\t}"
        : "=r"(q) : "r"(u), "r"(cm));
    p = (q << 16) + t;
    return p;
}


// 12/13: multiply the zero-extended window directly by 0x83DC0000.
// For a 16-bit w this is exactly ((w*0x83DC) mod 2^16) << 16.
__device__ __forceinline__ uint32_t prod_direct0(uint32_t u) {
    const uint32_t w = u & 0xffffu;
    const uint32_t t = w * 0xd12du;
    return w * 0x83dc0000u + t;
}
__device__ __forceinline__ uint32_t prod_direct1(uint32_t u) {
    const uint32_t w = u >> 16;
    const uint32_t t = w * 0xd12du;
    return w * 0x83dc0000u + t;
}

// 14/15: same algebra through one PTX mad.lo.u32 asm block.
__device__ __forceinline__ uint32_t prod_dasm0(uint32_t u) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi;\n\t.reg .u32 w, t;\n\tmov.b32 {lo, hi}, %1;\n\tcvt.u32.u16 w, lo;\n\t"
        "mul.wide.u16 t, lo, 0xd12d;\n\tmad.lo.u32 %0, w, 0x83dc0000, t;\n\t}"
        : "=r"(p) : "r"(u));
    return p;
}
__device__ __forceinline__ uint32_t prod_dasm1(uint32_t u) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi;\n\t.reg .u32 w, t;\n\tmov.b32 {lo, hi}, %1;\n\tcvt.u32.u16 w, hi;\n\t"
        "mul.wide.u16 t, hi, 0xd12d;\n\tmad.lo.u32 %0, w, 0x83dc0000, t;\n\t}"
        : "=r"(p) : "r"(u));
    return p;
}

// 16/17: use PRMT to place the window in the high half, then multiply by 0x83dc.
// The shifted-out low half is immaterial because only the low 32 bits are kept.
__device__ __forceinline__ uint32_t prod_prmt0(uint32_t u) {
    const uint32_t wh = __byte_perm(u, 0u, 0x4100u);   // low window in bits 16..31
    const uint32_t wl = u & 0xffffu;
    const uint32_t t = wl * 0xd12du;
    return wh * 0x83dcu + t;
}
__device__ __forceinline__ uint32_t prod_prmt1(uint32_t u) {
    const uint32_t wl = u >> 16;
    const uint32_t t = wl * 0xd12du;
    return (u & 0xffff0000u) * 0x83dcu + t;
}


// 18/19: build the shifted window explicitly from u16 zero + window halves.
__device__ __forceinline__ uint32_t prod_zpack0(uint32_t u) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi, z;\n\t.reg .u32 w, wh, t;\n\tmov.b32 {lo, hi}, %1;\n\tmov.u16 z, 0;\n\t"
        "mov.b32 wh, {z, lo};\n\tmul.wide.u16 t, lo, 0xd12d;\n\tmad.lo.u32 %0, wh, 0x83dc, t;\n\t}"
        : "=r"(p) : "r"(u));
    return p;
}
__device__ __forceinline__ uint32_t prod_zpack1(uint32_t u) {
    uint32_t p;
    asm("{\n\t.reg .u16 lo, hi, z;\n\t.reg .u32 wh, t;\n\tmov.b32 {lo, hi}, %1;\n\tmov.u16 z, 0;\n\t"
        "mov.b32 wh, {z, hi};\n\tmul.wide.u16 t, hi, 0xd12d;\n\tmad.lo.u32 %0, wh, 0x83dc, t;\n\t}"
        : "=r"(p) : "r"(u));
    return p;
}

template <int MODE>
__global__ void __launch_bounds__(32) k_emit(const uint32_t* __restrict__ in,
                                             const float* __restrict__ x,
                                             float* __restrict__ out, int n) {
    float acc = 0.f;
    const uint32_t cm = ch2_cm;
    for (int i = threadIdx.x; i < n; i += 32) {
        const uint32_t u = in[i];
        uint32_t p;
        if constexpr (MODE == 0) p = prod_expr0(u);
        else if constexpr (MODE == 1) p = prod_expr1(u);
        else if constexpr (MODE == 2) p = prod_split0(u, cm);
        else if constexpr (MODE == 3) p = prod_split1(u, cm);
        else if constexpr (MODE == 4) p = prod_one0(u, cm);
        else if constexpr (MODE == 5) p = prod_one1(u, cm);
        else if constexpr (MODE == 6) p = prod_imm0(u);
        else if constexpr (MODE == 7) p = prod_imm1(u);
        else if constexpr (MODE == 8) p = prod_vol0(u, cm);
        else if constexpr (MODE == 9) p = prod_vol1(u, cm);
        else if constexpr (MODE == 10) p = prod_shift0(u, cm);
        else if constexpr (MODE == 11) p = prod_shift1(u, cm);
        else if constexpr (MODE == 12) p = prod_direct0(u);
        else if constexpr (MODE == 13) p = prod_direct1(u);
        else if constexpr (MODE == 14) p = prod_dasm0(u);
        else if constexpr (MODE == 15) p = prod_dasm1(u);
        else if constexpr (MODE == 16) p = prod_prmt0(u);
        else if constexpr (MODE == 17) p = prod_prmt1(u);
        else if constexpr (MODE == 18) p = prod_zpack0(u);
        else p = prod_zpack1(u);
        acc = finish(p, x[i], acc);
    }
    if (threadIdx.x == 0 && acc == 1234567.f) out[0] = acc;
}

extern "C" void launch_emit_probe(int mode, const uint32_t* in, const float* x, float* out, int n, void* stream) {
    switch (mode) {
        case 0: k_emit<0><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 1: k_emit<1><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 2: k_emit<2><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 3: k_emit<3><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 4: k_emit<4><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 5: k_emit<5><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 6: k_emit<6><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 7: k_emit<7><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 8: k_emit<8><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 9: k_emit<9><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 10: k_emit<10><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 11: k_emit<11><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 12: k_emit<12><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 13: k_emit<13><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 14: k_emit<14><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 15: k_emit<15><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 16: k_emit<16><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 17: k_emit<17><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        case 18: k_emit<18><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
        default: k_emit<19><<<1, 32, 0, (cudaStream_t)stream>>>(in, x, out, n); break;
    }
}
