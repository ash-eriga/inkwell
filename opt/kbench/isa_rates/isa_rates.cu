#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define OP_IADD   asm volatile("add.u32 %0, %0, %1;" : "+r"(v[u]) : "r"(y));
#define OP_IADD3  asm volatile("add.u32 %0, %0, %1;" : "+r"(v[u]) : "r"(y)); asm volatile("add.u32 %0, %0, %1;" : "+r"(v[u]) : "r"(z));
#define OP_LOP3   asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v[u]) : "r"(y), "r"(z));
#define OP_LOP32I asm volatile("and.b32 %0, %0, 0x7fff3fff;" : "+r"(v[u])); asm volatile("or.b32 %0, %0, %1;" : "+r"(v[u]) : "r"(y));
#define OP_SHF    asm volatile("shf.r.wrap.b32 %0, %0, %1, %2;" : "+r"(v[u]) : "r"(y), "r"(z));
#define OP_SHR    asm volatile("shr.b32 %0, %0, %1;" : "+r"(v[u]) : "r"(z));
#define OP_BFE    asm volatile("bfe.u32 %0, %0, %1, %2;" : "+r"(v[u]) : "r"(y), "r"(z));
#define OP_BFI    asm volatile("bfi.b32 %0, %0, %1, %2, %3;" : "+r"(v[u]) : "r"(y), "r"(z), "r"(y));
#define OP_PRMT   asm volatile("prmt.b32 %0, %0, %1, %2;" : "+r"(v[u]) : "r"(y), "r"(z));
#define OP_ISCADD asm volatile("{.reg .u32 t; shl.b32 t, %0, 3; add.u32 %0, t, %1;}" : "+r"(v[u]) : "r"(y));
#define OP_IDP    asm volatile("dp4a.u32.u32 %0, %0, %1, %2;" : "+r"(v[u]) : "r"(y), "r"(z));
#define OP_XMAD   asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v[u]) : "r"(y), "r"(z));
#define OP_POPC   asm volatile("popc.b32 %0, %0;" : "+r"(v[u]));
#define OP_I2F    asm volatile("{.reg .f32 t; cvt.rn.f32.s32 t, %0; mov.b32 %0, t;}" : "+r"(v[u]));
#define OP_FFMA   asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv[u]) : "f"(fy), "f"(fz));
#define OP_FADD   asm volatile("add.f32 %0, %0, %1;" : "+f"(fv[u]) : "f"(fy));
#define OP_SHFL   asm volatile("shfl.sync.idx.b32 %0, %0, %1, 0x1f, 0xffffffff;" : "+r"(v[u]) : "r"(src));
#define OP_LDS    asm volatile("ld.shared.u32 %0, [%0];" : "+r"(v[u]));

__shared__ uint32_t tab[64 * 32];

enum { NOLDS = -1, LDS_OWNBANK = 0, LDS_BROADCAST = 1, LDS_RANDOM = 2 };

template <int U, int MODE>
__device__ __forceinline__ void init_tab(uint32_t* v, int lane) {
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(tab);
    for (int e = threadIdx.x; e < 2048; e += blockDim.x) {
        const int r = e >> 5, b = e & 31;
        int nx;
        if (MODE == LDS_OWNBANK) nx = ((r + 1) & 63) * 32 + b;
        else if (MODE == LDS_BROADCAST) nx = ((r + 1) & 63) * 32;
        else { uint32_t h = (uint32_t)e * 2654435761u; h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15; nx = h & 2047; }
        tab[e] = base + 4 * nx;
    }
    __syncthreads();
    for (int u = 0; u < U; ++u)
        v[u] = base + 4 * (MODE == LDS_OWNBANK ? u * 32 + lane : MODE == LDS_BROADCAST ? u * 32 : (u * 32 + lane * 7) & 2047);
}

#define KERNEL(NAME, BODY)                                                                          \
template <int U, int MODE>                                                                         \
__global__ void __launch_bounds__(1024) k_##NAME(uint32_t* out, long long* cyc, int iters, uint32_t seed) { \
    const int lane = threadIdx.x & 31;                                                             \
    const uint32_t y = seed ^ (threadIdx.x * 0x9e3779b9u), z = (seed * 3u + threadIdx.x) & 15u;    \
    const uint32_t src = (lane + 1) & 31;                                                          \
    const float fy = 1.0f + 1e-7f * (seed & 7), fz = 1e-30f;                                       \
    uint32_t v[U]; float fv[U];                                                                    \
    for (int u = 0; u < U; ++u) { v[u] = y + u; fv[u] = (float)u; }                                \
    if (MODE != NOLDS) init_tab<U, MODE>(v, lane);                                                 \
    (void)src; (void)z; (void)fy; (void)fz;                                                        \
    __syncthreads();                                                                               \
    const long long t0 = clock64();                                                                \
    _Pragma("unroll 4")                                                                            \
    for (int i = 0; i < iters; ++i) {                                                              \
        _Pragma("unroll")                                                                          \
        for (int u = 0; u < U; ++u) { BODY }                                                       \
    }                                                                                              \
    __syncthreads();                                                                               \
    const long long t1 = clock64();                                                                \
    uint32_t acc = 0;                                                                              \
    for (int u = 0; u < U; ++u) acc ^= v[u] ^ __float_as_uint(fv[u]);                             \
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;                                              \
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;                                               \
}

KERNEL(iadd, OP_IADD) KERNEL(iadd3, OP_IADD3) KERNEL(lop3, OP_LOP3) KERNEL(lop32i, OP_LOP32I)
KERNEL(shf, OP_SHF) KERNEL(shr, OP_SHR) KERNEL(bfe, OP_BFE) KERNEL(bfi, OP_BFI) KERNEL(prmt, OP_PRMT)
KERNEL(iscadd, OP_ISCADD) KERNEL(idp, OP_IDP) KERNEL(xmad, OP_XMAD) KERNEL(popc, OP_POPC OP_IADD)
KERNEL(i2f, OP_I2F) KERNEL(ffma, OP_FFMA) KERNEL(fadd, OP_FADD) KERNEL(shfl, OP_SHFL) KERNEL(lds, OP_LDS)
KERNEL(iadd_idp, OP_IADD OP_IDP) KERNEL(ffma_iadd, OP_FFMA OP_IADD) KERNEL(ffma_shf, OP_FFMA OP_SHF)
KERNEL(ffma_3shf, OP_FFMA OP_SHF OP_SHF OP_SHF) KERNEL(xmad_shf, OP_XMAD OP_SHF)
KERNEL(shf_bfe, OP_SHF OP_BFE) KERNEL(shf_prmt, OP_SHF OP_PRMT) KERNEL(shf_iscadd, OP_SHF OP_ISCADD)
KERNEL(ffma_i2f, OP_FFMA OP_I2F) KERNEL(ffma3_i2f, OP_FFMA OP_FFMA OP_FFMA OP_I2F)
KERNEL(ffma_lds, OP_FFMA OP_LDS) KERNEL(ffma3_lds, OP_FFMA OP_FFMA OP_FFMA OP_LDS) KERNEL(ffma_shfl, OP_FFMA OP_SHFL)
KERNEL(shf_i2f, OP_SHF OP_I2F) KERNEL(i2f_popc, OP_I2F OP_POPC) KERNEL(i2f_shfl, OP_I2F OP_SHFL)
KERNEL(i2f_lds, OP_I2F OP_LDS) KERNEL(shfl_lds, OP_SHFL OP_LDS)

typedef void (*kfn)(uint32_t*, long long*, int, uint32_t);
struct Bench { const char* name; kfn thr; kfn lat; };
#define B(N) {#N, k_##N<8, NOLDS>, k_##N<1, NOLDS>}
#define BL(N, M, L) {L, k_##N<8, M>, k_##N<1, M>}

int main() {
    const Bench benches[] = {
        B(iadd), B(iadd3), B(lop3), B(lop32i), B(shf), B(shr), B(bfe), B(bfi), B(prmt), B(iscadd), B(idp),
        B(xmad), B(popc), B(i2f), B(ffma), B(fadd), B(shfl),
        BL(lds, LDS_OWNBANK, "lds_ownbank"), BL(lds, LDS_BROADCAST, "lds_broadcast"), BL(lds, LDS_RANDOM, "lds_random"),
        B(iadd_idp), B(ffma_iadd), B(ffma_shf), B(ffma_3shf), B(xmad_shf), B(shf_bfe), B(shf_prmt), B(shf_iscadd),
        B(ffma_i2f), B(ffma3_i2f), BL(ffma_lds, LDS_OWNBANK, "ffma_lds"), BL(ffma3_lds, LDS_OWNBANK, "ffma3_lds"),
        B(ffma_shfl), B(shf_i2f), B(i2f_popc), B(i2f_shfl), BL(i2f_lds, LDS_OWNBANK, "i2f_lds"),
        BL(shfl_lds, LDS_OWNBANK, "shfl_lds"),
    };
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    uint32_t* out; long long* cyc;
    cudaMalloc(&out, nsm * 1024 * 4);
    cudaMalloc(&cyc, nsm * 8);
    const int iters = 4096;
    printf("%-14s %14s %14s\n", "bench", "steps/clk/SM", "cyc/step(1w)");
    for (const Bench& b : benches) {
        double thr = 0, lat = 1e30;
        long long hc[128];
        for (int rep = 0; rep < 3; ++rep) {
            b.thr<<<nsm, 1024>>>(out, cyc, iters, 12345u + rep);
            cudaMemcpy(hc, cyc, nsm * 8, cudaMemcpyDeviceToHost);
            long long mx = 0;
            for (int i = 0; i < nsm; ++i) mx = hc[i] > mx ? hc[i] : mx;
            const double r = 32.0 * 8 * iters / mx;
            thr = r > thr ? r : thr;
            b.lat<<<1, 32>>>(out, cyc, iters, 999u + rep);
            cudaMemcpy(hc, cyc, 8, cudaMemcpyDeviceToHost);
            const double l = (double)hc[0] / iters;
            lat = l < lat ? l : lat;
        }
        const cudaError_t err = cudaGetLastError();
        printf("%-14s %14.3f %14.2f %s\n", b.name, thr, lat, err ? cudaGetErrorString(err) : "");
    }
    return 0;
}
