// Decode-arithmetic probe for sm_61: device props and the throughput of
// alternative formulations of the EXL3 "bytesum(w16 * M)" decode step, with the
// real window extraction (512-bit ring in smem) and NC consumer FMAs.
// build+run: bash opt/kbench/build.sh probe_decode
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

// window i (16 bits at a 2-bit step) of a 512-bit ring held as 16 u32
__device__ __forceinline__ uint32_t win16(const uint32_t* ring, int i) {
    const int s_i = (i + 1) * 2;
    const int b0 = s_i - 16 + (s_i < 16 ? 512 : 0);
    const int i0 = b0 >> 5, i1 = (i0 + 1 == 16) ? 0 : i0 + 1;
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[i1];
    return (uint32_t)(v >> (48 - (b0 & 31))) & 0xffffu;
}

// A: as compiled today — one 32x32 multiply
__device__ __forceinline__ int bs_cur(uint32_t w16) {
    return (int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
}
// B: the product as two 16x16 pieces: p = w*Ml + ((w*Mh) & 0xFFFF) << 16
__device__ __forceinline__ int bs_2mad(uint32_t w16) {
    const uint32_t p = w16 * 0xD12Du + ((w16 * 0x83DCu) << 16);
    return (int)__dp4a(p, 0x01010101u, 0u);
}
// C: same with unsigned-short operands
__device__ __forceinline__ int bs_2mad16(uint32_t w16) {
    const unsigned short a = (unsigned short)w16;
    const uint32_t p = (uint32_t)a * 0xD12Du + (((uint32_t)a * 0x83DCu) << 16);
    return (int)__dp4a(p, 0x01010101u, 0u);
}
// D: no multiply at all (lower bound of everything else)
__device__ __forceinline__ int bs_nomul(uint32_t w16) {
    return (int)__dp4a(w16, 0x01010101u, 0u);
}

// window with two 32-bit shifts + or instead of the 64-bit funnel shift
__device__ __forceinline__ uint32_t win16_shift(const uint32_t* ring, int i) {
    const int s_i = (i + 1) * 2;
    const int b0 = s_i - 16 + (s_i < 16 ? 512 : 0);
    const int i0 = b0 >> 5, i1 = (i0 + 1 == 16) ? 0 : i0 + 1;
    const uint32_t lo = ring[i0], hi = ring[i1];
    const uint32_t s = b0 & 31;
    return ((lo >> s) | (hi << (32 - s))) & 0xffffu;   // shl by 32 -> 0 (PTX clamp)
}
// bytesum -> float without I2F: as_float(0x4B000000 + bs) == bs + 2^23 exactly
__device__ __forceinline__ float bs_f32(int bs) { return __uint_as_float(0x4B000000u + (uint32_t)bs); }

template <int V> __device__ __forceinline__ int dec(uint32_t w16) {
    return V == 0 ? bs_cur(w16) : V == 1 ? bs_2mad(w16) : V == 2 ? bs_2mad16(w16) : bs_nomul(w16);
}

// EXTRACT 0: funnel shift (SHF), 1: two 32-bit shifts; CONV 0: I2F, 1: magic-number bits
template <int EXTRACT, int CONV, int NC>
__global__ void __launch_bounds__(256) k_decode2(const uint32_t* __restrict__ rings_in,
                                                 const float* __restrict__ xh_in,
                                                 float* __restrict__ out, int nrounds) {
    constexpr int NDEC = 16;
    __shared__ uint32_t ring[16];
    __shared__ float xh[NC][16];
    const int t = threadIdx.x;
    if (t < 16) ring[t] = rings_in[blockIdx.x * 16 + t];
    for (int j = t; j < NC * 16; j += 256) xh[j / 16][j & 15] = xh_in[blockIdx.x * NC * 16 + j];
    __syncthreads();
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    for (int r = 0; r < nrounds; ++r) {
        __syncthreads();
        if (t == 0) ring[r & 15] += 0x9e3779b9u;
        for (int j = t; j < NC * 16; j += 256) xh[j / 16][j & 15] += (float)(r + j);
        __syncthreads();
        float s[NC], sx[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[c] = 0.f; sx[c] = 0.f; }
#pragma unroll
        for (int q = 0; q < NDEC; ++q) {
            const int i = ((t & 15) + 16 * (q & 1) + 32 * ((q >> 1) & 7)) & 255;
            const uint32_t w16 = EXTRACT == 0 ? win16(ring, i) : win16_shift(ring, i);
            const int bs = (int)__dp4a(w16 * 0x83DCD12Du, 0x01010101u, 0u);
            const float fbs = CONV == 0 ? (float)bs : bs_f32(bs);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float xr = xh[c][(t + q) & 15];
                s[c] += xr * fbs; sx[c] += xr;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            acc[c] += CONV == 0 ? fmaf(KINV, s[c], KOFF * sx[c])
                                : fmaf(KINV, fmaf(-8388608.f, sx[c], s[c]), KOFF * sx[c]);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) if (acc[c] == 12345.f) out[c] = acc[c];
}

template <int E, int C, int NC>
float bench2(int nblocks, int nrounds, int reps) {
    uint32_t* rings; float* xh; float* out;
    CK(cudaMalloc(&rings, (size_t)nblocks * 16 * 4));
    CK(cudaMalloc(&xh, (size_t)nblocks * NC * 16 * 4));
    CK(cudaMalloc(&out, 256));
    CK(cudaMemset(rings, 0x5a, (size_t)nblocks * 16 * 4));
    CK(cudaMemset(xh, 0, (size_t)nblocks * NC * 16 * 4));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    k_decode2<E, C, NC><<<nblocks, 256>>>(rings, xh, out, 1); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) k_decode2<E, C, NC><<<nblocks, 256>>>(rings, xh, out, nrounds);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= reps;
    CK(cudaGetLastError());
    CK(cudaFree(rings)); CK(cudaFree(xh)); CK(cudaFree(out));
    return ms;
}

template <int NC>
void row2(const char* name, int e, int c, double ghz, int nsm, int nblocks, double wpl) {
    const int nrounds = 64, reps = 20;
    float ms = e == 0 && c == 0 ? bench2<0,0,NC>(nblocks,nrounds,reps) : e == 0 && c == 1 ? bench2<0,1,NC>(nblocks,nrounds,reps)
             : e == 1 && c == 0 ? bench2<1,0,NC>(nblocks,nrounds,reps) : bench2<1,1,NC>(nblocks,nrounds,reps);
    const double cyc = ms * 1e-3 * ghz * 1e9 * nsm / (wpl / 32.0);
    printf("%-26s %3d   %8.2f   %8.3f\n", name, NC, cyc, ms * (2.65e6 / wpl));
}

// block = 256 threads; each decodes NDEC weights per round (windows striding
// through one tile ring) and applies them to NC consumers from smem.
template <int VARIANT, int NC, int NDEC>
__global__ void __launch_bounds__(256) k_decode(const uint32_t* __restrict__ rings_in,
                                                const float* __restrict__ xh_in,
                                                float* __restrict__ out, int nrounds) {
    __shared__ uint32_t ring[16];
    __shared__ float xh[NC][16];
    const int t = threadIdx.x;
    if (t < 16) ring[t] = rings_in[blockIdx.x * 16 + t];
    for (int j = t; j < NC * 16; j += 256) xh[j / 16][j & 15] = xh_in[blockIdx.x * NC * 16 + j];
    __syncthreads();
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;
    const float KINV = __half2float(__ushort_as_half(0x1eee));
    const float KOFF = 1024.f * KINV + __half2float(__ushort_as_half(0xc931));
    for (int r = 0; r < nrounds; ++r) {
        // mutate the inputs every round so nothing can be hoisted out of the loop
        __syncthreads();
        if (t == 0) ring[r & 15] += 0x9e3779b9u;
        for (int j = t; j < NC * 16; j += 256) xh[j / 16][j & 15] += (float)(r + j);
        __syncthreads();
        float s[NC], sx[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[c] = 0.f; sx[c] = 0.f; }
#pragma unroll
        for (int q = 0; q < NDEC; ++q) {
            const int i = ((t & 15) + 16 * (q & 1) + 32 * ((q >> 1) & 7)) & 255;
            const float fbs = (float)dec<VARIANT>(win16(ring, i));
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float xr = xh[c][(t + q) & 15];
                s[c] += xr * fbs; sx[c] += xr;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] += fmaf(KINV, s[c], KOFF * sx[c]);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) if (acc[c] == 12345.f) out[c] = acc[c];
}

template <int V, int NC>
float bench(int nblocks, int nrounds, int reps) {
    const int NDEC = 16;
    uint32_t* rings; float* xh; float* out;
    CK(cudaMalloc(&rings, (size_t)nblocks * 16 * 4));
    CK(cudaMalloc(&xh, (size_t)nblocks * NC * 16 * 4));
    CK(cudaMalloc(&out, 256));
    CK(cudaMemset(rings, 0x5a, (size_t)nblocks * 16 * 4));
    CK(cudaMemset(xh, 0, (size_t)nblocks * NC * 16 * 4));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    k_decode<V, NC, NDEC><<<nblocks, 256>>>(rings, xh, out, 1); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) k_decode<V, NC, NDEC><<<nblocks, 256>>>(rings, xh, out, nrounds);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b)); ms /= reps;
    CK(cudaGetLastError());
    CK(cudaFree(rings)); CK(cudaFree(xh)); CK(cudaFree(out));
    return ms;
}

template <int NC>
void row(const char* name, int v, double ghz, int nsm, int nblocks, int wpl) {
    const int nrounds = 64, reps = 20;
    float ms = v == 0 ? bench<0, NC>(nblocks, nrounds, reps) : v == 1 ? bench<1, NC>(nblocks, nrounds, reps)
             : v == 2 ? bench<2, NC>(nblocks, nrounds, reps) : bench<3, NC>(nblocks, nrounds, reps);
    // SM-cycles per warp-weight = (time * clocks/s * nSMs) / (weights / 32)
    const double cyc = ms * 1e-3 * ghz * 1e9 * nsm / (wpl / 32.0);
    const double mat_ms = ms * (2.65e6 / wpl);      // time for one 1152x2304 matrix
    printf("%-18s %3d   %8.2f   %8.3f\n", name, NC, cyc, mat_ms);
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("%s sm_%d%d SMs=%d regs/SM=%u smem/SM=%zu smem/block=%zu smem/blockOptin=%zu L2=%zu KB clock=%d kHz\n",
        p.name, p.major, p.minor, p.multiProcessorCount, p.regsPerMultiprocessor,
        p.sharedMemPerMultiprocessor, p.sharedMemPerBlock, p.sharedMemPerBlockOptin,
        p.l2CacheSize >> 10, p.clockRate);
    const double ghz = p.clockRate / 1e6; const int nsm = p.multiProcessorCount;
    const char* names[] = {"A cur (3 XMAD)", "B 2xmad u32", "C 2xmad u16", "D nomul (floor)"};
    const int nblocks = 20 * 4;                 // 4 blocks/SM
    const double wpl = (double)nblocks * 256 * 16 * 64;   // weights per launch
    printf("\ndecode-only throughput (ring in smem, no weight traffic; %d blocks x 256 thr)\n", nblocks);
    printf("%-18s %3s %10s %10s\n", "variant", "nc", "cyc/w-wgt", "ms/matrix");
    printf("\nextract/conversion variants (same harness)\n");
    printf("%-26s %3s %10s %10s\n", "variant", "nc", "cyc/w-wgt", "ms/matrix");
    const char* n2[] = {"SHF + I2F (cur)", "SHF + magic", "SHR/SHL + I2F", "SHR/SHL + magic"};
    for (int nc : {1, 2, 3, 8})
        for (int v = 0; v < 4; ++v) {
            int e = v >> 1, c = v & 1;
            if (nc == 1) row2<1>(n2[v], e, c, ghz, nsm, nblocks, wpl);
            else if (nc == 2) row2<2>(n2[v], e, c, ghz, nsm, nblocks, wpl);
            else if (nc == 3) row2<3>(n2[v], e, c, ghz, nsm, nblocks, wpl);
            else row2<8>(n2[v], e, c, ghz, nsm, nblocks, wpl);
            fflush(stdout);
        }
    return 0;
}
