// isolate win_geo+funnelshift: dump the 4 w16 each (c,m) lane extracts
// usage: v2_isolate ka halfk tn  — prints mismatches vs CPU window()
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/exl3_gemv_v2.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void dump_w16(const uint32_t* __restrict__ ring, uint32_t* __restrict__ out,
                         int ka, int halfk, int tn) {
    const int tid = threadIdx.x;
    const int c = (tid & 63) >> 2, m = tid & 3;
    const int w = exl3v2::words_of(ka, halfk);
    const exl3v2::WinGeo geo = exl3v2::win_geo(c, m, ka, halfk);
    const uint32_t X = __funnelshift_l(ring[geo.m1], ring[geo.m0], geo.d);
    out[tid * 4 + 0] = X >> 16;
    out[tid * 4 + 1] = (X >> (16 - geo.s1)) & 0xffffu;
    out[tid * 4 + 2] = (X >> (16 - geo.s2)) & 0xffffu;
    out[tid * 4 + 3] = (X >> (16 - geo.s3)) & 0xffffu;
    if (tid == 0) printf("geo m0=%d m1=%d d=%d s123=%d,%d,%d w=%d\n", geo.m0, geo.m1, geo.d, geo.s1, geo.s2, geo.s3, w);
}

int main(int argc, char** argv) {
    const int ka = argc > 1 ? atoi(argv[1]) : 2;
    const int halfk = argc > 2 ? atoi(argv[2]) : 0;
    const int tn = 1;
    const int w = exl3v2::words_of(ka, halfk);
    std::vector<uint32_t> ring(w);
    uint32_t st = 7u;
    for (auto& v : ring) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; v = st; }
    uint32_t* dr; CK(cudaMalloc(&dr, w * 4)); CK(cudaMemcpy(dr, ring.data(), w * 4, cudaMemcpyHostToDevice));
    uint32_t* dout; CK(cudaMalloc(&dout, 64 * 4 * 4));
    dump_w16<<<1, 64>>>(dr, dout, ka, halfk, tn);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<uint32_t> o(64 * 4);
    CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
    // CPU ref: v1 window() for positions i0..i0+3 of each lane
    auto window = [&](int i) {
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int s_i = (i + 1) * ka + (halfk ? (i + 1) >> 1 : 0);
        const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
        const int nw = r_bits >> 5, i0 = b0 >> 5, i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
        const uint64_t v = ((uint64_t)ring[i0] << 32) | ring[i1];
        return (uint32_t)(v >> (48 - (b0 & 31))) & 0xffffu;
    };
    int bad = 0;
    for (int tid = 0; tid < 64; ++tid) {
        const int c = tid >> 2, m = tid & 3;
        const int i0 = 32 * (c & 7) + 8 * m + 4 * (c >> 3);
        for (int k = 0; k < 4; ++k) {
            const uint32_t ref = window(i0 + k);
            if (o[tid * 4 + k] != ref) {
                if (bad < 12) printf("c=%d m=%d k=%d i=%d: gpu %04x ref %04x\n", c, m, k, i0 + k, o[tid * 4 + k], ref);
                ++bad;
            }
        }
    }
    printf("ka=%d halfk=%d w=%d: %d/256 windows wrong\n", ka, halfk, w, bad);
    return 0;
}
