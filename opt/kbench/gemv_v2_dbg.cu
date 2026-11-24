// kbench: tiny end-to-end check of exl3v2::gemv_n against a CPU reference.
// build+run: bash opt/kbench/build.sh gemv_v2_dbg && opt/kbench/gemv_v2_dbg [NTOK] [ka] [halfk] [tk] [nky]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"
#include "../../crates/inkwell-cuda/src/exl3_gemv_v2.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

int main(int argc, char** argv) {
    const int NTOK = argc > 1 ? atoi(argv[1]) : 1;
    const int ka = argc > 2 ? atoi(argv[2]) : 2;
    const int halfk = argc > 3 ? atoi(argv[3]) : 0;
    const int tk = argc > 4 ? atoi(argv[4]) : 32;
    const int nky = argc > 5 ? atoi(argv[5]) : 1;
    const int tn = argc > 6 ? atoi(argv[6]) : 8;
    const int N = tn * 16, K = tk * 16, w = exl3v2::words_of(ka, halfk);
    std::vector<uint32_t> trel((size_t)tk * tn * w);
    uint32_t st = 7u;
    const int pat = argc > 7 ? atoi(argv[7]) : 0;
    for (auto& v : trel) { st ^= st << 13; st ^= st >> 17; st ^= st << 5; v = st; }
    if (pat == 1) std::fill(trel.begin(), trel.end(), 0u);            // all windows 0
    if (pat == 2) std::fill(trel.begin(), trel.end(), 0xffffffffu);   // all windows 0xffff
    if (pat >= 3) { std::fill(trel.begin(), trel.end(), 0u);          // single word/variant per tile
        for (int tile = 0; tile < tn; ++tile) trel[(size_t)tile * w + (pat - 3) % w] = 0x01020304u << ((pat - 3) / w); }
    std::vector<float> xh((size_t)NTOK * K);
    for (size_t i = 0; i < xh.size(); ++i) xh[i] = ((int)(i * 2654435761u % 2001) - 1000) / 400.f;
    uint32_t* dtrel; float *dxh, *dt; CK(cudaMalloc(&dtrel, trel.size() * 4)); CK(cudaMalloc(&dxh, xh.size() * 4));
    CK(cudaMalloc(&dt, (size_t)nky * NTOK * N * 4));
    CK(cudaMemcpy(dtrel, trel.data(), trel.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxh, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dt, 0, (size_t)nky * NTOK * N * 4));
    const int nb = (tn + 3) / 4;
    if (NTOK == 1) exl3v2::gemv_n<1><<<dim3(nb, nky), 256>>>(dtrel, dxh, dt, tk, tn, ka, halfk);
    else if (NTOK == 2) exl3v2::gemv_n<2><<<dim3(nb, nky), 256>>>(dtrel, dxh, dt, tk, tn, ka, halfk);
    else exl3v2::gemv_n<8><<<dim3(nb, nky), 256>>>(dtrel, dxh, dt, tk, tn, ka, halfk);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<float> t((size_t)nky * NTOK * N);
    CK(cudaMemcpy(t.data(), dt, t.size() * 4, cudaMemcpyDeviceToHost));
    // v1 kernel on the same data, as a sanity check of the CPU reference below
    {
        std::vector<float> t1((size_t)nky * N); float* dt1; CK(cudaMalloc(&dt1, t1.size() * 4)); CK(cudaMemset(dt1, 0, t1.size() * 4));
        inkwell_exl3_gemv_dp4a<<<dim3(nb, nky), 256>>>(dtrel, dxh, dt1, tk, tn, ka, halfk);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(t1.data(), dt1, t1.size() * 4, cudaMemcpyDeviceToHost));
        auto window1 = [&](int tile, int i) {
            const uint32_t* ring = &trel[(size_t)tile * w];
            const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
            const int s_i = (i + 1) * ka + (halfk ? (i + 1) >> 1 : 0);
            const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
            const int nw = r_bits >> 5, i0 = b0 >> 5, i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
            const uint64_t v = ((uint64_t)ring[i0] << 32) | ring[i1];
            return (uint32_t)(v >> (48 - (b0 & 31))) & 0xffffu;
        };
        auto bsf1 = [&](uint32_t w16) { const uint32_t p = w16 * 0x83DCD12Du;
            return (float)((p & 0xffu) + ((p >> 8) & 0xffu) + ((p >> 16) & 0xffu) + ((p >> 24) & 0xffu)); };
        auto f16 = [](uint32_t bits) { const int e = (bits >> 10) & 31, m = bits & 1023;
            const double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp(m / 1024.0, -14);
            return (float)((bits & 0x8000) ? -v : v); };
        const float K1 = f16(0x1eee), O1 = 1024.f * K1 + f16(0xc931);
        int bad1 = 0; double w1 = 0;
        for (int col = 0; col < N; ++col) {
            double s = 0, sx = 0; const int tile = col / 16, c = col % 16;
            for (int r = 0; r < 16; ++r) for (int row = 0; row < tk; ++row) {
                const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3) | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
                const float x = xh[(size_t)row * 16 + r];
                s += (double)x * bsf1(window1(row * tn + tile, i)); sx += x;
            }
            const double ref = (double)K1 * s + (double)O1 * sx, got = [&]{ double a = 0; for (int z = 0; z < nky; ++z) a += t1[(size_t)z * N + col]; return a; }();
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1.0);
            if (rel > 1e-4) { if (bad1 < 3) printf("  v1 col %d: gpu %g ref %g rel %.2e\n", col, got, ref, rel); ++bad1; }
            w1 = std::max(w1, rel);
        }
        printf("v1 inkwell_exl3_gemv_dp4a vs CPU ref: %d/%d off, worst %.2e\n", bad1, N, w1);
        CK(cudaFree(dt1));
    }
    // CPU reference with the v1 window formula
    auto window = [&](int tile, int i) {
        const uint32_t* ring = &trel[(size_t)tile * w];
        const int r_bits = halfk ? 256 * ka + 128 : 256 * ka;
        const int s_i = (i + 1) * ka + (halfk ? (i + 1) >> 1 : 0);
        const int b0 = s_i - 16 + (s_i < 16 ? r_bits : 0);
        const int nw = r_bits >> 5, i0 = b0 >> 5, i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
        const uint64_t v = ((uint64_t)ring[i0] << 32) | ring[i1];
        return (uint32_t)(v >> (48 - (b0 & 31))) & 0xffffu;
    };
    auto bsf = [&](uint32_t w16) {
        const uint32_t p = w16 * 0x83DCD12Du;
        return (float)(((p & 0xffu) >> 0) + ((p >> 8) & 0xffu) + ((p >> 16) & 0xffu) + ((p >> 24) & 0xffu));
    };
    auto f16 = [](uint32_t bits) {
        const int e = (bits >> 10) & 31, m = bits & 1023;
        const double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp(m / 1024.0, -14);
        return (float)((bits & 0x8000) ? -v : v);
    };
    const float KINV = f16(0x1eee), KOFF = 1024.f * KINV + f16(0xc931);
    int bad = 0; double worst = 0;
    for (int tt = 0; tt < NTOK; ++tt) {
        for (int col = 0; col < N; ++col) {
            double s = 0, sx = 0;
            const int tile = col / 16, c = col % 16;
            for (int r = 0; r < 16; ++r)
                for (int row = 0; row < tk; ++row) {
                    const int i = ((((c & 7) << 2) | ((r >> 1) & 3)) << 3) | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1);
                    const float x = xh[(size_t)tt * K + row * 16 + r];
                    s += (double)x * bsf(window(row * tn + tile, i));
                    sx += x;
                }
            const double ref = (double)KINV * s + (double)KOFF * sx;
            double got = 0;
            for (int z = 0; z < nky; ++z) got += t[((size_t)z * NTOK + tt) * N + col];
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1.0);
            if (rel > 1e-4) { if (bad < (pat ? 4 : 6)) printf("  tt %d col %d (tile %d c %d): gpu %g  ref %g  rel %.2e\n", tt, col, tile, c, got, ref, rel); ++bad; }
            worst = std::max(worst, rel);
        }
    }
    printf("NTOK=%d ka=%d halfk=%d tk=%d nky=%d tn=%d pat=%d: %d/%d columns off, worst rel %.2e\n", NTOK, ka, halfk, tk, nky, tn, pat, bad, NTOK * N, worst);
    return 0;
}
