// Per-K kbench: shipped sgemv_run vs patched CH2 exact 2-XMAD core, DRAM-bound shapes.
// MoE: 8 slots each with a DISTINCT tensor copy; dense: tn=1152 (10MB+).
// Interleaved min-of-31 x2. Usage: flock /tmp/inkwell-gpu.lock ./ch2_perk
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../../crates/inkwell-cuda/src/kernels.cu"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while(0)
static uint32_t rs_ = 777001u;
static uint32_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 17; rs_ ^= rs_ << 5; return rs_; }
static float timeit(std::function<void()> f, int reps = 31) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) f();
    CK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    return v[0];
}
int main() {
    struct K { int ka, hk, kk; const char* name; } ks[] = {
        {1,1,3,"1.5"},{2,0,4,"2.0"},{2,1,5,"2.5"},{3,0,6,"3.0"},{3,1,7,"3.5"},{4,0,8,"4.0"},{5,0,10,"5.0"}
    };
    printf("%-4s %-24s %-24s %-24s\n", "K", "moe gu old->new %", "moe dn old->new %", "dense old->new %");
    for (auto& K : ks) {
        const int nw = 8 * K.ka + 4 * K.hk;
        std::vector<double> out(3);
        // ---- MoE shapes: (tk,tn) in {(144,72)=gate/up, (72,144)=down}, 8 distinct slots
        for (int sh = 0; sh < 3; ++sh) {
            const int tk = sh == 2 ? 72 : 144, tn = sh == 2 ? 144 : 72;
            const int nslots = sh == 2 ? 8 : 8;
            const size_t words = (size_t)tk * tn * nw;
            std::vector<uint32_t> hw(words);
            for (auto& w : hw) w = rnd();
            const int k = tk * 16, n = tn * 16;
            std::vector<float> hx((size_t)nslots * k);
            for (auto& v : hx) v = ((int)(rnd() % 2001) - 1000) / 512.f;
            float *dxhg, *dt;
            CK(cudaMalloc(&dxhg, hx.size() * 4)); CK(cudaMemcpy(dxhg, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dt, (size_t)nslots * n * 4));
            std::vector<uint64_t> hd(nslots * 4);
            std::vector<uint32_t*> slotptr(nslots);
            for (int s = 0; s < nslots; ++s) {
                CK(cudaMalloc(&slotptr[s], words * 4));
                CK(cudaMemcpy(slotptr[s], hw.data(), words * 4, cudaMemcpyHostToDevice));
                hd[s * 4 + 0] = (uint64_t)(uintptr_t)slotptr[s];
                hd[s * 4 + 3] = K.ka | (K.hk << 8);
            }
            uint64_t* dd; int* dids;
            CK(cudaMalloc(&dd, hd.size() * 8)); CK(cudaMemcpy(dd, hd.data(), hd.size() * 8, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dids, nslots * 4));
            std::vector<int> hi(nslots); for (int i = 0; i < nslots; ++i) hi[i] = i;
            CK(cudaMemcpy(dids, hi.data(), nslots * 4, cudaMemcpyHostToDevice));
            const double w_el = (double)tk * tn * 256 * nslots;
            double t_o = 1e30, t_n = 1e30;
            if (sh == 2) {   // dense shape: one big tensor, gemv_s nky=2
                uint32_t* dtrel = slotptr[0];
                for (int r = 0; r < 2; ++r) {
                    t_o = std::min(t_o, (double)timeit([&]{ cudaMemset(dt, 0, 2 * n * 4);
                        inkwell_exl3_gemv_s<<<dim3((tn + 7) / 8, 2, 1), 256>>>(dtrel, dxhg, dt, tk, tn, K.ka, K.hk, 0); }));
                    t_n = std::min(t_n, (double)timeit([&]{ cudaMemset(dt, 0, 2 * n * 4);
                        inkwell_exl3_gemv_s<<<dim3((tn + 7) / 8, 2, 1), 256>>>(dtrel, dxhg, dt, tk, tn, K.ka, K.hk, 1 << K.kk); }));
                }
                out[sh] = t_o / t_n;
            } else {          // moe shape with distinct slot tensors, nky=2
                if (sh == 0) {
                    const size_t outn = (size_t)nslots * n;
                    std::vector<float> oldo(outn), newo(outn);
                    CK(cudaMemset(dt, 0, outn * 4));
                    inkwell_exl3_gemv_moe_s<<<dim3((tn + 7) / 8, nslots, 1), 256>>>(
                        dd, nullptr, dids, dxhg, nullptr, 0, dt, n, 0, tk, tn, k, n, 1, nullptr, 0, 0);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    CK(cudaMemcpy(oldo.data(), dt, outn * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemset(dt, 0, outn * 4));
                    inkwell_exl3_gemv_moe_s<<<dim3((tn + 7) / 8, nslots, 1), 256>>>(
                        dd, nullptr, dids, dxhg, nullptr, 0, dt, n, 0, tk, tn, k, n, 1, nullptr, 0, 1 << 30);
                    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                    CK(cudaMemcpy(newo.data(), dt, outn * 4, cudaMemcpyDeviceToHost));
                    size_t bad = 0; float maxd = 0.f;
                    for (size_t i = 0; i < outn; ++i) {
                        maxd = std::max(maxd, fabsf(oldo[i] - newo[i]));
                        if (memcmp(&oldo[i], &newo[i], 4) != 0) ++bad;
                    }
                    printf("  exact K=%s badbits=%zu/%zu maxabs=%.3g\n", K.name, bad, outn, maxd);
                    if (bad) { printf("CH2 exactness FAIL\n"); exit(2); }
                }
                for (int r = 0; r < 2; ++r) {
                    t_o = std::min(t_o, (double)timeit([&]{ cudaMemset(dt, 0, (size_t)nslots * n * 4);
                        inkwell_exl3_gemv_moe_s<<<dim3((tn + 7) / 8, nslots, 2), 256>>>(
                            dd, nullptr, dids, dxhg, nullptr, 0, dt, n, 0, tk, tn, k, n, 2, nullptr, 0, 0); }));
                    t_n = std::min(t_n, (double)timeit([&]{ cudaMemset(dt, 0, (size_t)nslots * n * 4);
                        inkwell_exl3_gemv_moe_s<<<dim3((tn + 7) / 8, nslots, 2), 256>>>(
                            dd, nullptr, dids, dxhg, nullptr, 0, dt, n, 0, tk, tn, k, n, 2, nullptr, 0, 1 << 30); }));
                }
                out[sh] = t_o / t_n;
            }
            printf("  dbg K=%s sh=%d old=%.3fms new=%.3fms speedup=%.3f\n", K.name, sh, t_o, t_n, t_o / t_n);
            for (auto p : slotptr) CK(cudaFree(p));
            CK(cudaFree(dxhg)); CK(cudaFree(dt)); CK(cudaFree(dd)); CK(cudaFree(dids));
        }
        printf("%-4s gu=%+.1f%%  dn=%+.1f%%  dense=%+.1f%%\n\n", K.name,
               100.0 * (out[0] - 1), 100.0 * (out[1] - 1), 100.0 * (out[2] - 1));
    }
    return 0;
}
