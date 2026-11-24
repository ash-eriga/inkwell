// gold test: inkwell_exl3_gemv_s vs inkwell_exl3_gemv_dp4a vs CPU double ref
// on REAL dense trellis tensors dumped from the rexl3 safetensors by
// opt/kbench/sgemv_dense_dump.py -> /tmp/sgemv_dense_gold/<name>.bin + meta.txt
// Covers every dense K in {3,3.5,4,5} (wq_du, gate_attn, wr_du, wk_dv, wv_dv,
// wo_ud, unembed), nky in {1,2,4,8}, an empty-chunk run, a INKWELL_SGEMV_DENSE_TGT
// sweep, and a 5-stream concurrent mz-set measurement.
// build+run: bash opt/kbench/build.sh sgemv_dense_gold &&
//   python3 -I opt/kbench/sgemv_dense_dump.py && opt/kbench/sgemv_dense_gold
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint32_t rs = 999331u;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; float v = e ? ldexpf(1.f + m / 1024.f, e - 15) : ldexpf(m / 1024.f, -14); return (h & 0x8000) ? -v : v; }

struct KH { int ka, hk; };
static int nwords(KH k) { return 8 * k.ka + 4 * k.hk; }

// CPU ref straight from the validated extractor formula (kernels.cu exl3_window) + tile_perm
static void ref_gemv(const uint32_t* tr, KH kh, const float* x, int tk, int tn, double* y) {
    const int nw = nwords(kh), R = nw * 32;
    const double KINV = h2f(0x1eee), KC = h2f(0xc931);
    for (int c = 0; c < tn * 16; ++c) y[c] = 0;
    for (int a = 0; a < tk; ++a)
        for (int b = 0; b < tn; ++b) {
            const uint32_t* ring = tr + ((size_t)a * tn + b) * nw;
            for (int i = 0; i < 256; ++i) {
                const int s = kh.hk ? (i + 1) * kh.ka + ((i + 1) >> 1) : (i + 1) * kh.ka;
                const int b0 = s - 16 + (s < 16 ? R : 0);
                const int i0 = b0 >> 5, s0 = 48 - (b0 & 31), i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
                const uint64_t v = ((uint64_t)ring[i0] << 32) | ring[i1];
                uint32_t p = (uint32_t)((v >> s0) & 0xffff) * 0x83DCD12Du;
                const int bs = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
                const int tt = i >> 3, j = i & 7;
                const int r = (tt & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
                const int cc = (tt >> 2) + ((j & 4) ? 8 : 0);
                y[b * 16 + cc] += (double)x[a * 16 + r] * ((1024.0 + bs) * KINV + KC);
            }
        }
}

template <typename F> static float timeit(F&& f, int reps = 31) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) f();
    CK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return v[v.size() / 2];
}

// production nky heuristics
static int pick_old(int tn, int tk) {  // INKWELL_GEMV_NKY_TGT default 512
    const int nb = (tn + 3) / 4; int nky = 1;
    while (nb * nky < 512 && nky < 8 && nky * 16 <= tk) nky *= 2;
    return nky;
}
static int pick_new(int tn, int tk, int tgt) {  // INKWELL_SGEMV_DENSE_TGT
    const int nb = (tn + 7) / 8; int nky = 1;
    while (nb * nky < tgt && nky < 8 && nky * 2 <= tk) nky *= 2;
    return nky;
}

struct Shape {
    std::string name, full;
    int tk, tn, ka, hk; size_t nb;
    uint32_t* dtr = nullptr; float* dxh = nullptr; float* dt = nullptr;
    int k() const { return tk * 16; } int n() const { return tn * 16; }
};

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/sgemv_dense_gold";
    std::vector<Shape> shapes;
    {
        char p[512]; snprintf(p, sizeof p, "%s/meta.txt", dir.c_str());
        FILE* f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "missing %s (run sgemv_dense_dump.py)\n", p); return 1; }
        char nm[64], full[512]; int tk, tn, ka, hk; long long bytes;
        while (fscanf(f, "%63s %d %d %d %d %lld %511s", nm, &tk, &tn, &ka, &hk, &bytes, full) == 7) {
            Shape s; s.name = nm; s.full = full; s.tk = tk; s.tn = tn; s.ka = ka; s.hk = hk; s.nb = (size_t)bytes;
            shapes.push_back(s);
        }
        fclose(f);
    }
    int fails = 0;
    // per-shape results for the sweep: nky -> us
    std::vector<std::vector<float>> tnew(shapes.size(), std::vector<float>(9, 0.f));
    std::vector<float> told(shapes.size());
    std::vector<int> nky_old(shapes.size());
    printf("%-9s %-22s %10s %10s %s\n", "shape", "config", "relerr", "time(us)", "verdict");
    for (size_t si = 0; si < shapes.size(); ++si) {
        Shape& S = shapes[si];
        const int k = S.k(), n = S.n(), tk = S.tk, tn = S.tn;
        // load trellis
        std::vector<uint32_t> htr(S.nb / 4);
        { char p[512]; snprintf(p, sizeof p, "%s/%s.bin", dir.c_str(), S.name.c_str());
          FILE* f = fopen(p, "rb");
          if (!f) { fprintf(stderr, "missing %s\n", p); return 1; }
          if (fread(htr.data(), 1, S.nb, f) != S.nb) { fprintf(stderr, "short read %s\n", p); return 1; }
          fclose(f); }
        CK(cudaMalloc(&S.dtr, S.nb));
        CK(cudaMemcpy(S.dtr, htr.data(), S.nb, cudaMemcpyHostToDevice));
        // x: gaussian-ish + one outlier (same style as sgemv_gold)
        std::vector<float> xh(k);
        for (auto& v : xh) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
        xh[rnd() % k] *= 6.f;
        CK(cudaMalloc(&S.dxh, k * 4));
        CK(cudaMemcpy(S.dxh, xh.data(), k * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&S.dt, (size_t)8 * n * 4));
        // CPU ref
        std::vector<double> ref(n);
        ref_gemv(htr.data(), {S.ka, S.hk}, xh.data(), tk, tn, ref.data());
        double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));
        std::vector<float> tb((size_t)8 * n), sum(n);
        auto sum_and_rel = [&](int nky) {
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(tb.data(), S.dt, (size_t)nky * n * 4, cudaMemcpyDeviceToHost));
            double e = 0;
            for (int i = 0; i < n; ++i) {
                float v = 0.f;
                for (int z = 0; z < nky; ++z) v += tb[(size_t)z * n + i];
                e = std::max(e, fabs((double)v - ref[i]));
            }
            return e / ymax;
        };
        auto verdict = [&](const char* cfg, double e, float us) {
            const bool ok = e <= 1e-5;
            if (!ok) ++fails;
            printf("%-9s %-22s %10.2e %10.1f %s\n", S.name.c_str(), cfg, e, us, ok ? "PASS" : "FAIL");
        };
        // ---- old kernel at production nky
        const int onky = pick_old(tn, tk); nky_old[si] = onky;
        auto oldk = [&]() {
            if (onky > 1) CK(cudaMemsetAsync(S.dt, 0, (size_t)onky * n * 4));
            inkwell_exl3_gemv_dp4a<<<dim3((tn + 3) / 4, onky, 1), 256>>>(
                S.dtr, S.dxh, S.dt, tk, tn, S.ka, S.hk);
        };
        oldk(); verdict("old dp4a", sum_and_rel(onky), 0.f);
        told[si] = timeit(oldk) * 1e3f;
        printf("%-9s %-22s %10s %10.1f\n", S.name.c_str(), "old (timed)", "", told[si]);
        // ---- new kernel at nky in {1,2,4,8}
        for (int nky = 1; nky <= 8; nky <<= 1) {
            auto newk = [&]() {
                if (nky > 1) CK(cudaMemsetAsync(S.dt, 0, (size_t)nky * n * 4));
                inkwell_exl3_gemv_s<<<dim3((tn + 7) / 8, nky, 1), 256>>>(
                    S.dtr, S.dxh, S.dt, tk, tn, S.ka, S.hk);
            };
            char cfg[32]; snprintf(cfg, sizeof cfg, "new nky=%d", nky);
            newk(); const double e = sum_and_rel(nky);
            tnew[si][nky] = timeit(newk) * 1e3f;
            verdict(cfg, e, tnew[si][nky]);
        }
        // ---- empty-chunk robustness: nky > tk on the smallest shape
        if (S.name == "wr_du") {   // smallest shape; nky > tk => all-empty tail chunks write 0
            const int nky = tk + 1;
            std::vector<float> big((size_t)nky * n);
            float* dtl; CK(cudaMalloc(&dtl, (size_t)nky * n * 4));
            inkwell_exl3_gemv_s<<<dim3((tn + 7) / 8, nky, 1), 256>>>(
                S.dtr, S.dxh, dtl, tk, tn, S.ka, S.hk);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(big.data(), dtl, big.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0;
            for (int i = 0; i < n; ++i) {
                float v = 0.f;
                for (int z = 0; z < nky; ++z) v += big[(size_t)z * n + i];
                e = std::max(e, fabs((double)v - ref[i]));
            }
            char cfg[48]; snprintf(cfg, sizeof cfg, "new nky=%d>tk", nky);
            verdict(cfg, e / ymax, 0.f);
            CK(cudaFree(dtl));
        }
        htr.clear(); htr.shrink_to_fit();
    }
    // ---- INKWELL_SGEMV_DENSE_TGT sweep: per-token cost = 48*(wq+gate+wr+wk+wv+wo) + unembed
    {
        // multiplicity per token of each shape
        auto mult = [&](const Shape& s) -> int { return s.name == "unembed" ? 1 : 48; };
        const int tgts[] = {1, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
        std::string ord;
        for (auto& s : shapes) ord += s.name + " ";
        printf("\n%-8s per-shape nky (%s) -> us/token-est\n", "tgt", ord.c_str());
        int best = -1; double bestc = 1e30;
        for (int tgt : tgts) {
            double cost = 0; std::string sel;
            char b[8];
            for (size_t si = 0; si < shapes.size(); ++si) {
                const Shape& S = shapes[si];
                const int nky = pick_new(S.tn, S.tk, tgt);
                snprintf(b, sizeof b, "%d ", nky); sel += b;
                cost += mult(S) * (double)tnew[si][nky];
            }
            printf("%-8d %-45s %9.1f us%s\n", tgt, sel.c_str(), cost,
                   cost < bestc ? (bestc = cost, best = tgt, " *") : "");
        }
        printf("best tgt=%d est %.1f us/token-gemv\n", best, bestc);
        // old-path equivalent for reference (old nky fixed by its own heuristic)
        double oc = 0;
        for (size_t si = 0; si < shapes.size(); ++si) oc += mult(shapes[si]) * (double)told[si];
        printf("old path est %.1f us/token-gemv\n", oc);
    }
    // ---- concurrent mz set: 5 projections on 5 streams, old vs new per tgt
    {
        const char* mz[5] = {"wq_du", "gate_attn", "wr_du", "wk_dv", "wv_dv"};
        Shape* ms[5];
        for (int z = 0; z < 5; ++z) {
            ms[z] = nullptr;
            for (auto& s : shapes) if (s.name == mz[z]) ms[z] = &s;
            if (!ms[z]) { fprintf(stderr, "missing mz shape %s\n", mz[z]); return 1; }
        }
        cudaStream_t st[5]; cudaEvent_t ev0, ev1[5];
        for (int z = 0; z < 5; ++z) CK(cudaStreamCreateWithFlags(&st[z], cudaStreamNonBlocking));
        CK(cudaEventCreate(&ev0)); CK(cudaEventCreateWithFlags(&ev1[0], cudaEventDisableTiming));
        for (int z = 1; z < 5; ++z) CK(cudaEventCreateWithFlags(&ev1[z], cudaEventDisableTiming));
        auto oldall = [&]() {
            cudaEventRecord(ev0);
            for (int z = 0; z < 5; ++z) {
                CK(cudaStreamWaitEvent(st[z], ev0, 0));
                Shape& S = *ms[z]; const int nky = pick_old(S.tn, S.tk);
                if (nky > 1) cudaMemsetAsync(S.dt, 0, (size_t)nky * S.n() * 4, st[z]);
                inkwell_exl3_gemv_dp4a<<<dim3((S.tn + 3) / 4, nky, 1), 256, 0, st[z]>>>(
                    S.dtr, S.dxh, S.dt, S.tk, S.tn, S.ka, S.hk);
            }
            for (int z = 0; z < 5; ++z) CK(cudaStreamSynchronize(st[z]));
        };
        printf("\nconcurrent mz (5 streams): old %.1f us\n", timeit(oldall) * 1e3f);
        const int tgts[] = {24, 32, 48, 64, 128, 256, 512, 1024};
        for (int tgt : tgts) {
            auto newall = [&]() {
                cudaEventRecord(ev0);
                for (int z = 0; z < 5; ++z) {
                    CK(cudaStreamWaitEvent(st[z], ev0, 0));
                    Shape& S = *ms[z]; const int nky = pick_new(S.tn, S.tk, tgt);
                    if (nky > 1) cudaMemsetAsync(S.dt, 0, (size_t)nky * S.n() * 4, st[z]);
                    inkwell_exl3_gemv_s<<<dim3((S.tn + 7) / 8, nky, 1), 256, 0, st[z]>>>(
                        S.dtr, S.dxh, S.dt, S.tk, S.tn, S.ka, S.hk);
                }
                for (int z = 0; z < 5; ++z) CK(cudaStreamSynchronize(st[z]));
            };
            printf("concurrent mz (5 streams): new tgt=%-4d %.1f us\n", tgt, timeit(newall) * 1e3f);
        }
    }
    printf("\n%s\n", fails ? "GOLD FAIL" : "GOLD PASS");
    return fails ? 1 : 0;
}
