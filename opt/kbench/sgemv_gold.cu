// gold test: inkwell_exl3_gemv_moe_s vs inkwell_exl3_gemv_moe vs CPU double ref
// on REAL expert trellis tensors dumped from the rexl3 safetensors by
// opt/kbench/sgemv_dump.py -> /tmp/sgemv_gold/{gate,up,down}_s<slot>.bin + meta.txt
// Covers every K in {1.5,2,2.5,3,3.5,4} for GU (nsel=2) and down (nsel=1),
// skipmode 1/2 with a missf mask, and nky=2.
// build+run: bash opt/kbench/build.sh sgemv_gold && python3 -I opt/kbench/sgemv_dump.py && opt/kbench/sgemv_gold
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
    return v[v.size() / 2];
}

static std::vector<uint32_t> load_bin(const std::string& dir, const char* kind, int slot, size_t nb) {
    char p[512]; snprintf(p, sizeof p, "%s/%s_s%d.bin", dir.c_str(), kind, slot);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s (run sgemv_dump.py)\n", p); exit(1); }
    std::vector<uint32_t> v(nb / 4);
    if (fread(v.data(), 1, nb, f) != nb) { fprintf(stderr, "short read %s\n", p); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/sgemv_gold";
    const int nslots = 8;
    // ---- read meta.txt: kind slot ka hk nbytes name
    KH kh[3][8]; size_t nb[3][8];
    {
        char p[512]; snprintf(p, sizeof p, "%s/meta.txt", dir.c_str());
        FILE* f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "missing %s (run sgemv_dump.py)\n", p); return 1; }
        char kind[16], name[512]; int slot, ka, hk; long long bytes;
        while (fscanf(f, "%15s %d %d %d %lld %511s", kind, &slot, &ka, &hk, &bytes, name) == 6) {
            int ki = kind[0] == 'g' ? 0 : (kind[0] == 'u' ? 1 : 2);
            kh[ki][slot] = {ka, hk}; nb[ki][slot] = (size_t)bytes;
        }
        fclose(f);
    }
    // ---- load trellis tensors -> device
    uint32_t* dtr[3][8];
    std::vector<std::vector<uint32_t>> htr[3];
    for (int ki = 0; ki < 3; ++ki) {
        htr[ki].resize(nslots);
        const char* kind = ki == 0 ? "gate" : (ki == 1 ? "up" : "down");
        for (int s = 0; s < nslots; ++s) {
            htr[ki][s] = load_bin(dir, kind, s, nb[ki][s]);
            CK(cudaMalloc(&dtr[ki][s], nb[ki][s]));
            CK(cudaMemcpy(dtr[ki][s], htr[ki][s].data(), nb[ki][s], cudaMemcpyHostToDevice));
        }
    }
    uint64_t *dd1, *dd2 = nullptr; int* dids; int* dmiss;
    {
        std::vector<uint64_t> d1(nslots * 4, 0), d2(nslots * 4, 0);
        for (int s = 0; s < nslots; ++s) {
            d1[s * 4 + 0] = (uint64_t)(uintptr_t)dtr[0][s]; d1[s * 4 + 3] = (uint64_t)(kh[0][s].ka | (kh[0][s].hk << 8));
            d2[s * 4 + 0] = (uint64_t)(uintptr_t)dtr[1][s]; d2[s * 4 + 3] = (uint64_t)(kh[1][s].ka | (kh[1][s].hk << 8));
        }
        CK(cudaMalloc(&dd1, d1.size() * 8)); CK(cudaMemcpy(dd1, d1.data(), d1.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dd2, d2.size() * 8)); CK(cudaMemcpy(dd2, d2.data(), d2.size() * 8, cudaMemcpyHostToDevice));
        std::vector<int> ids(nslots); for (int i = 0; i < nslots; ++i) ids[i] = i;
        CK(cudaMalloc(&dids, nslots * 4)); CK(cudaMemcpy(dids, ids.data(), nslots * 4, cudaMemcpyHostToDevice));
        int miss[8] = {0, 1, 1, 0, 1, 0, 0, 1};
        CK(cudaMalloc(&dmiss, sizeof miss)); CK(cudaMemcpy(dmiss, miss, sizeof miss, cudaMemcpyHostToDevice));
    }
    int fails = 0;
    struct Stage { const char* name; int k, nout, nsel; };
    const Stage st[2] = {{"GU  2304->2x1152", 2304, 1152, 2}, {"down 1152->2304 ", 1152, 2304, 1}};
    printf("%-16s %-14s %10s  %s\n", "stage", "config", "relerr", "verdict");
    for (int si = 0; si < 2; ++si) {
        const Stage S = st[si];
        const int tk = S.k / 16, tn = S.nout / 16, n = S.nout * S.nsel;
        const int xh2s = S.nsel == 2 ? 2 * S.nout : 0;
        const int njobs = nslots * S.nsel;
        // down descs reuse dtr[2]; GU uses dd1/dd2 built above
        uint64_t* dd_d;
        {
            std::vector<uint64_t> dd(nslots * 4, 0);
            for (int s = 0; s < nslots; ++s) {
                dd[s * 4 + 0] = (uint64_t)(uintptr_t)dtr[2][s];
                dd[s * 4 + 3] = (uint64_t)(kh[2][s].ka | (kh[2][s].hk << 8));
            }
            CK(cudaMalloc(&dd_d, dd.size() * 8)); CK(cudaMemcpy(dd_d, dd.data(), dd.size() * 8, cudaMemcpyHostToDevice));
        }
        const uint64_t* DA = si == 0 ? dd1 : dd_d;
        const uint64_t* DB = si == 0 ? dd2 : nullptr;
        const KH* jkh[2] = {si == 0 ? kh[0] : kh[2], kh[1]};
        const int hki[2] = {si == 0 ? 0 : 2, 1};   // htr index per zsel
        // x: xh[slot*k], xh2[slot*xh2s]
        std::vector<float> xh((size_t)nslots * S.k), xh2((size_t)nslots * (xh2s ? xh2s : 1));
        for (auto& v : xh) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
        if (xh2s) for (auto& v : xh2) { float g = 0; for (int i = 0; i < 4; ++i) g += (int)(rnd() % 2001 - 1000) / 1000.f; v = g * 0.5f; }
        for (int j = 0; j < njobs; ++j) {
            // one outlier per job row, like the bench
            float* xj = (j % S.nsel == 0 ? xh.data() : xh2.data()) + (size_t)(j / S.nsel) * (j % S.nsel == 0 ? S.k : xh2s);
            xj[rnd() % S.k] *= 6.f;
        }
        float *dxh, *dxh2 = nullptr, *dt;
        CK(cudaMalloc(&dxh, xh.size() * 4)); CK(cudaMemcpy(dxh, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
        if (xh2s) { CK(cudaMalloc(&dxh2, xh2.size() * 4)); CK(cudaMemcpy(dxh2, xh2.data(), xh2.size() * 4, cudaMemcpyHostToDevice)); }
        CK(cudaMalloc(&dt, (size_t)nslots * n * 4));
        // CPU ref per job
        std::vector<double> ref((size_t)nslots * n);
        for (int s = 0; s < nslots; ++s)
            for (int z = 0; z < S.nsel; ++z) {
                const float* xj = z ? xh2.data() + (size_t)s * xh2s : xh.data() + (size_t)s * S.k;
                ref_gemv(htr[hki[z]][s].data(), jkh[z][s], xj, tk, tn, &ref[(size_t)s * n + z * S.nout]);
            }
        double ymax = 0; for (double v : ref) ymax = std::max(ymax, fabs(v));
        std::vector<float> outbuf((size_t)nslots * n);
        auto rel = [&]() {
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(outbuf.data(), dt, outbuf.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0; for (size_t i = 0; i < outbuf.size(); ++i) e = std::max(e, fabs(outbuf[i] - ref[i]));
            return e / ymax;
        };
        auto verdict = [&](const char* cfg, double e) {
            const bool ok = e <= 1e-5;
            if (!ok) ++fails;
            printf("%-16s %-14s %10.2e  %s\n", S.name, cfg, e, ok ? "PASS" : "FAIL");
        };
        const int tzoff = S.nsel == 2 ? S.nout : 0;
        // ---- old kernel, nky=1 and nky=2 (baseline correctness + timing)
        auto old1 = [&]() {
            inkwell_exl3_gemv_moe<<<dim3((tn + 3) / 4, nslots, S.nsel), 256>>>(
                DA, DB, dids, dxh, dxh2, xh2s, dt, n, tzoff, tk, tn, S.k, n, 1, nullptr, 0);
        };
        auto old2 = [&]() {
            CK(cudaMemsetAsync(dt, 0, (size_t)nslots * n * 4));
            inkwell_exl3_gemv_moe<<<dim3((tn + 3) / 4, nslots, S.nsel * 2), 256>>>(
                DA, DB, dids, dxh, dxh2, xh2s, dt, n, tzoff, tk, tn, S.k, n, 2, nullptr, 0);
        };
        auto newk = [&](int nkyv) {
            return [&, nkyv]() {
                if (nkyv > 1) CK(cudaMemsetAsync(dt, 0, (size_t)nslots * n * 4));
                inkwell_exl3_gemv_moe_s<<<dim3((tn + 7) / 8, nslots, S.nsel * nkyv), 256>>>(
                    DA, DB, dids, dxh, dxh2, xh2s, dt, n, tzoff, tk, tn, S.k, n, nkyv, nullptr, 0);
            };
        };
        CK(cudaMemset(dt, 0, (size_t)nslots * n * 4)); old1(); verdict("old nky=1", rel());
        CK(cudaMemset(dt, 0, (size_t)nslots * n * 4)); old2(); verdict("old nky=2", rel());
        auto nk1 = newk(1);
        auto nk2 = newk(2);
        CK(cudaMemset(dt, 0, (size_t)nslots * n * 4)); nk1(); verdict("new nky=1", rel());
        CK(cudaMemset(dt, 0, (size_t)nslots * n * 4)); nk2(); verdict("new nky=2", rel());
        // ---- skipmode 1/2: sentinel-prefill t, run, skipped slots must stay sentinel
        const int miss[8] = {0, 1, 1, 0, 1, 0, 0, 1};
        for (int sm = 1; sm <= 2; ++sm) {
            std::vector<float> sent((size_t)nslots * n, -777.f);
            CK(cudaMemcpy(dt, sent.data(), sent.size() * 4, cudaMemcpyHostToDevice));
            inkwell_exl3_gemv_moe_s<<<dim3((tn + 7) / 8, nslots, S.nsel), 256>>>(
                DA, DB, dids, dxh, dxh2, xh2s, dt, n, tzoff, tk, tn, S.k, n, 1, dmiss, sm);
            CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(outbuf.data(), dt, outbuf.size() * 4, cudaMemcpyDeviceToHost));
            double e = 0; int untouched = 1;
            for (int s = 0; s < nslots; ++s) {
                const bool skip = (miss[s] != 0) == (sm == 1);
                for (int i = 0; i < n; ++i) {
                    const float o = outbuf[(size_t)s * n + i];
                    if (skip) { if (o != -777.f) untouched = 0; }
                    else e = std::max(e, fabs(o - ref[(size_t)s * n + i]));
                }
            }
            char cfg[32]; snprintf(cfg, sizeof cfg, "new skipmode=%d", sm);
            const bool ok = untouched && e / ymax <= 1e-5;
            if (!ok) ++fails;
            printf("%-16s %-14s %10.2e  %s%s\n", S.name, cfg, e / ymax,
                   ok ? "PASS" : "FAIL", untouched ? "" : " (skipped slot written!)");
        }
        // ---- timings (median of 31)
        const float ub = timeit(old2) * 1e3f, un = timeit(nk2) * 1e3f;
        printf("%-16s timing: old(nky=2) %7.1f us | new(nky=2) %7.1f us  %.2fx\n", S.name, ub, un, ub / un);
        const float ub1 = timeit(old1) * 1e3f, un1 = timeit(nk1) * 1e3f;
        printf("%-16s timing: old(nky=1) %7.1f us | new(nky=1) %7.1f us  %.2fx\n", S.name, ub1, un1, ub1 / un1);
    }
    printf("\n%s\n", fails ? "GOLD FAIL" : "GOLD PASS");
    return fails ? 1 : 0;
}
