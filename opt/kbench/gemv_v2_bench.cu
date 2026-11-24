// kbench: v1 (kernels.cu) vs v2 (exl3_gemv_v2.cuh) EXL3 GEMVs on synthetic data.
//   dense : xh [8][2304] -> t for tk=144, tn=72 at ka/halfk in {(3,0),(3,1),(4,0),(5,0)},
//           nky in {1,5,8}; checks every NTOK in {1,2,4,8} against the v1 ntok kernel.
//   moe   : union GEMV over the real router ids of opt/kbench/ids_block.bin with
//           mixed expert widths, descriptors into host-pinned and into device blobs.
// build+run: bash opt/kbench/build.sh gemv_v2_bench && opt/kbench/gemv_v2_bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"
#include "../../crates/inkwell-cuda/src/exl3_gemv_v2.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static float ms_of(cudaEvent_t a, cudaEvent_t b) { float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }

static uint32_t rnd_state = 12345u;
static uint32_t rnd() { rnd_state ^= rnd_state << 13; rnd_state ^= rnd_state >> 17; rnd_state ^= rnd_state << 5; return rnd_state; }

template <typename F> static float timeit(F&& f, int reps = 15) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) { CK(cudaEventRecord(a)); f(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); v.push_back(ms_of(a, b)); }
    std::sort(v.begin(), v.end());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return v[v.size() / 2];
}

static void dense_test() {
    const int K = 2304, N = 1152, tk = K / 16, tn = N / 16, nb = (tn + 3) / 4;
    const int nkys[] = {1, 5, 8};
    struct KH { int ka, halfk; } khs[] = {{3, 0}, {3, 1}, {4, 0}, {5, 0}};
    std::vector<float> xh((size_t)8 * K);
    for (auto& v : xh) v = (int)(rnd() % 2001 - 1000) / 500.f;
    float* dxh; CK(cudaMalloc(&dxh, xh.size() * 4)); CK(cudaMemcpy(dxh, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    printf("\n== dense GEMV (k=%d n=%d): median us, and max rel |v1 - v2| over all rows ==\n", K, N);
    printf("%-8s %4s %10s %10s %10s %10s %10s   %s\n", "ka/half", "nky", "v1 ntok8", "v2 ntok8", "v2 ntok4", "v2 ntok2", "v2 ntok1", "max rel err");
    for (const KH& kh : khs) {
        const int w = exl3v2::words_of(kh.ka, kh.halfk);
        std::vector<uint32_t> trel((size_t)tk * tn * w);
        for (auto& v : trel) v = rnd();
        uint32_t* dtrel; CK(cudaMalloc(&dtrel, trel.size() * 4)); CK(cudaMemcpy(dtrel, trel.data(), trel.size() * 4, cudaMemcpyHostToDevice));
        for (int nky : nkys) {
            std::vector<float> t1((size_t)nky * 8 * N), t2((size_t)nky * 8 * N), t3((size_t)nky * 8 * N);
            float *dt1, *dt2; CK(cudaMalloc(&dt1, t1.size() * 4)); CK(cudaMalloc(&dt2, t2.size() * 4));
            auto v1 = [&]() { inkwell_exl3_gemv_dp4a_ntok<<<dim3(nb, nky), 256>>>(dtrel, dxh, dt1, tk, tn, kh.ka, kh.halfk); };
            v1(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
            CK(cudaMemcpy(t1.data(), dt1, t1.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0; int worst_tt = -1; float tv[8] = {0}, us[4] = {0};
            auto v2n = [&](auto ntc, float* dst) {
                constexpr int NT = decltype(ntc)::value;
                exl3v2::gemv_n<NT><<<dim3(nb, nky), 256>>>(dtrel, dxh, dst, tk, tn, kh.ka, kh.halfk);
            };
            const int nts[4] = {8, 4, 2, 1};
            for (int i = 0; i < 4; ++i) {
                std::fill(t2.begin(), t2.end(), 0.f); CK(cudaMemset(dt2, 0, t2.size() * 4));
                auto v2 = [&]() {
                    if (nts[i] == 8) v2n(std::integral_constant<int, 8>{}, dt2);
                    else if (nts[i] == 4) v2n(std::integral_constant<int, 4>{}, dt2);
                    else if (nts[i] == 2) v2n(std::integral_constant<int, 2>{}, dt2);
                    else v2n(std::integral_constant<int, 1>{}, dt2);
                };
                v2(); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                CK(cudaMemcpy(t3.data(), dt2, t2.size() * 4, cudaMemcpyDeviceToHost));
                // compare the z-sum per row (what had_out adds up)
                for (int tt = 0; tt < nts[i]; ++tt) {
                    double scale = 0, err = 0;
                    for (int col = 0; col < N; ++col) {
                        double a = 0, b = 0;
                        for (int z = 0; z < nky; ++z) { a += t1[((size_t)z * 8 + tt) * N + col]; b += t3[((size_t)z * nts[i] + tt) * N + col]; }
                        scale = std::max(scale, std::fabs(a)); err = std::max(err, std::fabs(a - b));
                    }
                    double rel = err / (scale + 1e-30);
                    if (rel > worst) { worst = rel; worst_tt = tt; }
                }
                us[i] = timeit(v2);
            }
            float u1 = timeit(v1);
            printf("%d/%-6d %4d %10.1f %10.1f %10.1f %10.1f %10.1f   %.2e (row %d)%s\n", kh.ka, kh.halfk, nky, u1 * 1e3, us[0] * 1e3, us[1] * 1e3, us[2] * 1e3, us[3] * 1e3,
                   worst, worst_tt, worst < 2e-5 ? "  ok" : "  <<<< MISMATCH");
            CK(cudaFree(dt1)); CK(cudaFree(dt2));
        }
        CK(cudaFree(dtrel));
    }
    CK(cudaFree(dxh));
}

struct Blobs {
    // per layer-kind: 128 experts x 3 projections, same packing as runner.rs load_st
    std::vector<uint64_t> descs[2];      // [host-backed, dev-backed] u64[3*128*4]
    uint64_t* ddesc[2] = {nullptr, nullptr};
};
static void moe_test(const char* ids_path, int depth) {
    const int KG = 2304, NG = 1152, KD = 1152, ND = 2304;
    const int NE = 128;
    const int khs[6][2] = {{1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}, {4, 0}};
    // build one host blob + one device blob with identical synthetic experts
    std::vector<size_t> trel_b(3 * NE), suh_b(3 * NE), off(3 * NE, 0);
    size_t total = 0;
    for (int e = 0; e < NE; ++e) for (int p = 0; p < 3; ++p) {
        const int i = p * NE + e; const int* kh = khs[(e * 7 + p * 3) % 6];
        trel_b[i] = (size_t)10368 * exl3v2::words_of(kh[0], kh[1]) * 4;
        suh_b[i] = (p == 2 ? 2304 : 4608) + (p == 2 ? 4608 : 2304);
        off[i] = total; total += trel_b[i] + suh_b[i];
    }
    std::vector<uint8_t> host(total);
    for (auto& v : host) v = (uint8_t)rnd();
    void* pinned; CK(cudaHostAlloc(&pinned, total, cudaHostAllocMapped));
    memcpy(pinned, host.data(), total);
    uint8_t* dev; CK(cudaMalloc(&dev, total)); CK(cudaMemcpy(dev, host.data(), total, cudaMemcpyHostToDevice));
    Blobs bl;
    uint64_t base[2] = {(uint64_t)pinned, (uint64_t)dev};
    for (int b = 0; b < 2; ++b) {
        bl.descs[b].assign(3 * NE * 4, 0);
        for (int e = 0; e < NE; ++e) for (int p = 0; p < 3; ++p) {
            const int i = p * NE + e; const int* kh = khs[(e * 7 + p * 3) % 6];
            uint64_t* d = &bl.descs[b][(size_t)p * NE * 4 + e * 4];
            d[0] = base[b] + off[i];
            d[1] = base[b] + off[i] + trel_b[i];
            d[2] = base[b] + off[i] + trel_b[i] + ((p == 2) ? 2304 : 4608);
            d[3] = (uint64_t)(kh[0]) | ((uint64_t)(kh[1]) << 8);
        }
        CK(cudaMalloc(&bl.ddesc[b], bl.descs[b].size() * 8));
        CK(cudaMemcpy(bl.ddesc[b], bl.descs[b].data(), bl.descs[b].size() * 8, cudaMemcpyHostToDevice));
    }
    // scratch
    int *uids, *eids, *euni, *etok, *estart, *epos, *unexp, *ids8;
    float *xh_e, *gu_e, *dn_a, *dn_b;
    CK(cudaMalloc(&uids, 64 * 4)); CK(cudaMalloc(&eids, 64 * 4)); CK(cudaMalloc(&euni, 64 * 4));
    CK(cudaMalloc(&etok, 64 * 4)); CK(cudaMalloc(&estart, 65 * 4)); CK(cudaMalloc(&epos, 64 * 4));
    CK(cudaMalloc(&unexp, 4)); CK(cudaMalloc(&ids8, 64 * 4));
    CK(cudaMalloc(&xh_e, 64 * KG * 4)); CK(cudaMalloc(&gu_e, 64 * KG * 4));
    CK(cudaMalloc(&dn_a, 64 * ND * 4)); CK(cudaMalloc(&dn_b, 64 * ND * 4));
    {
        std::vector<float> v((size_t)64 * KG);
        for (auto& x : v) x = (int)(rnd() % 2001 - 1000) / 500.f;
        CK(cudaMemcpy(xh_e, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        for (auto& x : v) x = (int)(rnd() % 2001 - 1000) / 500.f;
        CK(cudaMemcpy(gu_e, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    }
    // real router ids
    FILE* f = fopen(ids_path, "rb");
    if (!f) { printf("cannot open %s\n", ids_path); exit(1); }
    std::vector<int> rec; { int buf[65]; while (fread(buf, 4, 65, f) == 65) rec.insert(rec.end(), buf, buf + 65); }
    fclose(f);
    const int nv = (int)(rec.size() / 65);
    std::vector<int> layer(nv), ids(nv * 64);
    for (int v = 0; v < nv; ++v) { layer[v] = rec[v * 65]; for (int j = 0; j < 64; ++j) ids[v * 64 + j] = rec[v * 65 + 1 + j]; }
    std::vector<float> a((size_t)64 * ND), b((size_t)64 * ND);
    auto chain = [&](int v, int blob, bool v2, float* dn) {
        const uint64_t* dg = bl.ddesc[blob]; const uint64_t* du = bl.ddesc[blob] + (size_t)NE * 4; const uint64_t* dd = bl.ddesc[blob] + (size_t)2 * NE * 4;
        CK(cudaMemcpy(ids8, &ids[v * 64], 256, cudaMemcpyHostToDevice));
        inkwell_exl3_union_expand<<<1, 128>>>(ids8, uids, unexp, eids, euni, etok, estart, epos, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (!v2) {
            inkwell_exl3_gemv_moe_ntok<<<dim3(NG / 64, 64, 2), 256>>>(dg, du, uids, estart, xh_e, gu_e, 2 * NG, dn, ND, NG, KG / 16, NG / 16, KG, ND, 1, nullptr, 0, unexp);
            inkwell_exl3_gemv_moe_ntok<<<dim3(ND / 64, 64, 1), 256>>>(dd, nullptr, uids, estart, xh_e, nullptr, 0, dn, ND, 0, KD / 16, ND / 16, KD, ND, 1, nullptr, 0, unexp);
        } else if (depth == 2) {
            exl3v2::gemv_moe_n<2><<<dim3(NG / 64, 64, 2), 256>>>(dg, du, uids, estart, xh_e, gu_e, 2 * NG, dn, ND, NG, KG / 16, NG / 16, unexp);
            exl3v2::gemv_moe_n<2><<<dim3(ND / 64, 64, 1), 256>>>(dd, nullptr, uids, estart, xh_e, nullptr, 0, dn, ND, 0, KD / 16, ND / 16, unexp);
        } else if (depth == 3) {
            exl3v2::gemv_moe_n<3><<<dim3(NG / 64, 64, 2), 256>>>(dg, du, uids, estart, xh_e, gu_e, 2 * NG, dn, ND, NG, KG / 16, NG / 16, unexp);
            exl3v2::gemv_moe_n<3><<<dim3(ND / 64, 64, 1), 256>>>(dd, nullptr, uids, estart, xh_e, nullptr, 0, dn, ND, 0, KD / 16, ND / 16, unexp);
        } else {
            exl3v2::gemv_moe_n<4><<<dim3(NG / 64, 64, 2), 256>>>(dg, du, uids, estart, xh_e, gu_e, 2 * NG, dn, ND, NG, KG / 16, NG / 16, unexp);
            exl3v2::gemv_moe_n<4><<<dim3(ND / 64, 64, 1), 256>>>(dd, nullptr, uids, estart, xh_e, nullptr, 0, dn, ND, 0, KD / 16, ND / 16, unexp);
        }
    };
    printf("\n== union MoE GEMV over %d real visits (x %d blocks): correctness + median ms per block ==\n", nv, nv / 48);
    double worst = 0; int worst_v = -1, worst_blob = -1;
    for (int blob = 0; blob < 2; ++blob) for (int v = 0; v < nv; ++v) {
        CK(cudaMemset(dn_a, 0, (size_t)64 * ND * 4)); CK(cudaMemset(dn_b, 0, (size_t)64 * ND * 4));
        chain(v, blob, false, dn_a); chain(v, blob, true, dn_b);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        CK(cudaMemcpy(a.data(), dn_a, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dn_b, b.size() * 4, cudaMemcpyDeviceToHost));
        int cnt = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double rel = std::fabs(a[i] - b[i]) / (std::fabs(a[i]) + 1.0);
            if (rel > 2e-5) ++cnt;
            if (rel > worst) { worst = rel; worst_v = v; worst_blob = blob;
                // dump worst row + owning expert
                std::vector<int> he(65), hu(64);
                CK(cudaMemcpy(he.data(), estart, 65 * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hu.data(), uids, 64 * 4, cudaMemcpyDeviceToHost));
                const int rrow = i / ND, ccol = i % ND;
                int uu = 0; while (uu < 63 && he[uu + 1] <= rrow) ++uu;
                const int e = hu[uu];
                const int* kh = khs[(e * 7 + 2 * 3) % 6];
                printf("   worst elem %d row %d col %d (tile %d): u=%d eid=%d ka=%d halfk=%d nc=%d tt=%d\n",
                       i, rrow, ccol, ccol / 16, uu, e, kh[0], kh[1], he[uu + 1] - he[uu], rrow - he[uu]);
            }
        }
        if (cnt && (v < 2 || v == 48)) {
            double ma = 0, mr = 0; size_t mi = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = std::fabs(a[i] - b[i]);
                if (d > ma) { ma = d; mi = i; }
                const double r = d / (std::fabs(a[i]) + 1.0);
                if (r > mr) mr = r;
            }
            printf("   visit %d blob %d: %d >2e-5 | max abs %.3e at |a|=%.3f | max rel %.3e\n", v, blob, cnt, ma, std::fabs(a[mi]), mr);
        }
    }
    printf("max rel err v1 vs v2: %.2e (visit %d, blob %d)%s\n", worst, worst_v, worst_blob, worst < 2e-5 ? "  ok" : "  <<<< MISMATCH");
    const char* bname[2] = {"host-pinned experts (PCIe)", "device-resident experts (VRAM)"};
    for (int blob = 0; blob < 2; ++blob) {
        auto blk1 = [&]() { for (int v = 0; v < 48; ++v) chain(v, blob, false, dn_a); };
        auto blk2 = [&](int dp) { for (int v = 0; v < 48; ++v) chain(v, blob, true, dn_b); (void)dp; };
        float t1 = timeit(blk1, 5);
        float t2 = timeit([&]() { blk2(2); }, 5), t3 = timeit([&]() { blk2(3); }, 5), t4 = timeit([&]() { blk2(4); }, 5);
        // per-layer split for v1
        printf("%-30s v1 %7.2f ms | v2 depth2 %7.2f | depth3 %7.2f | depth4 %7.2f  (%.2fx)\n", bname[blob], t1, t2, t3, t4, t1 / t4);
    }
}

int main(int argc, char** argv) {
    dense_test();
    moe_test(argc > 1 ? argv[1] : "ids_block.bin", argc > 2 ? atoi(argv[2]) : 4);
    return 0;
}
