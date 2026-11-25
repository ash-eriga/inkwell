#include "kernels.cu"
extern "C" const unsigned char ink_ch2nopless_cubin_start[1] = {0};
extern "C" const unsigned char ink_ch2nopless_cubin_end[1] = {0};
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float frand(uint32_t& s, float scale) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return ((float)(s >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale;
}

template<typename F>
static float time_one(F&& f, cudaStream_t st, cudaError_t* err) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, st);
    f();
    cudaError_t le = cudaGetLastError();
    cudaEventRecord(b, st);
    cudaEventSynchronize(b);
    float ms = -1.f;
    if (le == cudaSuccess) cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    *err = le;
    return ms;
}

template<typename F, typename G>
static void time_pair(F&& oldf, G&& newf, cudaStream_t st, int reps, float* old_us, float* new_us, cudaError_t* err) {
    for (int i = 0; i < 3; ++i) {
        oldf();
        newf();
    }
    cudaStreamSynchronize(st);
    std::vector<float> ot, nt;
    for (int i = 0; i < reps; ++i) {
        cudaError_t e = cudaSuccess;
        const float o = time_one(oldf, st, &e);
        if (e != cudaSuccess) { *err = e; return; }
        const float n = time_one(newf, st, &e);
        if (e != cudaSuccess) { *err = e; return; }
        ot.push_back(o);
        nt.push_back(n);
    }
    cudaStreamSynchronize(st);
    std::sort(ot.begin(), ot.end());
    std::sort(nt.begin(), nt.end());
    *old_us = ot[ot.size() / 2] * 1000.f;
    *new_us = nt[nt.size() / 2] * 1000.f;
    *err = cudaSuccess;
}

int main() {
    setbuf(stdout, nullptr);
    const int nh = 24, hkv = 8, hd = 128;
    const int cases[] = {1, 7, 32, 63, 64, 65, 128, 200, 319};
    const float scale = 1.0f / sqrtf((float)hd);
    const int reps = getenv("ATTN_GQA3_REPS") ? atoi(getenv("ATTN_GQA3_REPS")) : 61;
    printf("n_kv bitdiffs old_us new_us speedup err\n");
    int bad = 0;
    for (int n_kv : cases) {
        const size_t kvn = (size_t)hkv * n_kv * hd;
        std::vector<__half> hk(kvn), hv(kvn);
        std::vector<float> hq(nh * hd), hb((size_t)nh * n_kv), hm(n_kv + 1, 0.f);
        uint32_t rs = 0x9e3779b9u;
        for (size_t i = 0; i < kvn; ++i) hk[i] = __float2half(frand(rs, 0.37f));
        for (size_t i = 0; i < kvn; ++i) hv[i] = __float2half(frand(rs, 0.29f));
        for (size_t i = 0; i < hq.size(); ++i) hq[i] = frand(rs, 0.43f);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = frand(rs, 0.07f);
        if (n_kv >= 32) {
            for (int j = n_kv - 7; j < n_kv; ++j) hm[j] = -INFINITY;
        }
        void *K = nullptr, *V = nullptr;
        float *Q = nullptr, *B = nullptr, *M = nullptr, *Oo = nullptr, *On = nullptr;
        cudaMalloc(&K, kvn * 2);
        cudaMalloc(&V, kvn * 2);
        cudaMalloc(&Q, hq.size() * 4);
        cudaMalloc(&B, hb.size() * 4);
        cudaMalloc(&M, hm.size() * 4);
        cudaMalloc(&Oo, nh * hd * 4);
        cudaMalloc(&On, nh * hd * 4);
        cudaMemcpy(K, hk.data(), kvn * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(V, hv.data(), kvn * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(Q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(B, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(M, hm.data(), hm.size() * 4, cudaMemcpyHostToDevice);
        cudaStream_t st;
        cudaStreamCreate(&st);
        const long long ks1 = hd * 2, ks2 = (long long)n_kv * hd * 2;
        auto oldf = [&] {
            ink_launch_attn_decode_old(K, V, Q, B, M, Oo, nh, hkv, n_kv, hd,
                ks1, ks2, ks1, ks2, hd, n_kv, scale, 1, st);
        };
        auto newf = [&] {
            ink_launch_attn_decode_gqa3_exact(K, V, Q, B, M, On, hkv, n_kv,
                ks1, ks2, ks1, ks2, hd, n_kv, scale, st);
        };
        oldf();
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaStreamSynchronize(st);
        newf();
        cudaError_t err2 = cudaGetLastError();
        if (err2 == cudaSuccess) err2 = cudaStreamSynchronize(st);
        if (err == cudaSuccess) err = err2;
        int diffs = -1;
        float old_us = -1.f, new_us = -1.f;
        if (err == cudaSuccess) {
            std::vector<float> ho(nh * hd), hn(nh * hd);
            cudaMemcpy(ho.data(), Oo, ho.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hn.data(), On, hn.size() * 4, cudaMemcpyDeviceToHost);
            diffs = 0;
            for (int i = 0; i < nh * hd; ++i) {
                uint32_t x, y;
                memcpy(&x, &ho[i], 4);
                memcpy(&y, &hn[i], 4);
                diffs += x != y;
            }
            time_pair(oldf, newf, st, reps, &old_us, &new_us, &err);
        }
        printf("%5d %8d %7.2f %7.2f %6.2fx %s\n", n_kv, diffs, old_us, new_us,
               old_us > 0.f && new_us > 0.f ? old_us / new_us : 0.f, cudaGetErrorString(err));
        bad += diffs != 0 || err != cudaSuccess;
        cudaStreamDestroy(st);
        cudaFree(K);
        cudaFree(V);
        cudaFree(Q);
        cudaFree(B);
        cudaFree(M);
        cudaFree(Oo);
        cudaFree(On);
    }
    return bad ? 1 : 0;
}
