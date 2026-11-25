// Context-length probe for inkwell_attn_decode.
// Times the shipped single-block kernel and the split-K kernel for one
// global-attention visit (24 q heads, 8 kv heads, hd 128, F16 K/V).
#include "kernels.cu"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>

__device__ float probe_rand(uint32_t x) {
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    return (float)(x & 0xffffu) * (1.0f / 32768.0f) - 1.0f;
}

__global__ void probe_fill_half(__half* y, size_t n, uint32_t seed, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(probe_rand((uint32_t)i ^ seed) * scale);
}

__global__ void probe_fill_float(float* y, size_t n, uint32_t seed, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = probe_rand((uint32_t)i ^ seed) * scale;
}

static float half_host(uint16_t x) {
    const int exp = (x >> 10) & 31;
    const int mant = x & 1023;
    float v;
    if (exp == 0) v = ldexpf((float)mant / 1024.0f, -14);
    else if (exp == 31) v = mant ? NAN : INFINITY;
    else v = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
    return (x & 0x8000) ? -v : v;
}

static void cpu_ref_one(int h, int nh, int hkv, int n_kv, int hd,
                        const std::vector<uint16_t>& kh, const std::vector<uint16_t>& vh,
                        const std::vector<float>& q, const std::vector<float>& bias,
                        const std::vector<float>& mask, float scale, std::vector<double>& out) {
    const int hk = h / (nh / hkv);
    std::vector<double> score(n_kv);
    double mx = mask[n_kv];
    const size_t row = (size_t)hd;
    const size_t head = (size_t)n_kv * hd;
    for (int j = 0; j < n_kv; ++j) {
        if (!std::isfinite(mask[j])) {
            score[j] = -INFINITY;
            continue;
        }
        double a = 0.0;
        for (int d = 0; d < hd; ++d) {
            a += (double)q[h * hd + d] * (double)half_host(kh[hk * head + j * row + d]);
        }
        const double s = a * (double)scale + (double)bias[h * n_kv + j] + (double)mask[j];
        score[j] = s;
        mx = std::max(mx, s);
    }
    std::vector<double> weight(n_kv);
    double den = exp((double)mask[n_kv] - mx);
    for (int j = 0; j < n_kv; ++j) {
        weight[j] = std::isfinite(mask[j]) ? exp(score[j] - mx) : 0.0;
        den += weight[j];
    }
    for (int d = 0; d < hd; ++d) {
        double o = 0.0;
        for (int j = 0; j < n_kv; ++j) {
            if (std::isfinite(mask[j])) o += weight[j] * (double)half_host(vh[hk * head + j * row + d]);
        }
        out[h * hd + d] = o / den;
    }
}

template<typename F>
static float time_median(F&& f, cudaStream_t st, int reps, cudaError_t* err) {
    std::vector<float> ts;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    cudaError_t e0 = cudaGetLastError();
    cudaError_t s0 = cudaStreamSynchronize(st);
    if (e0 != cudaSuccess || s0 != cudaSuccess) {
        *err = e0 != cudaSuccess ? e0 : s0;
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return -1.0f;
    }
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a, st);
        f();
        cudaError_t le = cudaGetLastError();
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        if (le != cudaSuccess) {
            *err = le;
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return -1.0f;
        }
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        ts.push_back(ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(ts.begin(), ts.end());
    *err = cudaSuccess;
    return ts[ts.size() / 2];
}

static bool read_exact(const std::string& path, void* dst, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read((char*)dst, (std::streamsize)bytes);
    return (size_t)f.gcount() == bytes;
}

static int load_case(const char* dir, int n_kv, int chunk) {
    const int nh = 24, hkv = 8, hd = 128;
    const size_t kvn = (size_t)hkv * n_kv * hd;
    const size_t kvb = kvn * 2;
    std::vector<uint16_t> kh(kvn), vh(kvn);
    std::vector<float> hq(nh * hd), hb((size_t)nh * n_kv), hm(n_kv + 1), hn(nh * hd), ho(nh * hd);
    const std::string d(dir);
    if (!read_exact(d + "/k.f16", kh.data(), kvb) || !read_exact(d + "/v.f16", vh.data(), kvb) ||
        !read_exact(d + "/q.f32", hq.data(), hq.size() * 4) ||
        !read_exact(d + "/bias.f32", hb.data(), hb.size() * 4) ||
        !read_exact(d + "/mask.f32", hm.data(), hm.size() * 4)) {
        printf("load_case n_kv=%d missing_or_short_input\n", n_kv);
        return 2;
    }
    void *K, *V;
    float *Q, *B, *Mk, *Oo, *On;
    cudaMalloc(&K, kvb); cudaMalloc(&V, kvb);
    cudaMalloc(&Q, hq.size() * 4); cudaMalloc(&B, hb.size() * 4);
    cudaMalloc(&Mk, hm.size() * 4); cudaMalloc(&Oo, ho.size() * 4); cudaMalloc(&On, hn.size() * 4);
    cudaMemcpy(K, kh.data(), kvb, cudaMemcpyHostToDevice);
    cudaMemcpy(V, vh.data(), kvb, cudaMemcpyHostToDevice);
    cudaMemcpy(Q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(Mk, hm.data(), hm.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreate(&st);
    const float scale = 1.0f / sqrtf((float)hd);
    const long long ks1 = hd * 2, ks2 = (long long)n_kv * hd * 2;
    ink_launch_attn_decode_old(K, V, Q, B, Mk, Oo, nh, hkv, n_kv, hd, ks1, ks2, ks1, ks2, hd, n_kv, scale, 1, st);
    cudaError_t eo = cudaGetLastError();
    if (eo == cudaSuccess) eo = cudaStreamSynchronize(st);
    ink_launch_attn_decode_split(K, V, Q, B, Mk, On, nh, hkv, n_kv, hd, ks1, ks2, ks1, ks2, hd, n_kv, scale, 1, chunk, st);
    cudaError_t en = cudaGetLastError();
    if (en == cudaSuccess) en = cudaStreamSynchronize(st);
    if (en != cudaSuccess) {
        printf("load_case n_kv=%d split_error=%s\n", n_kv, cudaGetErrorString(en));
        return 3;
    }
    cudaMemcpy(hn.data(), On, hn.size() * 4, cudaMemcpyDeviceToHost);
    double cmp_abs = -1.0, cmp_rel = -1.0;
    if (eo == cudaSuccess) {
        cudaMemcpy(ho.data(), Oo, ho.size() * 4, cudaMemcpyDeviceToHost);
        double scale_out = 0.0;
        for (float x : ho) scale_out = std::max(scale_out, (double)fabs(x));
        for (int i = 0; i < nh * hd; ++i) {
            const double x = fabs((double)ho[i] - (double)hn[i]);
            cmp_abs = std::max(cmp_abs, x);
            cmp_rel = std::max(cmp_rel, x / (scale_out + 1e-20));
        }
    }
    std::vector<double> ref(nh * hd);
    for (int h = 0; h < nh; ++h) cpu_ref_one(h, nh, hkv, n_kv, hd, kh, vh, hq, hb, hm, scale, ref);
    double ref_abs = 0.0, ref_rel = 0.0, old_ref_abs = -1.0, old_ref_rel = -1.0, ref_scale = 0.0;
    for (double x : ref) ref_scale = std::max(ref_scale, fabs(x));
    for (int i = 0; i < nh * hd; ++i) {
        const double x = fabs(ref[i] - (double)hn[i]);
        ref_abs = std::max(ref_abs, x);
        ref_rel = std::max(ref_rel, x / (ref_scale + 1e-20));
        if (eo == cudaSuccess) {
            const double xo = fabs(ref[i] - (double)ho[i]);
            old_ref_abs = std::max(old_ref_abs, xo);
            old_ref_rel = std::max(old_ref_rel, xo / (ref_scale + 1e-20));
        }
    }
    printf("load_case n_kv=%d old=%s old_new_abs=%.6g old_new_rel=%.6g ref_abs=%.6g ref_rel=%.6g old_ref_abs=%.6g old_ref_rel=%.6g\n",
           n_kv, cudaGetErrorString(eo), cmp_abs, cmp_rel, ref_abs, ref_rel, old_ref_abs, old_ref_rel);
    cudaStreamDestroy(st);
    cudaFree(K); cudaFree(V); cudaFree(Q); cudaFree(B); cudaFree(Mk); cudaFree(Oo); cudaFree(On);
    return 0;
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    if (argc >= 4 && strcmp(argv[1], "load") == 0) {
        const int chunk = getenv("INKWELL_ATTNSPLIT_CHUNK") ? atoi(getenv("INKWELL_ATTNSPLIT_CHUNK")) : 512;
        return load_case(argv[2], atoi(argv[3]), chunk);
    }
    const int nh = 24, hkv = 8, hd = 128;
    const int nkvs[] = {64, 128, 256, 320, 384, 448, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    const int reps = getenv("ATTN_PROBE_REPS") ? atoi(getenv("ATTN_PROBE_REPS")) : 15;
    const int chunk_env = getenv("INKWELL_ATTNSPLIT_CHUNK") ? atoi(getenv("INKWELL_ATTNSPLIT_CHUNK")) : 0;
    const int ref_on = getenv("ATTN_PROBE_REF") ? atoi(getenv("ATTN_PROBE_REF")) : 1;
    const int max_nkv = getenv("ATTN_PROBE_MAX") ? atoi(getenv("ATTN_PROBE_MAX")) : 131072;
    printf("chunk=%s reps=%d\n", chunk_env ? "env" : "auto", reps);
    printf("n_kv old_ms new_ms speedup new_GBps old_new_abs old_new_rel ref_abs ref_rel launch\n");
    for (int n_kv : nkvs) {
        if (n_kv > max_nkv) break;
        const int chunk = chunk_env ? chunk_env : (n_kv <= 1024 ? 64 : (n_kv <= 2048 ? 128 : 256));
        const size_t kvb = (size_t)hkv * n_kv * hd * 2;
        void *K = nullptr, *V = nullptr;
        float *Q = nullptr, *B = nullptr, *Mk = nullptr, *Oo = nullptr, *On = nullptr;
        cudaError_t ae = cudaMalloc(&K, kvb);
        ae = ae != cudaSuccess ? ae : cudaMalloc(&V, kvb);
        ae = ae != cudaSuccess ? ae : cudaMalloc(&Q, nh * hd * 4);
        ae = ae != cudaSuccess ? ae : cudaMalloc(&B, (size_t)nh * n_kv * 4);
        ae = ae != cudaSuccess ? ae : cudaMalloc(&Mk, (n_kv + 1) * 4);
        ae = ae != cudaSuccess ? ae : cudaMalloc(&Oo, nh * hd * 4);
        ae = ae != cudaSuccess ? ae : cudaMalloc(&On, nh * hd * 4);
        if (ae != cudaSuccess) {
            printf("%6d alloc_failed=%s\n", n_kv, cudaGetErrorString(ae));
            break;
        }
        const size_t kvn = (size_t)hkv * n_kv * hd;
        const float stress = getenv("ATTN_PROBE_STRESS") ? atof(getenv("ATTN_PROBE_STRESS")) : 1.0f;
        probe_fill_half<<<(unsigned)((kvn + 255) / 256), 256>>>((__half*)K, kvn, 0x1234567u, 0.07f * stress);
        probe_fill_half<<<(unsigned)((kvn + 255) / 256), 256>>>((__half*)V, kvn, 0x7654321u, 0.15f);
        probe_fill_float<<<(nh * hd + 255) / 256, 256>>>(Q, nh * hd, 0xabcdefu, 0.08f * stress);
        probe_fill_float<<<((size_t)nh * n_kv + 255) / 256, 256>>>(B, (size_t)nh * n_kv, 0x2468aceu, 0.03f);
        std::vector<float> hm(n_kv + 1, 0.f);
        for (int j = n_kv - 7; j < n_kv; ++j) if (j >= 0) hm[j] = -INFINITY;
        cudaMemcpy(Mk, hm.data(), hm.size() * 4, cudaMemcpyHostToDevice);
        cudaStream_t st;
        cudaStreamCreate(&st);
        const float scale = 1.0f / sqrtf((float)hd);
        const long long ks1 = hd * 2, ks2 = (long long)n_kv * hd * 2;

        cudaError_t eo = cudaSuccess, en = cudaSuccess;
        auto old_launch = [&] {
            ink_launch_attn_decode_old(K, V, Q, B, Mk, Oo, nh, hkv, n_kv, hd,
                ks1, ks2, ks1, ks2, hd, n_kv, scale, 1, st);
        };
        auto new_launch = [&] {
            ink_launch_attn_decode_split(K, V, Q, B, Mk, On, nh, hkv, n_kv, hd,
                ks1, ks2, ks1, ks2, hd, n_kv, scale, 1, chunk, st);
        };
        const float old_ms = time_median(old_launch, st, reps, &eo);
        const float new_ms = time_median(new_launch, st, reps, &en);
        float red_ms = -1.f;
        if (getenv("ATTN_PROBE_RED")) {
            const int nchunks = (n_kv + chunk - 1) / chunk;
            float* part = inkwell_attn_partbuf((size_t)INKWELL_ATTN_MAX_SPLIT * nh * (hd + 2));
            cudaError_t er = cudaSuccess;
            auto red_launch = [&] {
                inkwell_attn_decode_reduce<<<dim3((unsigned)nh), 128, 0, st>>>(
                    part, Mk, On, nh, n_kv, hd, nchunks);
            };
            red_ms = time_median(red_launch, st, reps, &er);
        }

        double cmp_abs = -1.0, cmp_rel = -1.0, ref_abs = -1.0, ref_rel = -1.0;
        if (eo == cudaSuccess && en == cudaSuccess) {
            std::vector<float> ho(nh * hd), hn(nh * hd);
            cudaMemcpy(ho.data(), Oo, ho.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hn.data(), On, hn.size() * 4, cudaMemcpyDeviceToHost);
            double scale = 0.0;
            for (int i = 0; i < nh * hd; ++i) scale = std::max(scale, (double)fabs(ho[i]));
            for (int i = 0; i < nh * hd; ++i) {
                const double d = fabs((double)ho[i] - (double)hn[i]);
                cmp_abs = std::max(cmp_abs, d);
                cmp_rel = std::max(cmp_rel, d / (scale + 1e-20));
            }
        }
        if (ref_on && en == cudaSuccess && n_kv <= 32768) {
            std::vector<uint16_t> kh(kvn), vh(kvn);
            std::vector<float> hq(nh * hd), hb((size_t)nh * n_kv), hn(nh * hd);
            cudaMemcpy(kh.data(), K, kvn * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(vh.data(), V, kvn * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hq.data(), Q, hq.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), B, hb.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hn.data(), On, hn.size() * 4, cudaMemcpyDeviceToHost);
            std::vector<double> ref(nh * hd);
            for (int h = 0; h < nh; ++h) cpu_ref_one(h, nh, hkv, n_kv, hd, kh, vh, hq, hb, hm, scale, ref);
            double scale = 0.0;
            for (double x : ref) scale = std::max(scale, fabs(x));
            for (int i = 0; i < nh * hd; ++i) {
                const double d = fabs(ref[i] - (double)hn[i]);
                ref_abs = std::max(ref_abs, d);
                ref_rel = std::max(ref_rel, d / (scale + 1e-20));
            }
        }

        const double floor_bytes = (double)kvb * 2.0;
        const double gbps = new_ms > 0 ? floor_bytes / ((double)new_ms * 1e6) : 0.0;
        printf("%6d %6.3f %6.3f %6.2fx %7.1f %.3g %.3g %.3g %.3g red=%.3f old=%s new=%s\n",
               n_kv, old_ms, new_ms, old_ms > 0 && new_ms > 0 ? old_ms / new_ms : 0.0, gbps,
               cmp_abs, cmp_rel, ref_abs, ref_rel, red_ms,
               cudaGetErrorString(eo), cudaGetErrorString(en));
        cudaStreamDestroy(st);
        cudaFree(K); cudaFree(V); cudaFree(Q); cudaFree(B); cudaFree(Mk); cudaFree(Oo); cudaFree(On);
    }
    return 0;
}
