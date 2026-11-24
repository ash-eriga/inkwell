// Comprehensive CUDA Graph microbenchmark & bit-exactness verification for:
//   1. mul_mat_vec_f (mlp.gate, attn_sink_proj, k/v_sconv) + inkwell_router
//   2. Former mul_mat_vec_f custom ops (inkwell_rms2down, inkwell_gf_apply)
//   3. rel_bias sgemm_32x32x32_NT_vec + inkwell_relbias_gather
//   4. MoE Hadamard (inkwell_exl3_had_in_moe, inkwell_exl3_had_out_moe) on real EXL3 weights
//
// Build & run via opt/kbench/mmvf_had/build_and_run.sh (under flock /tmp/inkwell-gpu.lock).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../../../crates/inkwell-cuda/src/kernels.cu"
#include "mmvf_had_kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)

// Deterministic RNG
static uint32_t g_rng = 0x12345678u;
static uint32_t xorshift32() {
    uint32_t x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return g_rng = x;
}
static float randf(float scale = 1.0f) {
    return (((int32_t)(xorshift32() & 0xffffffu) / 8388608.0f) - 1.0f) * scale;
}

// Hardware calibration kernels
__global__ void noop_kernel(int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = 1;
}
__global__ void stream_ldg128(const uint4* __restrict__ src, uint32_t* __restrict__ out, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (; i < n4; i += stride) {
        uint4 v = __ldg(&src[i]);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) out[0] = acc;
}

// Verification helper
struct VerifyStats {
    size_t n;
    size_t bit_diffs;
    uint32_t max_ulp;
    double max_abs;
    double max_rel;
};

static uint32_t ulp_diff(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
    if (ua == ub) return 0;
    if ((ua & 0x7fffffffu) == 0 && (ub & 0x7fffffffu) == 0) return 0; // +0 vs -0
    if ((ua >> 31) != (ub >> 31)) return 0xffffffffu;
    return ua > ub ? ua - ub : ub - ua;
}

static VerifyStats compare_dev(const float* d_ref, const float* d_cand, size_t n) {
    std::vector<float> hr(n), hc(n);
    CK(cudaMemcpy(hr.data(), d_ref, n * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc.data(), d_cand, n * 4, cudaMemcpyDeviceToHost));
    VerifyStats s{n, 0, 0, 0.0, 0.0};
    double max_mag = 0.0;
    for (size_t i = 0; i < n; ++i) max_mag = std::max(max_mag, (double)fabsf(hr[i]));
    for (size_t i = 0; i < n; ++i) {
        uint32_t ur, uc;
        memcpy(&ur, &hr[i], 4); memcpy(&uc, &hc[i], 4);
        if (ur != uc) {
            s.bit_diffs++;
            uint32_t u = ulp_diff(hr[i], hc[i]);
            if (u > s.max_ulp) s.max_ulp = u;
            double ad = fabs((double)hc[i] - (double)hr[i]);
            if (ad > s.max_abs) s.max_abs = ad;
        }
    }
    s.max_rel = max_mag > 0.0 ? s.max_abs / max_mag : s.max_abs;
    return s;
}

static void print_verify(const char* name, const VerifyStats& s) {
    printf("  [VERIFY] %-32s | n=%6zu | bit_diffs=%6zu | max_ulp=%4u | max_abs=%.3e | max_rel=%.3e | %s\n",
           name, s.n, s.bit_diffs, s.max_ulp, s.max_abs, s.max_rel,
           s.bit_diffs == 0 ? "BIT-EXACT" : "REORDERED");
}

static void print_verify_router(const char* name, const float* d_ref, const float* d_cand) {
    float hr[16], hc[16];
    CK(cudaMemcpy(hr, d_ref, 64, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc, d_cand, 64, cudaMemcpyDeviceToHost));
    int id_match = 0, w_bdiff = 0;
    uint32_t max_u = 0;
    double max_a = 0.0;
    for (int i = 0; i < 8; ++i) {
        uint32_t ir, ic, wr, wc;
        memcpy(&ir, &hr[8 + i], 4); memcpy(&ic, &hc[8 + i], 4);
        if (ir == ic) id_match++;
        memcpy(&wr, &hr[i], 4); memcpy(&wc, &hc[i], 4);
        if (wr != wc) {
            w_bdiff++;
            max_u = std::max(max_u, ulp_diff(hr[i], hc[i]));
            max_a = std::max(max_a, fabs((double)hc[i] - (double)hr[i]));
        }
    }
    printf("  [VERIFY] %-32s | ids=%d/8 | w_bit_diffs=%d/8 | max_ulp=%u | max_abs=%.3e | %s\n",
           name, id_match, w_bdiff, max_u, max_a,
           (id_match == 8 && w_bdiff == 0) ? "BIT-EXACT" : (id_match == 8 ? "IDS-EXACT/W-REORD" : "ID-DIFF!"));
}

// CUDA Graph capture & interleaved benchmarking
struct BenchItem {
    std::string name;
    std::function<void(cudaStream_t)> capture_body;
    int inner_iters;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<float> samples_us;
};

static void run_interleaved_graphs(std::vector<BenchItem>& items, int reps = 25) {
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));

    for (auto& it : items) {
        // Warmup once eagerly before capture (important for cuBLAS/first-launch state)
        it.capture_body(st);
        CK(cudaStreamSynchronize(st));
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        it.capture_body(st);
        CK(cudaStreamEndCapture(st, &it.graph));
        CK(cudaGraphInstantiate(&it.exec, it.graph, nullptr, nullptr, 0));
        // Warmup graph
        for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(it.exec, st));
        CK(cudaStreamSynchronize(st));
        it.samples_us.clear();
        it.samples_us.reserve(reps);
    }

    for (int r = 0; r < reps; ++r) {
        for (auto& it : items) {
            CK(cudaEventRecord(ev0, st));
            CK(cudaGraphLaunch(it.exec, st));
            CK(cudaEventRecord(ev1, st));
            CK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, ev0, ev1));
            it.samples_us.push_back((ms * 1000.0f) / (float)it.inner_iters);
        }
    }

    for (auto& it : items) {
        CK(cudaGraphExecDestroy(it.exec));
        CK(cudaGraphDestroy(it.graph));
    }
    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    CK(cudaStreamDestroy(st));
}

static void print_bench_table(const char* title, std::vector<BenchItem>& items, double calls_per_tok) {
    printf("\n--- %s (calls/tok = %.0f) ---\n", title, calls_per_tok);
    printf("  %-36s | %8s | %8s | %8s | %8s | %9s\n",
           "variant", "med (us)", "min (us)", "ms/tok", "speedup", "save (ms)");
    double base_med = 0.0;
    for (size_t i = 0; i < items.size(); ++i) {
        auto v = items[i].samples_us;
        std::sort(v.begin(), v.end());
        double med = v[v.size() / 2];
        double mn = v[0];
        if (i == 0) base_med = med;
        double ms_tok = med * calls_per_tok / 1000.0;
        double sp = base_med / med;
        double save_ms = (base_med - med) * calls_per_tok / 1000.0;
        printf("  %-36s | %8.2f | %8.2f | %8.3f | %7.2fx | %+8.3f\n",
               items[i].name.c_str(), med, mn, ms_tok, sp, save_ms);
    }
}

static std::vector<uint32_t> load_bin(const std::string& dir, const char* kind, int slot, size_t nb) {
    char p[512]; snprintf(p, sizeof p, "%s/%s_s%d.bin", dir.c_str(), kind, slot);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    std::vector<uint32_t> v(nb / 4);
    if (fread(v.data(), 1, nb, f) != nb) { fprintf(stderr, "short read %s\n", p); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const std::string gold_dir = argc > 1 ? argv[1] : "/tmp/sgemv_gold";
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs, L2=%d KB, memClock=%d MHz, bus=%d-bit)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / 1024, prop.memoryClockRate / 1000, prop.memoryBusWidth);

    // ========================================================================
    // SUITE 0: Hardware Calibration (LDG.128 DRAM BW + CUDA Graph node floor)
    // ========================================================================
    {
        size_t nbytes = 16u << 20;
        uint4* d_src; uint32_t* d_out; int* d_noop;
        CK(cudaMalloc(&d_src, nbytes));
        CK(cudaMalloc(&d_out, 16));
        CK(cudaMalloc(&d_noop, 16));
        CK(cudaMemset(d_src, 0x55, nbytes));

        std::vector<BenchItem> cal;
        cal.push_back({"graph_noop_1thr (launch floor)", [&](cudaStream_t st) {
            for (int i = 0; i < 128; ++i) noop_kernel<<<1, 1, 0, st>>>(d_noop);
        }, 128});
        cal.push_back({"stream_ldg128_16MiB", [&](cudaStream_t st) {
            for (int i = 0; i < 8; ++i) stream_ldg128<<<160, 256, 0, st>>>(d_src, d_out, nbytes / 16);
        }, 8});
        run_interleaved_graphs(cal, 25);
        auto v0 = cal[0].samples_us; std::sort(v0.begin(), v0.end());
        auto v1 = cal[1].samples_us; std::sort(v1.begin(), v1.end());
        double bw = (double)nbytes / (v1[v1.size() / 2] * 1e-6) / 1e9;
        printf("\n=== SUITE 0: HARDWARE CALIBRATION ===\n");
        printf("  CUDA Graph empty-kernel floor : med=%.3f us (min=%.3f us)\n", v0[v0.size() / 2], v0[0]);
        printf("  Raw LDG.128 DRAM bandwidth    : med=%.2f us -> %.1f GB/s (16 MiB)\n", v1[v1.size() / 2], bw);
        CK(cudaFree(d_src)); CK(cudaFree(d_out)); CK(cudaFree(d_noop));
    }

    // ========================================================================
    // SUITE 1: mlp.gate [128, 2304] BF16 + inkwell_router (48x / token)
    // ========================================================================
    {
        printf("\n=== SUITE 1: mlp.gate [128, 2304] BF16 + router (48x/tok) ===\n");
        const int N = 128, K = 2304, M = 8; // 8 * 590 KB = 4.7 MB > 3x L2
        std::vector<__nv_bfloat16> hw(N * K);
        std::vector<float> hx(K), hb(N);
        for (auto& v : hw) v = __float2bfloat16(randf(0.1f));
        for (auto& v : hx) v = randf(1.0f);
        for (auto& v : hb) v = randf(0.05f);

        __nv_bfloat16* dw[M];
        for (int m = 0; m < M; ++m) {
            CK(cudaMalloc(&dw[m], (size_t)N * K * 2));
            CK(cudaMemcpy(dw[m], hw.data(), (size_t)N * K * 2, cudaMemcpyHostToDevice));
        }
        float *dx, *db, *dlog_ref, *dlog_c, *dout_ref, *dout_c;
        unsigned int* dctr;
        CK(cudaMalloc(&dx, K * 4)); CK(cudaMemcpy(dx, hx.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&db, N * 4)); CK(cudaMemcpy(db, hb.data(), N * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dlog_ref, N * 4)); CK(cudaMalloc(&dlog_c, N * 4));
        CK(cudaMalloc(&dout_ref, 16 * 4)); CK(cudaMalloc(&dout_c, 16 * 4));
        CK(cudaMalloc(&dctr, 4)); CK(cudaMemset(dctr, 0, 4));

        // Verify correctness
        mh::mmvf_bf16_ref<<<N, 256, 128>>>(dw[0], dx, dlog_ref, K / 2);
        inkwell_router<<<1, 256>>>(dlog_ref, db, dout_ref, 8, 128, 2.5f, 6.103515625e-5f);
        CK(cudaDeviceSynchronize());

        mh::mmvf_bf16_mr<4><<<N / 4, 256>>>(dw[0], dx, dlog_c, N, K / 2);
        CK(cudaDeviceSynchronize());
        print_verify("mmvf_bf16_mr<4> (logits)", compare_dev(dlog_ref, dlog_c, N));

        mh::router_par_128<<<1, 128>>>(dlog_ref, db, dout_c, 8, 2.5f, 6.103515625e-5f);
        CK(cudaDeviceSynchronize());
        print_verify_router("router_par_128", dout_ref, dout_c);

        mh::gate_router_fused_exact<4><<<N / 4, 256>>>(dw[0], dx, db, dlog_c, dctr, dout_c, K / 2, 8, 2.5f, 6.103515625e-5f);
        CK(cudaDeviceSynchronize());
        print_verify_router("gate_router_fused_exact<4>", dout_ref, dout_c);

        mh::gate_router_fused_exact<8><<<N / 8, 256>>>(dw[0], dx, db, dlog_c, dctr, dout_c, K / 2, 8, 2.5f, 6.103515625e-5f);
        CK(cudaDeviceSynchronize());
        print_verify_router("gate_router_fused_exact<8>", dout_ref, dout_c);

        mh::gate_router_fused_ldg128<<<16, 256>>>(dw[0], dx, db, dlog_c, dctr, dout_c, 8, 2.5f, 6.103515625e-5f);
        CK(cudaDeviceSynchronize());
        print_verify("gate_router_ldg128 (logits)", compare_dev(dlog_ref, dlog_c, N));
        print_verify_router("gate_router_ldg128 (out)", dout_ref, dout_c);

        const int GI = 32;
        std::vector<BenchItem> b1;
        b1.push_back({"BASE: mmvf_bf16 + inkwell_router", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::mmvf_bf16_ref<<<N, 256, 128, st>>>(dw[i % M], dx, dlog_ref, K / 2);
                inkwell_router<<<1, 256, 0, st>>>(dlog_ref, db, dout_ref, 8, 128, 2.5f, 6.103515625e-5f);
            }
        }, GI});
        b1.push_back({"  (alone) mmvf_bf16_ref [128,2304]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) mh::mmvf_bf16_ref<<<N, 256, 128, st>>>(dw[i % M], dx, dlog_ref, K / 2);
        }, GI});
        b1.push_back({"  (alone) inkwell_router (serial)", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) inkwell_router<<<1, 256, 0, st>>>(dlog_ref, db, dout_ref, 8, 128, 2.5f, 6.103515625e-5f);
        }, GI});
        b1.push_back({"  (alone) router_par_128 (exact)", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) mh::router_par_128<<<1, 128, 0, st>>>(dlog_ref, db, dout_c, 8, 2.5f, 6.103515625e-5f);
        }, GI});
        b1.push_back({"M1a: mmvf_mr<4> + router_par [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::mmvf_bf16_mr<4><<<N / 4, 256, 0, st>>>(dw[i % M], dx, dlog_c, N, K / 2);
                mh::router_par_128<<<1, 128, 0, st>>>(dlog_c, db, dout_c, 8, 2.5f, 6.103515625e-5f);
            }
        }, GI});
        b1.push_back({"M1b-4: gate_router_exact<4> [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::gate_router_fused_exact<4><<<N / 4, 256, 0, st>>>(dw[i % M], dx, db, dlog_c, dctr, dout_c, K / 2, 8, 2.5f, 6.103515625e-5f);
        }, GI});
        b1.push_back({"M1b-8: gate_router_exact<8> [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::gate_router_fused_exact<8><<<N / 8, 256, 0, st>>>(dw[i % M], dx, db, dlog_c, dctr, dout_c, K / 2, 8, 2.5f, 6.103515625e-5f);
        }, GI});
        b1.push_back({"M1c: gate_router_ldg128 [REORDERED]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::gate_router_fused_ldg128<<<16, 256, 0, st>>>(dw[i % M], dx, db, dlog_c, dctr, dout_c, 8, 2.5f, 6.103515625e-5f);
        }, GI});
        run_interleaved_graphs(b1, 25);
        print_bench_table("SUITE 1: mlp.gate + router", b1, 48.0);

        for (int m = 0; m < M; ++m) CK(cudaFree(dw[m]));
        CK(cudaFree(dx)); CK(cudaFree(db)); CK(cudaFree(dlog_ref)); CK(cudaFree(dlog_c));
        CK(cudaFree(dout_ref)); CK(cudaFree(dout_c)); CK(cudaFree(dctr));
    }

    // ========================================================================
    // SUITE 2: Controllers rms2down [64, 4608] + gf_apply [4608, 64]
    //          (24x3 attn triad + 24x1 ffn = 96x/tok)
    // ========================================================================
    {
        printf("\n=== SUITE 2: Controllers rms2down + gf_apply (96x/tok) ===\n");
        const int N = 2304, R_DN = 64, M = 8; // M=8 sets of 3 controllers -> 24 * 1.18 MB = 28 MB cold!
        std::vector<__nv_bfloat16> hwdn(R_DN * 2 * N), hwgf(2 * N * R_DN);
        std::vector<float> hx0(N), hx1(N);
        for (auto& v : hwdn) v = __float2bfloat16(randf(0.1f));
        for (auto& v : hwgf) v = __float2bfloat16(randf(0.1f));
        for (auto& v : hx0) v = randf(1.0f);
        for (auto& v : hx1) v = randf(1.0f);

        __nv_bfloat16 *dwdn[M][3], *dwgf[M][3];
        float *dx0[3], *dx1[3], *dg_ref[3], *dg_c[3], *dy_ref[3], *dy_c[3];
        for (int b = 0; b < 3; ++b) {
            CK(cudaMalloc(&dx0[b], N * 4)); CK(cudaMemcpy(dx0[b], hx0.data(), N * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dx1[b], N * 4)); CK(cudaMemcpy(dx1[b], hx1.data(), N * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dg_ref[b], R_DN * 4)); CK(cudaMalloc(&dg_c[b], R_DN * 4));
            CK(cudaMalloc(&dy_ref[b], N * 4)); CK(cudaMalloc(&dy_c[b], N * 4));
        }
        for (int m = 0; m < M; ++m) {
            for (int b = 0; b < 3; ++b) {
                CK(cudaMalloc(&dwdn[m][b], (size_t)R_DN * 2 * N * 2));
                CK(cudaMemcpy(dwdn[m][b], hwdn.data(), (size_t)R_DN * 2 * N * 2, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dwgf[m][b], (size_t)2 * N * R_DN * 2));
                CK(cudaMemcpy(dwgf[m][b], hwgf.data(), (size_t)2 * N * R_DN * 2, cudaMemcpyHostToDevice));
            }
        }
        // Device pointer arrays for batched kernels (M entries each of 3 ptrs)
        const float **px0, **px1, **pg_ref, **pg_c;
        const __nv_bfloat16 **pwdn[M], **pwgf[M];
        float **pdg_c, **pdy_c;
        CK(cudaMalloc(&px0, 3 * sizeof(void*))); CK(cudaMemcpy(px0, dx0, 3 * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&px1, 3 * sizeof(void*))); CK(cudaMemcpy(px1, dx1, 3 * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&pg_ref, 3 * sizeof(void*))); CK(cudaMemcpy(pg_ref, dg_ref, 3 * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&pg_c, 3 * sizeof(void*))); CK(cudaMemcpy(pg_c, dg_c, 3 * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&pdg_c, 3 * sizeof(void*))); CK(cudaMemcpy(pdg_c, dg_c, 3 * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&pdy_c, 3 * sizeof(void*))); CK(cudaMemcpy(pdy_c, dy_c, 3 * sizeof(void*), cudaMemcpyHostToDevice));
        for (int m = 0; m < M; ++m) {
            CK(cudaMalloc(&pwdn[m], 3 * sizeof(void*))); CK(cudaMemcpy(pwdn[m], dwdn[m], 3 * sizeof(void*), cudaMemcpyHostToDevice));
            CK(cudaMalloc(&pwgf[m], 3 * sizeof(void*))); CK(cudaMemcpy(pwgf[m], dwgf[m], 3 * sizeof(void*), cudaMemcpyHostToDevice));
        }

        // Verify correctness
        inkwell_rms2down<__nv_bfloat16><<<R_DN, 256>>>(dx0[0], dx1[0], dwdn[0][0], dg_ref[0], N, 1e-6f);
        inkwell_gf_apply<__nv_bfloat16><<<288, 256>>>(dwgf[0][0], dg_ref[0], dx0[0], dx1[0], dy_ref[0], N, N);
        CK(cudaDeviceSynchronize());

        mh::rms2down_nosmem_mr<2, 1><<<R_DN / 2, 256>>>(px0, px1, pwdn[0], pdg_c, N, 1e-6f);
        CK(cudaDeviceSynchronize());
        print_verify("rms2down_nosmem_mr<2,1>", compare_dev(dg_ref[0], dg_c[0], R_DN));

        mh::rms2down_nosmem_mr<4, 1><<<R_DN / 4, 256>>>(px0, px1, pwdn[0], pdg_c, N, 1e-6f);
        CK(cudaDeviceSynchronize());
        print_verify("rms2down_nosmem_mr<4,1>", compare_dev(dg_ref[0], dg_c[0], R_DN));

        mh::rms2down_ldg128<1><<<R_DN / 8, 256>>>(px0, px1, pwdn[0], pdg_c, N, 1e-6f);
        CK(cudaDeviceSynchronize());
        print_verify("rms2down_ldg128<1>", compare_dev(dg_ref[0], dg_c[0], R_DN));

        mh::gf_apply_opt<1><<<288, 256>>>(pwgf[0], pg_ref, px0, px1, pdy_c, N, N);
        CK(cudaDeviceSynchronize());
        print_verify("gf_apply_opt<1>", compare_dev(dy_ref[0], dy_c[0], N));

        mh::gf_apply_ldg128<1><<<72, 256>>>(pwgf[0], pg_ref, px0, px1, pdy_c, N, N);
        CK(cudaDeviceSynchronize());
        print_verify("gf_apply_ldg128<1>", compare_dev(dy_ref[0], dy_c[0], N));

        const int GI = 24;
        // Single-controller table (96x/tok if applied to all 96, or 24x/tok for FFN alone)
        std::vector<BenchItem> b2s;
        b2s.push_back({"BASE: rms2down + gf_apply (1x)", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                inkwell_rms2down<__nv_bfloat16><<<R_DN, 256, 0, st>>>(dx0[0], dx1[0], dwdn[i % M][0], dg_ref[0], N, 1e-6f);
                inkwell_gf_apply<__nv_bfloat16><<<288, 256, 0, st>>>(dwgf[i % M][0], dg_ref[0], dx0[0], dx1[0], dy_ref[0], N, N);
            }
        }, GI});
        b2s.push_back({"  (alone) shipped rms2down", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                inkwell_rms2down<__nv_bfloat16><<<R_DN, 256, 0, st>>>(dx0[0], dx1[0], dwdn[i % M][0], dg_ref[0], N, 1e-6f);
        }, GI});
        b2s.push_back({"  (alone) rms2down_nosmem<2> [EX]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::rms2down_nosmem_mr<2, 1><<<R_DN / 2, 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
        }, GI});
        b2s.push_back({"  (alone) rms2down_nosmem<4> [EX]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::rms2down_nosmem_mr<4, 1><<<R_DN / 4, 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
        }, GI});
        b2s.push_back({"  (alone) rms2down_ldg128 [REORD]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::rms2down_ldg128<1><<<R_DN / 8, 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
        }, GI});
        b2s.push_back({"  (alone) shipped gf_apply", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                inkwell_gf_apply<__nv_bfloat16><<<288, 256, 0, st>>>(dwgf[i % M][0], dg_ref[0], dx0[0], dx1[0], dy_ref[0], N, N);
        }, GI});
        b2s.push_back({"  (alone) gf_apply_opt [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::gf_apply_opt<1><<<288, 256, 0, st>>>(pwgf[i % M], pg_ref, px0, px1, pdy_c, N, N);
        }, GI});
        b2s.push_back({"  (alone) gf_apply_ldg128 [REORD]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::gf_apply_ldg128<1><<<72, 256, 0, st>>>(pwgf[i % M], pg_ref, px0, px1, pdy_c, N, N);
        }, GI});
        b2s.push_back({"M2a-4: nosmem<4> + gf_opt (1x) [EX]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::rms2down_nosmem_mr<4, 1><<<R_DN / 4, 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
                mh::gf_apply_opt<1><<<288, 256, 0, st>>>(pwgf[i % M], pg_c, px0, px1, pdy_c, N, N);
            }
        }, GI});
        b2s.push_back({"M2c: ldg128 pair (1x) [REORDERED]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::rms2down_ldg128<1><<<R_DN / 8, 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
                mh::gf_apply_ldg128<1><<<72, 256, 0, st>>>(pwgf[i % M], pg_c, px0, px1, pdy_c, N, N);
            }
        }, GI});
        run_interleaved_graphs(b2s, 25);
        print_bench_table("SUITE 2A: Single controller (if 96x/tok)", b2s, 96.0);

        // Batched x3 attn controller triad (24 visits/tok = 72 of the 96 sites!)
        std::vector<BenchItem> b2b;
        b2b.push_back({"BASE: 3x (rms2down + gf_apply)", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                for (int b = 0; b < 3; ++b) {
                    inkwell_rms2down<__nv_bfloat16><<<R_DN, 256, 0, st>>>(dx0[b], dx1[b], dwdn[i % M][b], dg_ref[b], N, 1e-6f);
                    inkwell_gf_apply<__nv_bfloat16><<<288, 256, 0, st>>>(dwgf[i % M][b], dg_ref[b], dx0[b], dx1[b], dy_ref[b], N, N);
                }
            }
        }, GI});
        b2b.push_back({"M2b-2: nosmem<2,3> + gf_opt<3> [EX]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::rms2down_nosmem_mr<2, 3><<<dim3(R_DN / 2, 3), 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
                mh::gf_apply_opt<3><<<dim3(288, 3), 256, 0, st>>>(pwgf[i % M], pg_c, px0, px1, pdy_c, N, N);
            }
        }, GI});
        b2b.push_back({"M2b-4: nosmem<4,3> + gf_opt<3> [EX]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::rms2down_nosmem_mr<4, 3><<<dim3(R_DN / 4, 3), 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
                mh::gf_apply_opt<3><<<dim3(288, 3), 256, 0, st>>>(pwgf[i % M], pg_c, px0, px1, pdy_c, N, N);
            }
        }, GI});
        b2b.push_back({"M2c-3: ldg128<3> pair [REORDERED]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::rms2down_ldg128<3><<<dim3(R_DN / 8, 3), 256, 0, st>>>(px0, px1, pwdn[i % M], pdg_c, N, 1e-6f);
                mh::gf_apply_ldg128<3><<<dim3(72, 3), 256, 0, st>>>(pwgf[i % M], pg_c, px0, px1, pdy_c, N, N);
            }
        }, GI});
        run_interleaved_graphs(b2b, 25);
        print_bench_table("SUITE 2B: Batched x3 attn controllers (24 visits/tok = 72 sites)", b2b, 24.0);

        for (int b = 0; b < 3; ++b) {
            CK(cudaFree(dx0[b])); CK(cudaFree(dx1[b])); CK(cudaFree(dg_ref[b]));
            CK(cudaFree(dg_c[b])); CK(cudaFree(dy_ref[b])); CK(cudaFree(dy_c[b]));
        }
        for (int m = 0; m < M; ++m) {
            for (int b = 0; b < 3; ++b) { CK(cudaFree(dwdn[m][b])); CK(cudaFree(dwgf[m][b])); }
            CK(cudaFree(pwdn[m])); CK(cudaFree(pwgf[m]));
        }
        CK(cudaFree(px0)); CK(cudaFree(px1)); CK(cudaFree(pg_ref)); CK(cudaFree(pg_c));
        CK(cudaFree(pdg_c)); CK(cudaFree(pdy_c));
    }

    // ========================================================================
    // SUITE 3: attn_sink_proj [24, 2304] + rel_bias_pad [1025, 16] * r [16, 24]
    //          + relbias_gather (48x/tok; tested at n_kv = 64 and n_kv = 512)
    // ========================================================================
    {
        printf("\n=== SUITE 3: attn_sink_proj + rel_bias sgemm + relbias_gather (48x/tok) ===\n");
        cublasHandle_t cublas;
        CB(cublasCreate(&cublas));
        const int NH = 24, K_INP = 2304, EXT = 1025, NREL = 16, M = 16;
        std::vector<__nv_bfloat16> hwsink(NH * K_INP), hrb_bf(EXT * NREL);
        std::vector<float> hrb_f32(EXT * NREL), hinp0(K_INP), hr(NREL * NH);
        for (auto& v : hwsink) v = __float2bfloat16(randf(0.1f));
        for (int i = 0; i < EXT * NREL; ++i) {
            __nv_bfloat16 bv = (i >= 1024 * NREL) ? __float2bfloat16(0.0f) : __float2bfloat16(randf(0.2f));
            hrb_bf[i] = bv;
            hrb_f32[i] = __bfloat162float(bv);
        }
        for (auto& v : hinp0) v = randf(1.0f);
        for (auto& v : hr) v = randf(1.0f);

        __nv_bfloat16 *dwsink[M], *drb_bf[M];
        float *drb_f32[M];
        for (int m = 0; m < M; ++m) {
            CK(cudaMalloc(&dwsink[m], NH * K_INP * 2));
            CK(cudaMemcpy(dwsink[m], hwsink.data(), NH * K_INP * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&drb_bf[m], EXT * NREL * 2));
            CK(cudaMemcpy(drb_bf[m], hrb_bf.data(), EXT * NREL * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&drb_f32[m], EXT * NREL * 4));
            CK(cudaMemcpy(drb_f32[m], hrb_f32.data(), EXT * NREL * 4, cudaMemcpyHostToDevice));
        }
        float *dinp0, *dr, *dsinks, *dtable, *dkqb_ref, *dkqb_c;
        int *drelpos;
        CK(cudaMalloc(&dinp0, K_INP * 4)); CK(cudaMemcpy(dinp0, hinp0.data(), K_INP * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dr, NREL * NH * 4)); CK(cudaMemcpy(dr, hr.data(), NREL * NH * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dsinks, NH * 4));
        CK(cudaMalloc(&dtable, EXT * NH * 4));
        CK(cudaMalloc(&dkqb_ref, 512 * NH * 4));
        CK(cudaMalloc(&dkqb_c, 512 * NH * 4));
        CK(cudaMalloc(&drelpos, 512 * 4));

        const float alpha = 1.0f, beta = 0.0f;
        for (int n_kv : {64, 512}) {
            std::vector<int> hpos(n_kv);
            for (int j = 0; j < n_kv; ++j) hpos[j] = (j < n_kv - 4) ? (n_kv - 1 - j) : 1024;
            CK(cudaMemcpy(drelpos, hpos.data(), n_kv * 4, cudaMemcpyHostToDevice));

            // Baseline: cublasSgemm + mmvf(sink) + relbias_gather
            CB(cublasSetStream(cublas, 0));
            CB(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, EXT, NH, NREL,
                           &alpha, drb_f32[0], NREL, dr, NREL, &beta, dtable, EXT));
            mh::mmvf_bf16_ref<<<NH, 256, 128>>>(dwsink[0], dinp0, dsinks, K_INP / 2);
            inkwell_relbias_gather<<<(n_kv * NH + 255) / 256, 256>>>(dtable, dsinks, drelpos, dkqb_ref, n_kv, NH, EXT);
            CK(cudaDeviceSynchronize());

            mh::sink_relbias_fused<<<NH, 256>>>(dwsink[0], dinp0, drb_bf[0], dr, drelpos, dkqb_c, n_kv, K_INP / 2);
            CK(cudaDeviceSynchronize());
            char lbl[64]; snprintf(lbl, sizeof lbl, "sink_relbias_fused (n_kv=%d)", n_kv);
            print_verify(lbl, compare_dev(dkqb_ref, dkqb_c, (size_t)n_kv * NH));

            const int GI = 32;
            std::vector<BenchItem> b3;
            b3.push_back({"BASE: sgemm + mmvf(sink) + gather", [&](cudaStream_t st) {
                CB(cublasSetStream(cublas, st));
                for (int i = 0; i < GI; ++i) {
                    CB(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, EXT, NH, NREL,
                                   &alpha, drb_f32[i % M], NREL, dr, NREL, &beta, dtable, EXT));
                    mh::mmvf_bf16_ref<<<NH, 256, 128, st>>>(dwsink[i % M], dinp0, dsinks, K_INP / 2);
                    inkwell_relbias_gather<<<(n_kv * NH + 255) / 256, 256, 0, st>>>(dtable, dsinks, drelpos, dkqb_ref, n_kv, NH, EXT);
                }
            }, GI});
            b3.push_back({"  (alone) cublasSgemm [1025,24,16]", [&](cudaStream_t st) {
                CB(cublasSetStream(cublas, st));
                for (int i = 0; i < GI; ++i)
                    CB(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, EXT, NH, NREL,
                                   &alpha, drb_f32[i % M], NREL, dr, NREL, &beta, dtable, EXT));
            }, GI});
            b3.push_back({"  (alone) mmvf_bf16_ref (sink 24)", [&](cudaStream_t st) {
                for (int i = 0; i < GI; ++i)
                    mh::mmvf_bf16_ref<<<NH, 256, 128, st>>>(dwsink[i % M], dinp0, dsinks, K_INP / 2);
            }, GI});
            b3.push_back({"M3a: sink_relbias_fused (1 launch)", [&](cudaStream_t st) {
                for (int i = 0; i < GI; ++i)
                    mh::sink_relbias_fused<<<NH, 256, 0, st>>>(dwsink[i % M], dinp0, drb_bf[i % M], dr, drelpos, dkqb_c, n_kv, K_INP / 2);
            }, GI});
            run_interleaved_graphs(b3, 25);
            char ttl[96]; snprintf(ttl, sizeof ttl, "SUITE 3: sink + relbias (n_kv=%d)", n_kv);
            print_bench_table(ttl, b3, 48.0);
        }

        for (int m = 0; m < M; ++m) { CK(cudaFree(dwsink[m])); CK(cudaFree(drb_bf[m])); CK(cudaFree(drb_f32[m])); }
        CK(cudaFree(dinp0)); CK(cudaFree(dr)); CK(cudaFree(dsinks)); CK(cudaFree(dtable));
        CK(cudaFree(dkqb_ref)); CK(cudaFree(dkqb_c)); CK(cudaFree(drelpos));
        CB(cublasDestroy(cublas));
    }

    // ========================================================================
    // SUITE 4: k/v_sconv.weight_proj [32, 2304] BF16 x 2 (8 full visits/tok)
    // ========================================================================
    {
        printf("\n=== SUITE 4: k_sconv + v_sconv [32, 2304] BF16 x2 (8 visits/tok = 16 launches) ===\n");
        const int N = 32, K = 2304, M = 16;
        std::vector<__nv_bfloat16> hwk(N * K), hwv(N * K);
        std::vector<float> hxk(K), hxv(K);
        for (auto& v : hwk) v = __float2bfloat16(randf(0.1f));
        for (auto& v : hwv) v = __float2bfloat16(randf(0.1f));
        for (auto& v : hxk) v = randf(1.0f);
        for (auto& v : hxv) v = randf(1.0f);

        __nv_bfloat16 *dwk[M], *dwv[M];
        for (int m = 0; m < M; ++m) {
            CK(cudaMalloc(&dwk[m], N * K * 2)); CK(cudaMemcpy(dwk[m], hwk.data(), N * K * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc(&dwv[m], N * K * 2)); CK(cudaMemcpy(dwv[m], hwv.data(), N * K * 2, cudaMemcpyHostToDevice));
        }
        float *dxk, *dxv, *dk_ref, *dv_ref, *dk_c, *dv_c;
        CK(cudaMalloc(&dxk, K * 4)); CK(cudaMemcpy(dxk, hxk.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxv, K * 4)); CK(cudaMemcpy(dxv, hxv.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dk_ref, N * 4)); CK(cudaMalloc(&dv_ref, N * 4));
        CK(cudaMalloc(&dk_c, N * 4)); CK(cudaMalloc(&dv_c, N * 4));

        mh::mmvf_bf16_ref<<<N, 256, 128>>>(dwk[0], dxk, dk_ref, K / 2);
        mh::mmvf_bf16_ref<<<N, 256, 128>>>(dwv[0], dxv, dv_ref, K / 2);
        mh::kv_sconv_batched<2><<<dim3(N / 2, 2), 256>>>(dwk[0], dwv[0], dxk, dxv, dk_c, dv_c, N, K / 2);
        CK(cudaDeviceSynchronize());
        print_verify("kv_sconv_batched<2> (k)", compare_dev(dk_ref, dk_c, N));
        print_verify("kv_sconv_batched<2> (v)", compare_dev(dv_ref, dv_c, N));

        const int GI = 32;
        std::vector<BenchItem> b4;
        b4.push_back({"BASE: 2x mmvf_bf16_ref [32,2304]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i) {
                mh::mmvf_bf16_ref<<<N, 256, 128, st>>>(dwk[i % M], dxk, dk_ref, K / 2);
                mh::mmvf_bf16_ref<<<N, 256, 128, st>>>(dwv[i % M], dxv, dv_ref, K / 2);
            }
        }, GI});
        b4.push_back({"M4-1: kv_sconv_batched<1> [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::kv_sconv_batched<1><<<dim3(N, 2), 256, 0, st>>>(dwk[i % M], dwv[i % M], dxk, dxv, dk_c, dv_c, N, K / 2);
        }, GI});
        b4.push_back({"M4-2: kv_sconv_batched<2> [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI; ++i)
                mh::kv_sconv_batched<2><<<dim3(N / 2, 2), 256, 0, st>>>(dwk[i % M], dwv[i % M], dxk, dxv, dk_c, dv_c, N, K / 2);
        }, GI});
        run_interleaved_graphs(b4, 25);
        print_bench_table("SUITE 4: k/v_sconv batched", b4, 8.0);

        for (int m = 0; m < M; ++m) { CK(cudaFree(dwk[m])); CK(cudaFree(dwv[m])); }
        CK(cudaFree(dxk)); CK(cudaFree(dxv)); CK(cudaFree(dk_ref)); CK(cudaFree(dv_ref));
        CK(cudaFree(dk_c)); CK(cudaFree(dv_c));
    }

    // ========================================================================
    // SUITE 5: MoE Hadamard + gemv_moe_s on REAL EXL3 weights (/tmp/sgemv_gold)
    // ========================================================================
    {
        printf("\n=== SUITE 5: MoE Hadamard + gemv_moe_s on REAL EXL3 weights ===\n");
        const int nslots = 8, KG = 2304, NG = 1152, KD = 1152, ND = 2304;
        int ka[3][8], hk[3][8]; size_t nb[3][8];
        {
            char p[512]; snprintf(p, sizeof p, "%s/meta.txt", gold_dir.c_str());
            FILE* f = fopen(p, "rb");
            if (!f) { fprintf(stderr, "missing %s\n", p); return 1; }
            char kind[16], name[512]; int slot, a, h; long long bytes;
            while (fscanf(f, "%15s %d %d %d %lld %511s", kind, &slot, &a, &h, &bytes, name) == 6) {
                int ki = kind[0] == 'g' ? 0 : (kind[0] == 'u' ? 1 : 2);
                ka[ki][slot] = a; hk[ki][slot] = h; nb[ki][slot] = (size_t)bytes;
            }
            fclose(f);
        }

        uint32_t* dtr[3][8];
        uint16_t *dsuh[3][8], *dsvh[3][8];
        for (int ki = 0; ki < 3; ++ki) {
            const char* kind = ki == 0 ? "gate" : (ki == 1 ? "up" : "down");
            int suh_n = (ki == 2) ? KD : KG;
            int svh_n = (ki == 2) ? ND : NG;
            for (int s = 0; s < nslots; ++s) {
                auto htr = load_bin(gold_dir, kind, s, nb[ki][s]);
                CK(cudaMalloc(&dtr[ki][s], nb[ki][s]));
                CK(cudaMemcpy(dtr[ki][s], htr.data(), nb[ki][s], cudaMemcpyHostToDevice));
                std::vector<uint16_t> hs1(suh_n), hs2(svh_n);
                for (auto& v : hs1) v = __half_as_ushort(__float2half((xorshift32() & 1) ? 1.0f : -1.0f));
                for (auto& v : hs2) v = __half_as_ushort(__float2half((xorshift32() & 1) ? 1.0f : -1.0f));
                CK(cudaMalloc(&dsuh[ki][s], suh_n * 2));
                CK(cudaMemcpy(dsuh[ki][s], hs1.data(), suh_n * 2, cudaMemcpyHostToDevice));
                CK(cudaMalloc(&dsvh[ki][s], svh_n * 2));
                CK(cudaMemcpy(dsvh[ki][s], hs2.data(), svh_n * 2, cudaMemcpyHostToDevice));
            }
        }

        // Build BOTH VRAM descriptors (H1) AND pinned host-mapped descriptors (H0 = today's on_dev!)
        uint64_t *dd_vram[3], *dd_pin_h[3], *dd_pin_d[3];
        for (int ki = 0; ki < 3; ++ki) {
            std::vector<uint64_t> hd(128 * 4, 0);
            for (int s = 0; s < nslots; ++s) {
                hd[s * 4 + 0] = (uint64_t)(uintptr_t)dtr[ki][s];
                hd[s * 4 + 1] = (uint64_t)(uintptr_t)dsuh[ki][s];
                hd[s * 4 + 2] = (uint64_t)(uintptr_t)dsvh[ki][s];
                hd[s * 4 + 3] = (uint64_t)(ka[ki][s] | (hk[ki][s] << 8));
            }
            CK(cudaMalloc(&dd_vram[ki], 128 * 32));
            CK(cudaMemcpy(dd_vram[ki], hd.data(), 128 * 32, cudaMemcpyHostToDevice));
            CK(cudaHostAlloc(&dd_pin_h[ki], 128 * 32, cudaHostAllocMapped));
            memcpy(dd_pin_h[ki], hd.data(), 128 * 32);
            CK(cudaHostGetDevicePointer(&dd_pin_d[ki], dd_pin_h[ki], 0));
        }

        int* dids;
        int hids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        CK(cudaMalloc(&dids, 32)); CK(cudaMemcpy(dids, hids, 32, cudaMemcpyHostToDevice));

        std::vector<float> hx(KG), hw(nslots);
        for (auto& v : hx) v = randf(0.5f);
        for (int s = 0; s < nslots; ++s) hw[s] = 0.1f + 0.05f * s;

        float *dx, *dw, *dxh, *dgu, *dhid, *ddn, *dout_ref, *dout_c, *dxh_ref, *dgu_ref;
        CK(cudaMalloc(&dx, KG * 4)); CK(cudaMemcpy(dx, hx.data(), KG * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dw, nslots * 4)); CK(cudaMemcpy(dw, hw.data(), nslots * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dxh, nslots * KG * 4));
        CK(cudaMalloc(&dxh_ref, nslots * KG * 4));
        CK(cudaMalloc(&dgu, nslots * 2 * NG * 4));
        CK(cudaMalloc(&dgu_ref, nslots * 2 * NG * 4));
        CK(cudaMalloc(&dhid, nslots * NG * 4));
        CK(cudaMalloc(&ddn, nslots * ND * 4));
        CK(cudaMalloc(&dout_ref, ND * 4));
        CK(cudaMalloc(&dout_c, ND * 4));

        // Clean helper lambdas
        auto launch_shipped = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            inkwell_exl3_had_in_moe<<<dim3(5, nslots, 2), 128, 0, st>>>(G, U, dids, dx, dxh, dgu, 2*NG, KG, nullptr, 0, nullptr, nullptr);
            inkwell_exl3_gemv_moe_s<<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dxh, dgu, 2*NG, ddn, ND, NG, KG/16, NG/16, KG, ND, 1, nullptr, 0, 0);
            inkwell_exl3_had_out_moe<<<dim3(3, nslots, 2), 128, 0, st>>>(G, U, dids, ddn, dgu, NG, 2*NG, 0, ND, NG);
            inkwell_situ_pair<<<(nslots*NG+255)/256, 256, 0, st>>>(dgu, nullptr, dhid, 4.f, 25.f, NG, nslots);
            inkwell_exl3_had_in_moe<<<dim3(3, nslots, 1), 128, 0, st>>>(D, nullptr, dids, dhid, dxh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
            inkwell_exl3_gemv_moe_s<<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND, 1, nullptr, 0, 0);
            inkwell_exl3_had_out_moe<<<dim3(5, nslots, 1), 128, 0, st>>>(D, nullptr, dids, ddn, dxh, ND, ND, 0, ND, 0);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(dxh, dw, nullptr, out, ND, nslots);
        };

        auto launch_h2_hadout = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            inkwell_exl3_had_in_moe<<<dim3(5, nslots, 2), 128, 0, st>>>(G, U, dids, dx, dxh, dgu, 2*NG, KG, nullptr, 0, nullptr, nullptr);
            mh::gemv_moe_s_fused<false, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dxh, dgu, 2*NG, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            inkwell_situ_pair<<<(nslots*NG+255)/256, 256, 0, st>>>(dgu, nullptr, dhid, 4.f, 25.f, NG, nslots);
            inkwell_exl3_had_in_moe<<<dim3(3, nslots, 1), 128, 0, st>>>(D, nullptr, dids, dhid, dxh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
            mh::gemv_moe_s_fused<false, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, dxh, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(dxh, dw, nullptr, out, ND, nslots);
        };

        auto launch_h3a_both = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::gemv_moe_s_fused<true, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dx, nullptr, 0, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            inkwell_situ_pair<<<(nslots*NG+255)/256, 256, 0, st>>>(dgu, nullptr, dhid, 4.f, 25.f, NG, nslots);
            mh::gemv_moe_s_fused<true, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dhid, nullptr, 0, dxh, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(dxh, dw, nullptr, out, ND, nslots);
        };

        auto launch_h3b_hybrid = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::gemv_moe_s_fused<true, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dx, nullptr, 0, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, D, dids, dxh, 4.f, 25.f, NG);
            mh::gemv_moe_s_fused<false, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, dxh, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(dxh, dw, nullptr, out, ND, nslots);
        };

        auto launch_h3c_vec_sep = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::had_in_moe_vec<<<dim3(9, nslots, 2), 64, 0, st>>>(G, U, dids, dx, dxh, dgu, 2*NG, KG);
            mh::gemv_moe_s_fused<false, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dxh, dgu, 2*NG, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, D, dids, dxh, 4.f, 25.f, NG);
            mh::gemv_moe_s_fused<false, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, dxh, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(dxh, dw, nullptr, out, ND, nslots);
        };

        auto launch_h3d_outwsum = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::gemv_moe_s_fused<true, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dx, nullptr, 0, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, D, dids, dxh, 4.f, 25.f, NG);
            mh::gemv_moe_s_fused<false, false><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND);
            mh::had_out_wsum_moe<<<18, 256, 0, st>>>(D, dids, ddn, dw, out, ND);
        };

        // Note: in launch_h2_hadout, `dgu` is BOTH input `xh2` AND output `y_out`!
        // Wait: in shipped chain, `had_in_moe` wrote up's xh2 into `dgu`, while `gemv_moe_s` wrote `ddn`!
        // If `gemv_moe_s_fused<false, true>` writes directly to `dgu` while OTHER blocks of `gemv_moe_s`
        // are still reading `dgu` (as `xh2`), that would RACE! So pass `ddn` as `xh2` instead (same size: 8*2304)!
        auto launch_h2_safe = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            inkwell_exl3_had_in_moe<<<dim3(5, nslots, 2), 128, 0, st>>>(G, U, dids, dx, dxh, ddn, 2*NG, KG, nullptr, 0, nullptr, nullptr);
            mh::gemv_moe_s_fused<false, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dxh, ddn, 2*NG, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            inkwell_situ_pair<<<(nslots*NG+255)/256, 256, 0, st>>>(dgu, nullptr, dhid, 4.f, 25.f, NG, nslots);
            inkwell_exl3_had_in_moe<<<dim3(3, nslots, 1), 128, 0, st>>>(D, nullptr, dids, dhid, dxh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
            mh::gemv_moe_s_fused<false, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(ddn, dw, nullptr, out, ND, nslots);
        };
        auto launch_h3b_safe = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::gemv_moe_s_fused<true, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dx, nullptr, 0, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, D, dids, dxh, 4.f, 25.f, NG);
            mh::gemv_moe_s_fused<false, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(ddn, dw, nullptr, out, ND, nslots);
        };
        auto launch_h3c_safe = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::had_in_moe_vec<<<dim3(9, nslots, 2), 64, 0, st>>>(G, U, dids, dx, dxh, ddn, 2*NG, KG);
            mh::gemv_moe_s_fused<false, true><<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dxh, ddn, 2*NG, dgu, 2*NG, NG, KG/16, NG/16, KG, NG);
            mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, D, dids, dxh, 4.f, 25.f, NG);
            mh::gemv_moe_s_fused<false, true><<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND);
            inkwell_expert_wsum<<<(ND+255)/256, 256, 0, st>>>(ddn, dw, nullptr, out, ND, nslots);
        };
        auto launch_h3e_clean6k = [&](uint64_t* G, uint64_t* U, uint64_t* D, float* out, cudaStream_t st) {
            mh::had_in_moe_vec<<<dim3(9, nslots, 2), 64, 0, st>>>(G, U, dids, dx, dxh, dgu, 2*NG, KG);
            inkwell_exl3_gemv_moe_s<<<dim3(9, nslots, 2), 256, 0, st>>>(G, U, dids, dxh, dgu, 2*NG, ddn, ND, NG, KG/16, NG/16, KG, ND, 1, nullptr, 0, 0);
            mh::had_out_moe_vec<<<dim3(3, nslots, 2), 96, 0, st>>>(G, U, dids, ddn, dgu, NG, 2*NG, 0, ND, NG);
            mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, D, dids, dxh, 4.f, 25.f, NG);
            inkwell_exl3_gemv_moe_s<<<dim3(18, nslots, 1), 256, 0, st>>>(D, nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND, 1, nullptr, 0, 0);
            mh::had_out_wsum_moe<<<18, 256, 0, st>>>(D, dids, ddn, dw, out, ND);
        };
        (void)launch_h2_hadout; (void)launch_h3b_hybrid; (void)launch_h3c_vec_sep;

        // Verify ALL MoE variants end-to-end vs shipped chain!
        launch_shipped(dd_vram[0], dd_vram[1], dd_vram[2], dout_ref, 0);
        CK(cudaDeviceSynchronize());

        launch_shipped(dd_pin_d[0], dd_pin_d[1], dd_pin_d[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H0_pinned vs H1_vram (e2e)", compare_dev(dout_ref, dout_c, ND));

        launch_h2_safe(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H2_hadout (e2e)", compare_dev(dout_ref, dout_c, ND));

        launch_h3a_both(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H3a_hadin+hadout (e2e)", compare_dev(dout_ref, dout_c, ND));

        launch_h3b_safe(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H3b_hybrid (e2e)", compare_dev(dout_ref, dout_c, ND));

        launch_h3c_safe(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H3c_vec+situ+hadout (e2e)", compare_dev(dout_ref, dout_c, ND));

        launch_h3d_outwsum(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H3d_hadout_wsum (e2e)", compare_dev(dout_ref, dout_c, ND));

        launch_h3e_clean6k(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, 0);
        CK(cudaDeviceSynchronize());
        print_verify("H3e_clean6k_shipped_gemv (e2e)", compare_dev(dout_ref, dout_c, ND));

        // 5A: Isolated Hadamard & gemv_moe_s kernels (Pinned descs vs VRAM descs vs Vectorized/Fused)
        const int GI_H = 32;
        std::vector<BenchItem> b5h;
        b5h.push_back({"had_in_moe GU  [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_in_moe<<<dim3(5, nslots, 2), 128, 0, st>>>(dd_pin_d[0], dd_pin_d[1], dids, dx, dxh, ddn, 2*NG, KG, nullptr, 0, nullptr, nullptr);
        }, GI_H});
        b5h.push_back({"had_in_moe GU  [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_in_moe<<<dim3(5, nslots, 2), 128, 0, st>>>(dd_vram[0], dd_vram[1], dids, dx, dxh, ddn, 2*NG, KG, nullptr, 0, nullptr, nullptr);
        }, GI_H});
        b5h.push_back({"had_in_moe_vec GU [VRAM+f4]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                mh::had_in_moe_vec<<<dim3(9, nslots, 2), 64, 0, st>>>(dd_vram[0], dd_vram[1], dids, dx, dxh, ddn, 2*NG, KG);
        }, GI_H});
        b5h.push_back({"had_out_moe GU [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_out_moe<<<dim3(3, nslots, 2), 128, 0, st>>>(dd_pin_d[0], dd_pin_d[1], dids, ddn, dgu, NG, 2*NG, 0, ND, NG);
        }, GI_H});
        b5h.push_back({"had_out_moe GU [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_out_moe<<<dim3(3, nslots, 2), 128, 0, st>>>(dd_vram[0], dd_vram[1], dids, ddn, dgu, NG, 2*NG, 0, ND, NG);
        }, GI_H});
        b5h.push_back({"had_out_moe_vec GU [VRAM+f4]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                mh::had_out_moe_vec<<<dim3(3, nslots, 2), 96, 0, st>>>(dd_vram[0], dd_vram[1], dids, ddn, dgu, NG, 2*NG, 0, ND, NG);
        }, GI_H});
        b5h.push_back({"had_in_moe DN  [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_in_moe<<<dim3(3, nslots, 1), 128, 0, st>>>(dd_pin_d[2], nullptr, dids, dhid, dxh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
        }, GI_H});
        b5h.push_back({"had_in_moe DN  [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_in_moe<<<dim3(3, nslots, 1), 128, 0, st>>>(dd_vram[2], nullptr, dids, dhid, dxh, nullptr, 0, KD, nullptr, 0, nullptr, nullptr);
        }, GI_H});
        b5h.push_back({"situ + had_in DN [VRAM, 2->1]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                mh::situ_had_in_moe<<<dim3(3, nslots, 1), 96, 0, st>>>(dgu, dd_vram[2], dids, dxh, 4.f, 25.f, NG);
        }, GI_H});
        b5h.push_back({"had_out_moe DN [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_out_moe<<<dim3(5, nslots, 1), 128, 0, st>>>(dd_pin_d[2], nullptr, dids, ddn, dxh, ND, ND, 0, ND, 0);
        }, GI_H});
        b5h.push_back({"had_out_moe DN [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                inkwell_exl3_had_out_moe<<<dim3(5, nslots, 1), 128, 0, st>>>(dd_vram[2], nullptr, dids, ddn, dxh, ND, ND, 0, ND, 0);
        }, GI_H});
        b5h.push_back({"had_out_wsum DN [VRAM, 2->1]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_H; ++i)
                mh::had_out_wsum_moe<<<18, 256, 0, st>>>(dd_vram[2], dids, ddn, dw, dout_c, ND);
        }, GI_H});
        b5h.push_back({"gemv_moe_s GU  [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < 8; ++i)
                inkwell_exl3_gemv_moe_s<<<dim3(9, nslots, 2), 256, 0, st>>>(dd_pin_d[0], dd_pin_d[1], dids, dxh, dgu, 2*NG, ddn, ND, NG, KG/16, NG/16, KG, ND, 1, nullptr, 0, 0);
        }, 8});
        b5h.push_back({"gemv_moe_s GU  [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < 8; ++i)
                inkwell_exl3_gemv_moe_s<<<dim3(9, nslots, 2), 256, 0, st>>>(dd_vram[0], dd_vram[1], dids, dxh, dgu, 2*NG, ddn, ND, NG, KG/16, NG/16, KG, ND, 1, nullptr, 0, 0);
        }, 8});
        b5h.push_back({"gemv_moe_s DN  [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < 8; ++i)
                inkwell_exl3_gemv_moe_s<<<dim3(18, nslots, 1), 256, 0, st>>>(dd_pin_d[2], nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND, 1, nullptr, 0, 0);
        }, 8});
        b5h.push_back({"gemv_moe_s DN  [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < 8; ++i)
                inkwell_exl3_gemv_moe_s<<<dim3(18, nslots, 1), 256, 0, st>>>(dd_vram[2], nullptr, dids, dxh, nullptr, 0, ddn, ND, 0, KD/16, ND/16, KD, ND, 1, nullptr, 0, 0);
        }, 8});
        run_interleaved_graphs(b5h, 25);
        print_bench_table("SUITE 5A: Isolated MoE Hadamard & gemv_moe_s (modeled x30 on_dev)", b5h, 30.0);

        // 5B: Full MoE 8-kernel chain (real weights, interleaved!)
        const int GI_M = 8;
        std::vector<BenchItem> b5c;
        b5c.push_back({"H0: shipped 8k [PINNED descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_M; ++i) launch_shipped(dd_pin_d[0], dd_pin_d[1], dd_pin_d[2], dout_ref, st);
        }, GI_M});
        b5c.push_back({"H1: shipped 8k [VRAM descs]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_M; ++i) launch_shipped(dd_vram[0], dd_vram[1], dd_vram[2], dout_ref, st);
        }, GI_M});
        b5c.push_back({"H2: +fuse had_out (6k) [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_M; ++i) launch_h2_safe(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, st);
        }, GI_M});
        b5c.push_back({"H3a: +fuse had_in+out (4k) [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_M; ++i) launch_h3a_both(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, st);
        }, GI_M});
        b5c.push_back({"H3d: GU(in+out)+situ_hi+DN+ho_wsum", [&](cudaStream_t st) {
            for (int i = 0; i < GI_M; ++i) launch_h3d_outwsum(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, st);
        }, GI_M});
        b5c.push_back({"H3e: clean 6k (shipped gemv) [EXACT]", [&](cudaStream_t st) {
            for (int i = 0; i < GI_M; ++i) launch_h3e_clean6k(dd_vram[0], dd_vram[1], dd_vram[2], dout_c, st);
        }, GI_M});
        run_interleaved_graphs(b5c, 25);
        print_bench_table("SUITE 5B: Full MoE chain (modeled x48 visits/tok)", b5c, 48.0);

        for (int ki = 0; ki < 3; ++ki) {
            for (int s = 0; s < nslots; ++s) { CK(cudaFree(dtr[ki][s])); CK(cudaFree(dsuh[ki][s])); CK(cudaFree(dsvh[ki][s])); }
            CK(cudaFree(dd_vram[ki])); CK(cudaFreeHost(dd_pin_h[ki]));
        }
        CK(cudaFree(dids)); CK(cudaFree(dx)); CK(cudaFree(dw)); CK(cudaFree(dxh)); CK(cudaFree(dxh_ref));
        CK(cudaFree(dgu)); CK(cudaFree(dgu_ref)); CK(cudaFree(dhid)); CK(cudaFree(ddn));
        CK(cudaFree(dout_ref)); CK(cudaFree(dout_c));
    }

    return 0;
}
