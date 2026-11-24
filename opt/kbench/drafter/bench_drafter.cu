// Microbenchmark suite for Questions 2, 3, and 4:
//   - Correctness checks vs CPU int64/fp64 reference for every kernel
//   - Q4: Hand-tuned sm_61 kernels: dp4a (int8xint8), dp2a (int16xint8), int4 nibble-unpack dp4a,
//         fused RMSNorm+quant, fused SwiGLU+quant, fused Head+Argmax (single launch, zero host sync)
//         Measured GB/s vs ~204 GB/s DRAM peak!
//   - Q2: Head comparison (all measured on silicon):
//         (2a) Target's REAL resident EXL3 unembed (K=4.0, 200,058 x 2304, /tmp/sgemv_dense_gold/unembed.bin)
//              batched over B=1, 2, 3, 4 rows
//         (2b) Reduced-vocab int8 head (V'=8K, 16K, 32K, 64K) at H=2304 (and narrow H=512, 1024)
//              with dp4a and dp2a, fused with argmax + vocab_ids remap
//         (2c) Int4 nibble-unpack head (V'=8K, 16K, 32K, 64K) + Two-stage shortlist(r=256)+rescore(M=64)
//   - Q3: Full synthetic drafter bodies + heads captured in ONE CUDA Graph (no host sync inside round!)
//         for k = 1, 2, 3 drafts:
//           1. EAGLE-1-Slim  (1L, D=512,  FF=1536, SWA-128, 4-tap fuse) + [V', 512] head
//           2. EAGLE-1-Med   (1L, D=1024, FF=2816, SWA-128, 4-tap fuse) + [V', 1024] head
//           3. EAGLE-2L-Med  (2L, D=1024, FF=2816, SWA-128, 4-tap fuse) + [V', 1024] head
//           4. EAGLE-1-Full  (1L, D=2304, FF=4608, SWA-128, 4-tap fuse) + resident EXL3 OR i4/i8 head
//           5. Medusa-MLP    (0 attn, 2L ResMLP D=1024/FF=2048, batched parallel k drafts) + [V', 1024]
//           6. DSpark-Int8 / DSpark-Int4 (3L D=2304/FF=9216 + Markov GRU, batched k+1 body + k Markov steps)
//
// Build (prints register/smem usage via -Xptxas -v):
//   ~/lfm-gtx1060-bench/cuda-12.6/bin/nvcc -O3 -std=c++17 --use_fast_math -Xptxas -v \
//     -ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++ \
//     -gencode arch=compute_61,code=sm_61 opt/kbench/drafter/bench_drafter.cu -o /tmp/bench_drafter
// Run (ALWAYS under flock):
//   flock /tmp/inkwell-gpu.lock /tmp/bench_drafter

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "drafter_kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint32_t rng_s = 0x12345678u;
static uint32_t xorshift() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5; return rng_s; }
static float randf() { return ((int)(xorshift() % 2001) - 1000) / 1000.f; }

template <typename F>
static float bench_us(F&& fn, int warmup = 5, int reps = 31) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) fn();
    CK(cudaDeviceSynchronize());
    std::vector<float> ts; ts.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a));
        fn();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(ms * 1000.f);
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2]; // median us
}

template <typename F>
static float bench_graph_us(F&& record_fn, cudaStream_t stream, int warmup = 5, int reps = 31) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    record_fn(stream);
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < warmup; ++i) CK(cudaGraphLaunch(exec, stream));
    CK(cudaStreamSynchronize(stream));
    std::vector<float> ts; ts.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CK(cudaEventRecord(a, stream));
        CK(cudaGraphLaunch(exec, stream));
        CK(cudaEventRecord(b, stream));
        CK(cudaEventSynchronize(b));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(ms * 1000.f);
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

// ---- Exactness verification of all kernels ----
static void verify_kernels() {
    printf("=== 1. KERNEL EXACTNESS VERIFICATION (vs CPU ref) ===\n");
    const int N = 256, K = 2304, B = 3;
    std::vector<float> hx(B * K), hres(B * K), hgam(K), hsw(N);
    std::vector<int8_t> hW8(N * K);
    std::vector<uint8_t> hW4(N * (K / 2));
    std::vector<uint32_t> hvids(N);
    for (int i = 0; i < B * K; ++i) { hx[i] = randf() * 2.f; hres[i] = randf() * 0.5f; }
    for (int i = 0; i < K; ++i) hgam[i] = 0.5f + fabsf(randf());
    for (int r = 0; r < N; ++r) {
        hsw[r] = 0.005f + 0.01f * fabsf(randf());
        hvids[r] = 100000u + r * 3u;
        for (int c = 0; c < K; ++c) hW8[r * K + c] = (int8_t)((int)(xorshift() % 255) - 127);
        // Pack int4: each uint32_t holds 8 weights: low nibbles cols 0..3, high nibbles cols 4..7
        uint32_t* w4p = reinterpret_cast<uint32_t*>(hW4.data() + (size_t)r * (K / 2));
        for (int c = 0; c < K; c += 8) {
            uint32_t p = 0;
            for (int j = 0; j < 4; ++j) {
                uint32_t u_lo = (uint32_t)((int)(xorshift() % 15) + 1); // 1..15 (q in -7..+7)
                uint32_t u_hi = (uint32_t)((int)(xorshift() % 15) + 1);
                p |= (u_lo << (8 * j)) | (u_hi << (8 * j + 4));
            }
            w4p[c / 8] = p;
        }
    }

    float *dx, *dres, *dgam, *dres_out, *dsw, *dsx, *dy, *dblk_v;
    int8_t *dW8, *dxq8; int16_t *dxq16; uint8_t *dW4;
    int *dsum_x; uint32_t *dvids, *dblk_i, *dticket, *dout_tok;
    CK(cudaMalloc(&dx, B * K * 4)); CK(cudaMalloc(&dres, B * K * 4));
    CK(cudaMalloc(&dgam, K * 4)); CK(cudaMalloc(&dres_out, B * K * 4));
    CK(cudaMalloc(&dsw, N * 4)); CK(cudaMalloc(&dsx, B * 4));
    CK(cudaMalloc(&dy, B * N * 4)); CK(cudaMalloc(&dW8, N * K));
    CK(cudaMalloc(&dW4, N * (K / 2))); CK(cudaMalloc(&dxq8, B * K));
    CK(cudaMalloc(&dxq16, B * K * 2)); CK(cudaMalloc(&dsum_x, B * 4));
    CK(cudaMalloc(&dvids, N * 4)); CK(cudaMalloc(&dblk_v, B * 256 * 4));
    CK(cudaMalloc(&dblk_i, B * 256 * 4)); CK(cudaMalloc(&dticket, 4));
    CK(cudaMalloc(&dout_tok, B * 4));
    CK(cudaMemset(dticket, 0, 4));
    CK(cudaMemcpy(dx, hx.data(), B * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dres, hres.data(), B * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgam, hgam.data(), K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsw, hsw.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW8, hW8.data(), N * K, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW4, hW4.data(), N * (K / 2), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvids, hvids.data(), N * 4, cudaMemcpyHostToDevice));

    // Test fused_rmsnorm_quant<false> (int8) + gemv_i8_dp4a<3>
    drf::fused_rmsnorm_quant<false><<<B, 256, K * 4>>>(dx, dres, dgam, dres_out, dxq8, dsx, dsum_x, K, 1e-6f);
    drf::gemv_i8_dp4a<3><<<N / 8, 256, B * K>>>(dW8, dsw, dxq8, dsx, dy, N, K);
    CK(cudaDeviceSynchronize());

    std::vector<int8_t> hxq8(B * K); std::vector<float> hsx(B), hy(B * N); std::vector<int> hsum_x(B);
    CK(cudaMemcpy(hxq8.data(), dxq8, B * K, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hsx.data(), dsx, B * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hsum_x.data(), dsum_x, B * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy.data(), dy, B * N * 4, cudaMemcpyDeviceToHost));

    double max_err_dp4a = 0.0, max_ref = 0.0;
    uint32_t ref_amax_8[3] = {0, 0, 0};
    for (int b = 0; b < B; ++b) {
        int s = 0; for (int c = 0; c < K; ++c) s += hxq8[b * K + c];
        if (s != hsum_x[b]) { printf("FAIL sum_x b=%d: got %d exp %d\n", b, hsum_x[b], s); exit(1); }
        double best = -1e30; uint32_t bi = 0;
        for (int r = 0; r < N; ++r) {
            int64_t dot = 0;
            for (int c = 0; c < K; ++c) dot += (int64_t)hxq8[b * K + c] * (int64_t)hW8[r * K + c];
            double ref = (double)dot * (double)hsw[r] * (double)hsx[b];
            max_err_dp4a = std::max(max_err_dp4a, fabs(ref - hy[b * N + r]));
            max_ref = std::max(max_ref, fabs(ref));
            if ((double)dot * hsw[r] > best) { best = (double)dot * hsw[r]; bi = hvids[r]; }
        }
        ref_amax_8[b] = bi;
    }
    // Check fused head_argmax_i8_dp4a<3>
    drf::head_argmax_i8_dp4a<3><<<16, 256, B * K>>>(dW8, dsw, dxq8, dsx, dvids, dblk_v, dblk_i, dticket, dout_tok, N, K);
    CK(cudaDeviceSynchronize());
    uint32_t got_amax[3]; CK(cudaMemcpy(got_amax, dout_tok, 12, cudaMemcpyDeviceToHost));
    bool amax8_ok = (got_amax[0] == ref_amax_8[0] && got_amax[1] == ref_amax_8[1] && got_amax[2] == ref_amax_8[2]);
    printf("  gemv_i8_dp4a<3>:      relerr = %.2e  head_argmax_i8_dp4a: %s (%u,%u,%u)\n",
           max_err_dp4a / max_ref, amax8_ok ? "PASS" : "FAIL", got_amax[0], got_amax[1], got_amax[2]);
    if (!amax8_ok || max_err_dp4a / max_ref > 1e-5) exit(1);

    // Test fused_rmsnorm_quant<true> (int16) + gemv_i8_dp2a<3> + head_argmax_i8_dp2a<3>
    drf::fused_rmsnorm_quant<true><<<B, 256, K * 4>>>(dx, dres, dgam, nullptr, dxq16, dsx, nullptr, K, 1e-6f);
    drf::gemv_i8_dp2a<3><<<N / 8, 256, B * K * 2>>>(dW8, dsw, dxq16, dsx, dy, N, K);
    drf::head_argmax_i8_dp2a<3><<<16, 256, B * K * 2>>>(dW8, dsw, dxq16, dsx, dvids, dblk_v, dblk_i, dticket, dout_tok, N, K);
    CK(cudaDeviceSynchronize());
    std::vector<int16_t> hxq16(B * K);
    CK(cudaMemcpy(hxq16.data(), dxq16, B * K * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hsx.data(), dsx, B * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy.data(), dy, B * N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got_amax, dout_tok, 12, cudaMemcpyDeviceToHost));
    double max_err_dp2a = 0.0; max_ref = 0.0;
    uint32_t ref_amax_16[3] = {0, 0, 0};
    for (int b = 0; b < B; ++b) {
        double best = -1e30; uint32_t bi = 0;
        for (int r = 0; r < N; ++r) {
            int64_t dot = 0;
            for (int c = 0; c < K; ++c) dot += (int64_t)hxq16[b * K + c] * (int64_t)hW8[r * K + c];
            double ref = (double)dot * (double)hsw[r] * (double)hsx[b];
            max_err_dp2a = std::max(max_err_dp2a, fabs(ref - hy[b * N + r]));
            max_ref = std::max(max_ref, fabs(ref));
            if ((double)dot * hsw[r] > best) { best = (double)dot * hsw[r]; bi = hvids[r]; }
        }
        ref_amax_16[b] = bi;
    }
    bool amax16_ok = (got_amax[0] == ref_amax_16[0] && got_amax[1] == ref_amax_16[1] && got_amax[2] == ref_amax_16[2]);
    printf("  gemv_i8_dp2a<3>:      relerr = %.2e  head_argmax_i8_dp2a: %s (%u,%u,%u)\n",
           max_err_dp2a / max_ref, amax16_ok ? "PASS" : "FAIL", got_amax[0], got_amax[1], got_amax[2]);
    if (!amax16_ok || max_err_dp2a / max_ref > 1e-5) exit(1);

    // Test gemv_i4_dp4a<3> + head_argmax_i4_dp4a<3>
    drf::fused_rmsnorm_quant<false><<<B, 256, K * 4>>>(dx, dres, dgam, nullptr, dxq8, dsx, dsum_x, K, 1e-6f);
    drf::gemv_i4_dp4a<3><<<N / 8, 256, B * K>>>(dW4, dsw, dxq8, dsx, dsum_x, dy, N, K);
    drf::head_argmax_i4_dp4a<3><<<16, 256, B * K>>>(dW4, dsw, dxq8, dsum_x, dvids, dblk_v, dblk_i, dticket, dout_tok, N, K);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hsx.data(), dsx, B * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hy.data(), dy, B * N * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got_amax, dout_tok, 12, cudaMemcpyDeviceToHost));
    double max_err_i4 = 0.0; max_ref = 0.0;
    uint32_t ref_amax_4[3] = {0, 0, 0};
    for (int b = 0; b < B; ++b) {
        double best = -1e30; uint32_t bi = 0;
        for (int r = 0; r < N; ++r) {
            int64_t dot = 0;
            const uint32_t* w4p = reinterpret_cast<const uint32_t*>(hW4.data() + (size_t)r * (K / 2));
            for (int c = 0; c < K; c += 8) {
                uint32_t p = w4p[c / 8];
                for (int j = 0; j < 4; ++j) {
                    int qlo = (int)((p >> (8 * j)) & 0xF) - 8;
                    int qhi = (int)((p >> (8 * j + 4)) & 0xF) - 8;
                    dot += (int64_t)hxq8[b * K + c + j] * qlo;
                    dot += (int64_t)hxq8[b * K + c + 4 + j] * qhi;
                }
            }
            double ref = (double)dot * (double)hsw[r] * (double)hsx[b];
            max_err_i4 = std::max(max_err_i4, fabs(ref - hy[b * N + r]));
            max_ref = std::max(max_ref, fabs(ref));
            if ((double)dot * hsw[r] > best) { best = (double)dot * hsw[r]; bi = hvids[r]; }
        }
        ref_amax_4[b] = bi;
    }
    bool amax4_ok = (got_amax[0] == ref_amax_4[0] && got_amax[1] == ref_amax_4[1] && got_amax[2] == ref_amax_4[2]);
    printf("  gemv_i4_dp4a<3>:      relerr = %.2e  head_argmax_i4_dp4a: %s (%u,%u,%u)\n",
           max_err_i4 / max_ref, amax4_ok ? "PASS" : "FAIL", got_amax[0], got_amax[1], got_amax[2]);
    if (!amax4_ok || max_err_i4 / max_ref > 1e-5) exit(1);

    // Verify head_argmax_i4g128_dp4a<3> against CPU int64/fp32 ref
    {
        const int ng = K / 128;
        std::vector<__half> hsg(N * ng);
        std::vector<int> hsxg(B * ng, 0);
        for (int r = 0; r < N; ++r)
            for (int g = 0; g < ng; ++g)
                hsg[r * ng + g] = __float2half(0.005f + 0.01f * fabsf(randf()));
        for (int b = 0; b < B; ++b)
            for (int g = 0; g < ng; ++g)
                for (int dc = 0; dc < 128; ++dc)
                    hsxg[b * ng + g] += hxq8[b * K + g * 128 + dc];
        __half* dsg; int* dsxg;
        CK(cudaMalloc(&dsg, N * ng * 2)); CK(cudaMalloc(&dsxg, B * ng * 4));
        CK(cudaMemcpy(dsg, hsg.data(), N * ng * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsxg, hsxg.data(), B * ng * 4, cudaMemcpyHostToDevice));
        drf::head_argmax_i4g128_dp4a<3><<<16, 256, B * K>>>(dW4, dsg, dxq8, dsxg, dvids, dblk_v, dblk_i, dticket, dout_tok, N, K);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got_amax, dout_tok, 12, cudaMemcpyDeviceToHost));
        uint32_t ref_g[3] = {0, 0, 0};
        for (int b = 0; b < B; ++b) {
            double best = -1e30; uint32_t bi = 0;
            for (int r = 0; r < N; ++r) {
                double tot = 0.0;
                const uint32_t* w4p = reinterpret_cast<const uint32_t*>(hW4.data() + (size_t)r * (K / 2));
                for (int g = 0; g < ng; ++g) {
                    int64_t gdot = 0;
                    for (int dc = 0; dc < 128; dc += 8) {
                        uint32_t p = w4p[(g * 128 + dc) / 8];
                        for (int j = 0; j < 4; ++j) {
                            gdot += (int64_t)hxq8[b * K + g * 128 + dc + j]     * ((int)((p >> (8 * j)) & 0xF) - 8);
                            gdot += (int64_t)hxq8[b * K + g * 128 + dc + 4 + j] * ((int)((p >> (8 * j + 4)) & 0xF) - 8);
                        }
                    }
                    tot += (double)gdot * (double)__half2float(hsg[r * ng + g]);
                }
                if (tot > best) { best = tot; bi = hvids[r]; }
            }
            ref_g[b] = bi;
        }
        bool g_ok = (got_amax[0] == ref_g[0] && got_amax[1] == ref_g[1] && got_amax[2] == ref_g[2]);
        printf("  head_argmax_i4g128<3>:                     status: %s (%u,%u,%u)\n",
               g_ok ? "PASS" : "FAIL", got_amax[0], got_amax[1], got_amax[2]);
        if (!g_ok) exit(1);
        cudaFree(dsg); cudaFree(dsxg);
    }

    cudaFree(dx); cudaFree(dres); cudaFree(dgam); cudaFree(dres_out); cudaFree(dsw);
    cudaFree(dsx); cudaFree(dy); cudaFree(dW8); cudaFree(dW4); cudaFree(dxq8);
    cudaFree(dxq16); cudaFree(dsum_x); cudaFree(dvids); cudaFree(dblk_v);
    cudaFree(dblk_i); cudaFree(dticket); cudaFree(dout_tok);
}

// ---- Q2a: Real resident EXL3 unembed (K=4.0, 200,058 x 2304) batched B=1..4 ----
static void bench_exl3_unembed() {
    printf("\n=== 2A. RESIDENT EXL3 UNEMBED (K=4.0, 200,058 x 2304, 0 MiB extra VRAM) ===\n");
    const int tk = 144, tn = 12568, K = tk * 16, N = tn * 16, V = 200058;
    const size_t nbytes = (size_t)tk * tn * 32 * 4; // 231,653,376 B (220.92 MiB)
    std::vector<uint32_t> htrel(nbytes / 4);
    FILE* f = fopen("/tmp/sgemv_dense_gold/unembed.bin", "rb");
    if (f) {
        size_t got = fread(htrel.data(), 1, nbytes, f);
        fclose(f);
        if (got != nbytes) { printf("short read unembed.bin (%zu/%zu)\n", got, nbytes); }
    } else {
        for (size_t i = 0; i < htrel.size(); ++i) htrel[i] = xorshift();
    }
    uint32_t* dtrel; float *dx, *dout, *dblk_v; uint16_t* dsvh;
    uint32_t *dblk_i, *dticket, *dout_tok;
    CK(cudaMalloc(&dtrel, nbytes));
    CK(cudaMemcpy(dtrel, htrel.data(), nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, (size_t)4 * K * 4));
    CK(cudaMalloc(&dout, (size_t)4 * N * 4));
    CK(cudaMalloc(&dsvh, N * 2));
    CK(cudaMalloc(&dblk_v, 4 * 256 * 4));
    CK(cudaMalloc(&dblk_i, 4 * 256 * 4));
    CK(cudaMalloc(&dticket, 4)); CK(cudaMalloc(&dout_tok, 16));
    CK(cudaMemset(dx, 0x3c, (size_t)4 * K * 4));
    CK(cudaMemset(dsvh, 0x3c, N * 2));
    CK(cudaMemset(dticket, 0, 4));
    const uint32_t CM = 0x83DCD12Du;
    const int nb = (tn + 7) / 8;

    auto run_b = [&](auto Btag) {
        constexpr int B = decltype(Btag)::value;
        float t_gemv = bench_us([&]() {
            drf::head_exl3_k4_batch_c0c<B><<<nb, 256>>>(dtrel, dx, dout, tk, tn, K, N, CM);
        });
        float t_had = bench_us([&]() {
            drf::exl3_had_out_argmax<B><<<160, 256>>>(dout, dsvh, dblk_v, dblk_i, dticket, dout_tok, V, N);
        });
        float tot = t_gemv + t_had;
        double gbps = (double)nbytes / (t_gemv * 1e3); // GB/s on weight read
        double eff_gmac = (double)B * K * N / (t_gemv * 1e3);
        printf("  EXL3 unembed K=4.0 B=%d: gemv=%7.1f us (%5.1f GB/s, %5.1f GMAC/s) + had+argmax=%5.1f us => TOTAL=%7.1f us (%.3f ms)\n",
               B, t_gemv, gbps, eff_gmac, t_had, tot, tot / 1000.f);
    };
    run_b(std::integral_constant<int, 1>{});
    run_b(std::integral_constant<int, 2>{});
    run_b(std::integral_constant<int, 3>{});
    run_b(std::integral_constant<int, 4>{});

    cudaFree(dtrel); cudaFree(dx); cudaFree(dout); cudaFree(dsvh);
    cudaFree(dblk_v); cudaFree(dblk_i); cudaFree(dticket); cudaFree(dout_tok);
}

// ---- Q2b, Q2c, Q4: Microbenchmarks of GEMVs, RMSNorm+Quant, and All Head Options ----
static void bench_kernels_and_heads() {
    printf("\n=== 2B/2C/4. HAND-TUNED SM_61 KERNELS & REDUCED-VOCAB HEADS (vs ~204 GB/s DRAM peak) ===\n");
    const int MAX_V = 65536, MAX_K = 2304, MAX_B = 4;
    int8_t *dW8, *dxq8; int16_t *dxq16; uint8_t *dW4;
    float *dsw, *dsx, *dx, *dres, *dgam, *dy, *dblk_v;
    int *dsum_x; uint32_t *dvids, *dblk_i, *dticket, *dout_tok;
    CK(cudaMalloc(&dW8, (size_t)MAX_V * MAX_K));
    CK(cudaMalloc(&dW4, (size_t)MAX_V * (MAX_K / 2)));
    CK(cudaMalloc(&dsw, MAX_V * 4));
    CK(cudaMalloc(&dx, MAX_B * MAX_K * 4));
    CK(cudaMalloc(&dres, MAX_B * MAX_K * 4));
    CK(cudaMalloc(&dgam, MAX_K * 4));
    CK(cudaMalloc(&dxq8, MAX_B * MAX_K));
    CK(cudaMalloc(&dxq16, MAX_B * MAX_K * 2));
    CK(cudaMalloc(&dsx, MAX_B * 4));
    CK(cudaMalloc(&dsum_x, MAX_B * 4));
    CK(cudaMalloc(&dy, (size_t)MAX_B * MAX_V * 4));
    CK(cudaMalloc(&dvids, MAX_V * 4));
    CK(cudaMalloc(&dblk_v, MAX_B * 256 * 4));
    CK(cudaMalloc(&dblk_i, MAX_B * 256 * 4));
    CK(cudaMalloc(&dticket, 4));
    CK(cudaMalloc(&dout_tok, MAX_B * 4));
    CK(cudaMemset(dW8, 1, (size_t)MAX_V * MAX_K));
    CK(cudaMemset(dW4, 0x88, (size_t)MAX_V * (MAX_K / 2)));
    CK(cudaMemset(dsw, 0x3c, MAX_V * 4));
    CK(cudaMemset(dx, 0x3c, MAX_B * MAX_K * 4));
    CK(cudaMemset(dres, 0x3c, MAX_B * MAX_K * 4));
    CK(cudaMemset(dgam, 0x3c, MAX_K * 4));
    CK(cudaMemset(dxq8, 1, MAX_B * MAX_K));
    CK(cudaMemset(dxq16, 1, MAX_B * MAX_K * 2));
    CK(cudaMemset(dsx, 0x3c, MAX_B * 4));
    CK(cudaMemset(dsum_x, 0, MAX_B * 4));
    CK(cudaMemset(dticket, 0, 4));

    // RMSNorm+Quant benchmarks
    printf("-- Fused RMSNorm+Quant & SwiGLU+Quant --\n");
    for (int K : {512, 1024, 2304}) {
        for (int B : {1, 2, 3}) {
            float t8 = bench_us([&]() {
                drf::fused_rmsnorm_quant<false><<<B, 256, K * 4>>>(dx, dres, dgam, dres, dxq8, dsx, dsum_x, K, 1e-6f);
            });
            float t16 = bench_us([&]() {
                drf::fused_rmsnorm_quant<true><<<B, 256, K * 4>>>(dx, dres, dgam, dres, dxq16, dsx, dsum_x, K, 1e-6f);
            });
            printf("  fused_rmsnorm_quant K=%4d B=%d: i8=%5.1f us, i16=%5.1f us\n", K, B, t8, t16);
        }
    }

    // Body GEMV benchmarks (N x K representative shapes)
    printf("\n-- GEMV Kernels (dp4a i8x8, dp2a i16x8, int4 nibble-dp4a) --\n");
    struct GShape { const char* name; int N, K; };
    GShape shapes[] = {
        {"FFN_512 (1536x512)",    1536,  512},
        {"FFN_1024 (2816x1024)",  2816, 1024},
        {"Q_2304 (3072x2304)",    3072, 2304},
        {"FFN_2304 (9216x2304)",  9216, 2304},
        {"Head_16K (16384x2304)",16384, 2304},
        {"Head_32K (32768x2304)",32768, 2304},
    };
    printf("%-24s | %14s | %14s | %14s | %14s | %14s\n",
           "Shape (N x K)", "i8 dp4a B=1", "i8 dp4a B=2", "i8 dp4a B=3", "i8 dp2a B=3", "i4 dp4a B=3");
    for (auto& s : shapes) {
        size_t bytes8 = (size_t)s.N * s.K + (size_t)s.N * 4;
        size_t bytes4 = (size_t)s.N * (s.K / 2) + (size_t)s.N * 4;
        float t1 = bench_us([&]() { drf::gemv_i8_dp4a<1><<<s.N/8, 256, 1*s.K>>>(dW8, dsw, dxq8, dsx, dy, s.N, s.K); });
        float t2 = bench_us([&]() { drf::gemv_i8_dp4a<2><<<s.N/8, 256, 2*s.K>>>(dW8, dsw, dxq8, dsx, dy, s.N, s.K); });
        float t3 = bench_us([&]() { drf::gemv_i8_dp4a<3><<<s.N/8, 256, 3*s.K>>>(dW8, dsw, dxq8, dsx, dy, s.N, s.K); });
        float t3_2a = bench_us([&]() { drf::gemv_i8_dp2a<3><<<s.N/8, 256, 3*s.K*2>>>(dW8, dsw, dxq16, dsx, dy, s.N, s.K); });
        float t3_i4 = bench_us([&]() { drf::gemv_i4_dp4a<3><<<s.N/8, 256, 3*s.K>>>(dW4, dsw, dxq8, dsx, dsum_x, dy, s.N, s.K); });
        printf("%-24s | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG)\n",
               s.name,
               t1, bytes8/(t1*1e3), t2, bytes8/(t2*1e3), t3, bytes8/(t3*1e3),
               t3_2a, bytes8/(t3_2a*1e3), t3_i4, bytes4/(t3_i4*1e3));
    }

    // Fused Head + Argmax benchmarks (single launch, zero host sync, vocab_ids remap)
    printf("\n-- Fused Head + Argmax + Vocab-ID Remap (Single Launch, Device Argmax) --\n");
    printf("%-18s %4s %6s | %14s | %14s | %14s | %14s\n",
           "Head Option", "H", "VRAM", "B=1 (us/GBs)", "B=2 (us/GBs)", "B=3 (us/GBs)", "B=3 dp2a");
    for (int H : {2304, 1024, 512}) {
        for (int V : {8192, 16384, 32768, 65536}) {
            if (H < 2304 && V == 65536) continue;
            size_t b8 = (size_t)V * H + (size_t)V * 8; // weights + scale_w + vocab_ids
            size_t b4 = (size_t)V * (H / 2) + (size_t)V * 8;
            int G = 160;
            float t8_1 = bench_us([&]() { drf::head_argmax_i8_dp4a<1><<<G, 256, 1*H>>>(dW8, dsw, dxq8, dsx, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            float t8_2 = bench_us([&]() { drf::head_argmax_i8_dp4a<2><<<G, 256, 2*H>>>(dW8, dsw, dxq8, dsx, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            float t8_3 = bench_us([&]() { drf::head_argmax_i8_dp4a<3><<<G, 256, 3*H>>>(dW8, dsw, dxq8, dsx, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            float t8_3_2a = bench_us([&]() { drf::head_argmax_i8_dp2a<3><<<G, 256, 3*H*2>>>(dW8, dsw, dxq16, dsx, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            printf("i8 V'=%-5d        %4d %5.1fM | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG)\n",
                   V, H, b8 / 1048576.0, t8_1, b8/(t8_1*1e3), t8_2, b8/(t8_2*1e3), t8_3, b8/(t8_3*1e3), t8_3_2a, b8/(t8_3_2a*1e3));

            float t4_1 = bench_us([&]() { drf::head_argmax_i4_dp4a<1><<<G, 256, 1*H>>>(dW4, dsw, dxq8, dsum_x, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            float t4_2 = bench_us([&]() { drf::head_argmax_i4_dp4a<2><<<G, 256, 2*H>>>(dW4, dsw, dxq8, dsum_x, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            float t4_3 = bench_us([&]() { drf::head_argmax_i4_dp4a<3><<<G, 256, 3*H>>>(dW4, dsw, dxq8, dsum_x, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
            printf("i4 V'=%-5d        %4d %5.1fM | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) |       n/a\n",
                   V, H, b4 / 1048576.0, t4_1, b4/(t4_1*1e3), t4_2, b4/(t4_2*1e3), t4_3, b4/(t4_3*1e3));
            if (V == 32768 || (H == 1024 && V == 16384)) {
                size_t b4g = (size_t)V * (H / 2) + (size_t)V * (H / 128) * 2 + (size_t)V * 4;
                // dsw is 65536*4 = 256KB; allocate enough fp16 scales for V*18*2 = 1.2MB
                __half* dsg; int* dsxg;
                CK(cudaMalloc(&dsg, (size_t)V * (H / 128) * 2));
                CK(cudaMalloc(&dsxg, MAX_B * 32 * 4));
                CK(cudaMemset(dsg, 0x3c, (size_t)V * (H / 128) * 2));
                CK(cudaMemset(dsxg, 0, MAX_B * 32 * 4));
                float tg1 = bench_us([&]() { drf::head_argmax_i4g128_dp4a<1><<<G, 256, 1*H>>>(dW4, dsg, dxq8, dsxg, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
                float tg2 = bench_us([&]() { drf::head_argmax_i4g128_dp4a<2><<<G, 256, 2*H>>>(dW4, dsg, dxq8, dsxg, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
                float tg3 = bench_us([&]() { drf::head_argmax_i4g128_dp4a<3><<<G, 256, 3*H>>>(dW4, dsg, dxq8, dsxg, dvids, dblk_v, dblk_i, dticket, dout_tok, V, H); });
                printf("i4-g128 V'=%-5d   %4d %5.1fM | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) | %5.1fu (%5.1fG) |       n/a\n",
                       V, H, b4g / 1048576.0, tg1, b4g/(tg1*1e3), tg2, b4g/(tg2*1e3), tg3, b4g/(tg3*1e3));
                cudaFree(dsg); cudaFree(dsxg);
            }
        }
    }

    // Two-stage Shortlist (r=256 int8 -> 64 cands) + Rescore (64 rows of full H=2304)
    printf("\n-- Two-Stage Shortlist (r=256 int8 -> top-64) + Exact Rescore (H=2304) --\n");
    for (int V : {16384, 32768, 65536}) {
        size_t b_s1 = (size_t)256 * 2304 + (size_t)V * 256 + (size_t)V * 8; // proj + shortlist table
        auto run_2s = [&](auto Btag) {
            constexpr int B = decltype(Btag)::value;
            return bench_us([&]() {
                // 1) project 2304 -> 256
                drf::gemv_i8_dp4a<B><<<256/8, 256, B*2304>>>(dW8, dsw, dxq8, dsx, dy, 256, 2304);
                // 2) quantize 256
                drf::quantize_act<false><<<B, 256>>>(dy, dxq8, dsx, dsum_x, 256);
                // 3) shortlist over V with G=64 blocks -> 64 candidates per row
                drf::head_shortlist_stage1<B, 64><<<64, 256>>>(dW8, dsw, dxq8, dblk_v, dblk_i, V, 256);
                // 4) rescore 64 candidates from full [V, 2304] (or EXL3 unembed)
                drf::head_shortlist_stage2_rescore<B><<<B, 256, 2304>>>(dW8, dsw, dxq8, dblk_i, dvids, dout_tok, 64, 2304);
            });
        };
        float ts1 = run_2s(std::integral_constant<int, 1>{});
        float ts2 = run_2s(std::integral_constant<int, 2>{});
        float ts3 = run_2s(std::integral_constant<int, 3>{});
        printf("  2-stage V'=%5d (r=256, M=64): s1_VRAM=%4.1fM (+full rescore) | B=1:%5.1fu  B=2:%5.1fu  B=3:%5.1fu\n",
               V, b_s1 / 1048576.0, ts1, ts2, ts3);
    }

    cudaFree(dW8); cudaFree(dW4); cudaFree(dsw); cudaFree(dx); cudaFree(dres); cudaFree(dgam);
    cudaFree(dxq8); cudaFree(dxq16); cudaFree(dsx); cudaFree(dsum_x); cudaFree(dy);
    cudaFree(dvids); cudaFree(dblk_v); cudaFree(dblk_i); cudaFree(dticket); cudaFree(dout_tok);
}

// ---- Q3: Full Synthetic Drafter Body + Head Captured in ONE CUDA Graph (k=1, 2, 3) ----
struct DrafterBufs {
    int8_t *W_big, *xq8, *emb8;
    uint8_t *W4_big;
    float *sw, *sx, *x, *res, *gam, *q, *kv, *kc, *vc, *att, *g, *u, *tap, *blk_v;
    int *sum_x;
    uint32_t *vids, *blk_i, *ticket, *out_tok;
    void alloc() {
        // Pool large enough for even DSpark-Int8 (250 MB) + heads!
        CK(cudaMalloc(&W_big,  300ULL * 1024 * 1024));
        CK(cudaMalloc(&W4_big, 150ULL * 1024 * 1024));
        CK(cudaMalloc(&emb8,    32ULL * 1024 * 1024));
        CK(cudaMalloc(&xq8,    4 * 9216));
        CK(cudaMalloc(&sw,     65536 * 4));
        CK(cudaMalloc(&sx,     16 * 4));
        CK(cudaMalloc(&sum_x,  16 * 4));
        CK(cudaMalloc(&x,      4 * 9216 * 4));
        CK(cudaMalloc(&res,    4 * 9216 * 4));
        CK(cudaMalloc(&gam,    9216 * 4));
        CK(cudaMalloc(&q,      4 * 3072 * 4));
        CK(cudaMalloc(&kv,     4 * 2048 * 4));
        CK(cudaMalloc(&kc,     256 * 1024 * 4));
        CK(cudaMalloc(&vc,     256 * 1024 * 4));
        CK(cudaMalloc(&att,    4 * 3072 * 4));
        CK(cudaMalloc(&g,      4 * 9216 * 4));
        CK(cudaMalloc(&u,      4 * 9216 * 4));
        CK(cudaMalloc(&tap,    4 * 2304 * 4));
        CK(cudaMalloc(&vids,   65536 * 4));
        CK(cudaMalloc(&blk_v,  4 * 256 * 4));
        CK(cudaMalloc(&blk_i,  4 * 256 * 4));
        CK(cudaMalloc(&ticket, 4));
        CK(cudaMalloc(&out_tok,16));
        CK(cudaMemset(W_big, 1, 300ULL * 1024 * 1024));
        CK(cudaMemset(W4_big, 0x88, 150ULL * 1024 * 1024));
        CK(cudaMemset(emb8, 1, 32ULL * 1024 * 1024));
        CK(cudaMemset(sw, 0x3c, 65536 * 4));
        CK(cudaMemset(x, 0x3c, 4 * 9216 * 4));
        CK(cudaMemset(res, 0x3c, 4 * 9216 * 4));
        CK(cudaMemset(gam, 0x3c, 9216 * 4));
        CK(cudaMemset(kc, 0x3c, 256 * 1024 * 4));
        CK(cudaMemset(vc, 0x3c, 256 * 1024 * 4));
        CK(cudaMemset(tap, 0x3c, 4 * 2304 * 4));
        CK(cudaMemset(ticket, 0, 4));
        CK(cudaMemset(out_tok, 0, 16));
    }
    void free_all() {
        cudaFree(W_big); cudaFree(W4_big); cudaFree(emb8); cudaFree(xq8);
        cudaFree(sw); cudaFree(sx); cudaFree(sum_x); cudaFree(x); cudaFree(res);
        cudaFree(gam); cudaFree(q); cudaFree(kv); cudaFree(kc); cudaFree(vc);
        cudaFree(att); cudaFree(g); cudaFree(u); cudaFree(tap); cudaFree(vids);
        cudaFree(blk_v); cudaFree(blk_i); cudaFree(ticket); cudaFree(out_tok);
    }
};

// Emit one autoregressive EAGLE transformer layer step (B=1, distinct weight offsets so L2 isn't artificially warm!)
static void emit_eagle_layer_i8(DrafterBufs& b, cudaStream_t s, size_t& woff,
                                int D, int FF, int nh, int nkv, int hd, int win) {
    int qdim = nh * hd, kvdim = nkv * hd;
    // 1) RMSNorm + quantize -> xq8[D]
    drf::fused_rmsnorm_quant<false><<<1, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
    // 2) QKV fused (qdim + 2*kvdim) x D
    int qkv = qdim + 2 * kvdim;
    drf::gemv_i8_dp4a<1><<<qkv/8, 256, D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.q, qkv, D);
    woff += (size_t)qkv * D;
    // 3) SWA attention over `win` tokens
    dim3 gr(nh, 1);
    drf::drafter_swa_attn<<<gr, 128, 0, s>>>(b.q, b.kc, b.vc, b.att, win, nh, nkv, hd);
    // 4) Quantize att -> O proj
    drf::quantize_act<false><<<1, 256, 0, s>>>(b.att, b.xq8, b.sx, b.sum_x, qdim);
    drf::gemv_i8_dp4a<1><<<D/8, 256, qdim, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.x, D, qdim);
    woff += (size_t)D * qdim;
    // 5) Post-attn RMSNorm + quantize
    drf::fused_rmsnorm_quant<false><<<1, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
    // 6) Gate + Up (fused 2*FF x D) -> fused SwiGLU+quant -> Down (D x FF)
    drf::gemv_i8_dp4a<1><<<(2*FF)/8, 256, D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.g, 2 * FF, D);
    woff += (size_t)(2 * FF) * D;
    drf::fused_swiglu_quant_i8<<<1, 256, FF * 4, s>>>(b.g, b.g + FF, b.xq8, b.sx, b.sum_x, FF);
    drf::gemv_i8_dp4a<1><<<D/8, 256, FF, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.x, D, FF);
    woff += (size_t)D * FF;
}

// Same for int4 weights
static void emit_eagle_layer_i4(DrafterBufs& b, cudaStream_t s, size_t& woff4,
                                int D, int FF, int nh, int nkv, int hd, int win) {
    int qdim = nh * hd, kvdim = nkv * hd;
    drf::fused_rmsnorm_quant<false><<<1, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
    int qkv = qdim + 2 * kvdim;
    drf::gemv_i4_dp4a<1><<<qkv/8, 256, D, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.q, qkv, D);
    woff4 += (size_t)qkv * (D / 2);
    dim3 gr(nh, 1);
    drf::drafter_swa_attn<<<gr, 128, 0, s>>>(b.q, b.kc, b.vc, b.att, win, nh, nkv, hd);
    drf::quantize_act<false><<<1, 256, 0, s>>>(b.att, b.xq8, b.sx, b.sum_x, qdim);
    drf::gemv_i4_dp4a<1><<<D/8, 256, qdim, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.x, D, qdim);
    woff4 += (size_t)D * (qdim / 2);
    drf::fused_rmsnorm_quant<false><<<1, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
    drf::gemv_i4_dp4a<1><<<(2*FF)/8, 256, D, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.g, 2 * FF, D);
    woff4 += (size_t)(2 * FF) * (D / 2);
    drf::fused_swiglu_quant_i8<<<1, 256, FF * 4, s>>>(b.g, b.g + FF, b.xq8, b.sx, b.sum_x, FF);
    drf::gemv_i4_dp4a<1><<<D/8, 256, FF, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.x, D, FF);
    woff4 += (size_t)D * (FF / 2);
}

static void bench_full_bodies() {
    printf("\n=== 3. FULL DRAFTER ROUNDS IN ONE CAPTURED CUDA GRAPH (zero host sync) ===\n");
    DrafterBufs b; b.alloc();
    cudaStream_t stream; CK(cudaStreamCreate(&stream));

    auto run_eagle = [&](const char* name, int L, int D, int FF, int nh, int nkv, int hd,
                         int V_head, bool use_i4_body, bool use_i4_head, int win) {
        // Compute exact VRAM MiB (weights + scales + vocab_ids + emb[V_head, D] + KV[L, win, 2, nkv, hd] fp16)
        size_t qdim = nh * hd, kvdim = nkv * hd;
        size_t layer_params = (size_t)(qdim + 2 * kvdim) * D + (size_t)D * qdim + 3ULL * D * FF;
        size_t tap_params = (D == 2304) ? 0 : (size_t)D * 2304; // 4-tap scalar mix + 2304->D proj
        size_t body_bytes = (size_t)((L * layer_params + tap_params) * (use_i4_body ? 0.5 : 1.0));
        size_t head_bytes = (size_t)((size_t)V_head * D * (use_i4_head ? 0.5 : 1.0)) + (size_t)V_head * 8;
        // Token embed for autoreg steps (only V_head rows needed in int8, or tied with head!)
        size_t emb_bytes = (size_t)V_head * D * 0.5; // int4 or tied with head (report both untied & tied)
        size_t kv_bytes = (size_t)L * win * 2 * kvdim * 2; // fp16 KV
        double vram_tied = (body_bytes + head_bytes + kv_bytes) / 1048576.0;
        double vram_untied = (body_bytes + head_bytes + emb_bytes + kv_bytes) / 1048576.0;

        float t_k[4] = {0, 0, 0, 0};
        for (int k : {1, 2, 3}) {
            t_k[k] = bench_graph_us([&](cudaStream_t s) {
                // Step 0: project target's 2304-dim mixed tap -> D once per round (if D < 2304)
                size_t woff_tap = 0;
                if (D < 2304) {
                    drf::fused_rmsnorm_quant<false><<<1, 256, 2304 * 4, s>>>(b.tap, nullptr, b.gam, nullptr, b.xq8, b.sx, b.sum_x, 2304, 1e-6f);
                    drf::gemv_i8_dp4a<1><<<D/8, 256, 2304, s>>>(b.W_big, b.sw, b.xq8, b.sx, b.res, D, 2304);
                    woff_tap = (size_t)D * 2304;
                }
                // Autoregressive k draft steps inside ONE graph (each reads out_tok[0] on device!)
                for (int step = 0; step < k; ++step) {
                    drf::drafter_embed_add<<<1, 256, 0, s>>>(b.emb8, b.sw, b.out_tok, b.res, b.x, D, V_head);
                    size_t woff = woff_tap;
                    size_t woff4 = 0;
                    for (int l = 0; l < L; ++l) {
                        if (use_i4_body) emit_eagle_layer_i4(b, s, woff4, D, FF, nh, nkv, hd, win);
                        else             emit_eagle_layer_i8(b, s, woff,  D, FF, nh, nkv, hd, win);
                    }
                    drf::fused_rmsnorm_quant<false><<<1, 256, D * 4, s>>>(b.x, b.res, b.gam, nullptr, b.xq8, b.sx, b.sum_x, D, 1e-6f);
                    if (use_i4_head) {
                        drf::head_argmax_i4_dp4a<1><<<160, 256, D, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sum_x, b.vids, b.blk_v, b.blk_i, b.ticket, b.out_tok, V_head, D);
                    } else {
                        drf::head_argmax_i8_dp4a<1><<<160, 256, D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.vids, b.blk_v, b.blk_i, b.ticket, b.out_tok, V_head, D);
                    }
                }
            }, stream);
        }
        printf("  %-34s | VRAM=%5.1fM (untied %5.1fM) | k=1:%6.1fu (%4.2fms) | k=2:%6.1fu (%4.2fms) | k=3:%6.1fu (%4.2fms)\n",
               name, vram_tied, vram_untied,
               t_k[1], t_k[1]/1000.f, t_k[2], t_k[2]/1000.f, t_k[3], t_k[3]/1000.f);
    };

    run_eagle("EAGLE-1-Slim  (1L D512  i8 + i8 16K)", 1,  512, 1536,  8, 2,  64, 16384, false, false, 128);
    run_eagle("EAGLE-1-Slim  (1L D512  i8 + i8 32K)", 1,  512, 1536,  8, 2,  64, 32768, false, false, 128);
    run_eagle("EAGLE-1-Slim  (1L D512  i8 + i4 32K)", 1,  512, 1536,  8, 2,  64, 32768, false, true,  128);
    run_eagle("EAGLE-1-Med   (1L D1024 i8 + i8 16K)", 1, 1024, 2816, 16, 4,  64, 16384, false, false, 128);
    run_eagle("EAGLE-1-Med   (1L D1024 i8 + i8 32K)", 1, 1024, 2816, 16, 4,  64, 32768, false, false, 128);
    run_eagle("EAGLE-1-Med   (1L D1024 i8 + i4 32K)", 1, 1024, 2816, 16, 4,  64, 32768, false, true,  128);
    run_eagle("EAGLE-2L-Med  (2L D1024 i8 + i4 32K)", 2, 1024, 2816, 16, 4,  64, 32768, false, true,  128);
    run_eagle("EAGLE-1-Full  (1L D2304 i8 + i4 32K)", 1, 2304, 4608, 18, 6, 128, 32768, false, true,  128);
    run_eagle("EAGLE-1-Full  (1L D2304 i4 + i4 32K)", 1, 2304, 4608, 18, 6, 128, 32768, true,  true,  128);

    // Design 5: Parallel Medusa-MLP (0 attention, 2L ResMLP D=1024 FF=2048, batched B=k in ONE pass!)
    auto run_medusa = [&](const char* name, int D, int FF, int V_head, bool use_i4_head) {
        // k parallel draft heads share weights or use small per-head adapters; batch B=k through shared ResMLP + head!
        size_t body_bytes = (size_t)D * 2304 + 2ULL * (3ULL * D * FF); // tap proj + 2 SwiGLU blocks (int8)
        size_t head_bytes = (size_t)((size_t)V_head * D * (use_i4_head ? 0.5 : 1.0)) + (size_t)V_head * 8;
        double vram = (body_bytes + head_bytes) / 1048576.0;
        auto run_k = [&](auto Ktag) {
            constexpr int B = decltype(Ktag)::value;
            return bench_graph_us([&](cudaStream_t s) {
                size_t woff = 0;
                drf::fused_rmsnorm_quant<false><<<B, 256, 2304 * 4, s>>>(b.tap, nullptr, b.gam, nullptr, b.xq8, b.sx, b.sum_x, 2304, 1e-6f);
                drf::gemv_i8_dp4a<B><<<D/8, 256, B * 2304, s>>>(b.W_big, b.sw, b.xq8, b.sx, b.res, D, 2304);
                woff += (size_t)D * 2304;
                for (int l = 0; l < 2; ++l) {
                    drf::fused_rmsnorm_quant<false><<<B, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
                    drf::gemv_i8_dp4a<B><<<(2*FF)/8, 256, B * D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.g, 2 * FF, D);
                    woff += (size_t)(2 * FF) * D;
                    drf::fused_swiglu_quant_i8<<<B, 256, FF * 4, s>>>(b.g, b.g + FF, b.xq8, b.sx, b.sum_x, FF);
                    drf::gemv_i8_dp4a<B><<<D/8, 256, B * FF, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.x, D, FF);
                    woff += (size_t)D * FF;
                }
                drf::fused_rmsnorm_quant<false><<<B, 256, D * 4, s>>>(b.x, b.res, b.gam, nullptr, b.xq8, b.sx, b.sum_x, D, 1e-6f);
                if (use_i4_head)
                    drf::head_argmax_i4_dp4a<B><<<160, 256, B * D, s>>>(b.W4_big, b.sw, b.xq8, b.sum_x, b.vids, b.blk_v, b.blk_i, b.ticket, b.out_tok, V_head, D);
                else
                    drf::head_argmax_i8_dp4a<B><<<160, 256, B * D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.vids, b.blk_v, b.blk_i, b.ticket, b.out_tok, V_head, D);
            }, stream);
        };
        float t1 = run_k(std::integral_constant<int, 1>{});
        float t2 = run_k(std::integral_constant<int, 2>{});
        float t3 = run_k(std::integral_constant<int, 3>{});
        printf("  %-34s | VRAM=%5.1fM                | k=1:%6.1fu (%4.2fms) | k=2:%6.1fu (%4.2fms) | k=3:%6.1fu (%4.2fms)\n",
               name, vram, t1, t1/1000.f, t2, t2/1000.f, t3, t3/1000.f);
    };
    run_medusa("Medusa-MLP    (2L D1024 i8 + i8 32K)", 1024, 2048, 32768, false);
    run_medusa("Medusa-MLP    (2L D1024 i8 + i4 32K)", 1024, 2048, 32768, true);

    // Design 6: Existing DSpark checkpoint re-quantized to int8 or int4 (ZERO retraining!)
    // Batched B=k+1 over 3 layers (D=2304, FF=9216, 24q/8kv hd=128) + ctx_append + k Markov GRU steps + head
    auto run_dspark = [&](const char* name, bool use_i4_body, int V_head, bool use_i4_head) {
        // 3 layers: q(3072x2304)+k(1024x2304)+v(1024x2304)+o(2304x3072)+kctx(1024x2304)+vctx(1024x2304)+3*9216*2304
        // = 87.36M params/layer * 3 = 262.1M params + rnn(768x5120=3.93M) + hout(2304x256=0.59M) = 266.6M params
        // Plus markov_w1[V_head, 256] in int8 (8 MiB at 32K) + KV cache (3*256*2*1024*2 = 3 MiB)
        size_t body_bytes = (size_t)(266.6e6 * (use_i4_body ? 0.5 : 1.0)) + (size_t)V_head * 256 + 3ULL * 1048576;
        size_t head_bytes = (size_t)((size_t)V_head * 2304 * (use_i4_head ? 0.5 : 1.0)) + (size_t)V_head * 8;
        double vram = (body_bytes + head_bytes) / 1048576.0;
        auto run_k = [&](auto Ktag) {
            constexpr int K_DRF = decltype(Ktag)::value;
            constexpr int B = K_DRF + 1; // [anchor, mask_1..mask_k]
            return bench_graph_us([&](cudaStream_t s) {
                size_t woff = 0, woff4 = 0;
                const int D = 2304, FF = 9216, Q = 3072, KV = 1024;
                // 3 layers (ctx_append kctx/vctx + block QKV/attn/O/FFN for B=k+1 rows!)
                for (int l = 0; l < 3; ++l) {
                    // ctx_append (kctx+vctx for B rows)
                    drf::fused_rmsnorm_quant<false><<<B, 256, D * 4, s>>>(b.tap, nullptr, b.gam, nullptr, b.xq8, b.sx, b.sum_x, D, 1e-6f);
                    if (use_i4_body) {
                        drf::gemv_i4_dp4a<B><<<(2*KV)/8, 256, B * D, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.kv, 2 * KV, D);
                        woff4 += (size_t)(2 * KV) * (D / 2);
                    } else {
                        drf::gemv_i8_dp4a<B><<<(2*KV)/8, 256, B * D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.kv, 2 * KV, D);
                        woff += (size_t)(2 * KV) * D;
                    }
                    // Block QKV
                    drf::fused_rmsnorm_quant<false><<<B, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
                    if (use_i4_body) {
                        drf::gemv_i4_dp4a<B><<<(Q+2*KV)/8, 256, B * D, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.q, Q + 2 * KV, D);
                        woff4 += (size_t)(Q + 2 * KV) * (D / 2);
                    } else {
                        drf::gemv_i8_dp4a<B><<<(Q+2*KV)/8, 256, B * D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.q, Q + 2 * KV, D);
                        woff += (size_t)(Q + 2 * KV) * D;
                    }
                    // Attn
                    dim3 gr(24, B);
                    drf::drafter_swa_attn<<<gr, 128, 0, s>>>(b.q, b.kc, b.vc, b.att, 128, 24, 8, 128);
                    // O proj
                    drf::quantize_act<false><<<B, 256, 0, s>>>(b.att, b.xq8, b.sx, b.sum_x, Q);
                    if (use_i4_body) {
                        drf::gemv_i4_dp4a<B><<<D/8, 256, B * Q, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.x, D, Q);
                        woff4 += (size_t)D * (Q / 2);
                    } else {
                        drf::gemv_i8_dp4a<B><<<D/8, 256, B * Q, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.x, D, Q);
                        woff += (size_t)D * Q;
                    }
                    // Post-LN + Gate/Up + SwiGLU + Down
                    drf::fused_rmsnorm_quant<false><<<B, 256, D * 4, s>>>(b.x, b.res, b.gam, b.res, b.xq8, b.sx, b.sum_x, D, 1e-6f);
                    if (use_i4_body) {
                        drf::gemv_i4_dp4a<B><<<(2*FF)/8, 256, B * D, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.g, 2 * FF, D);
                        woff4 += (size_t)(2 * FF) * (D / 2);
                    } else {
                        drf::gemv_i8_dp4a<B><<<(2*FF)/8, 256, B * D, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.g, 2 * FF, D);
                        woff += (size_t)(2 * FF) * D;
                    }
                    drf::fused_swiglu_quant_i8<<<B, 256, FF * 4, s>>>(b.g, b.g + FF, b.xq8, b.sx, b.sum_x, FF);
                    // Note: down_proj K=9216 -> split or smem B*9216 (at B=4: 36864 B <= 48KB!)
                    if (use_i4_body) {
                        drf::gemv_i4_dp4a<B><<<D/8, 256, B * FF, s>>>(b.W4_big + woff4, b.sw, b.xq8, b.sx, b.sum_x, b.x, D, FF);
                        woff4 += (size_t)D * (FF / 2);
                    } else {
                        drf::gemv_i8_dp4a<B><<<D/8, 256, B * FF, s>>>(b.W_big + woff, b.sw, b.xq8, b.sx, b.x, D, FF);
                        woff += (size_t)D * FF;
                    }
                }
                // k sequential Markov GRU steps + Head+Argmax
                for (int step = 0; step < K_DRF; ++step) {
                    drf::quantize_act<false><<<1, 256, 0, s>>>(b.x, b.xq8, b.sx, b.sum_x, 5120);
                    drf::gemv_i8_dp4a<1><<<768/8, 256, 5120, s>>>(b.W_big, b.sw, b.xq8, b.sx, b.q, 768, 5120);
                    drf::quantize_act<false><<<1, 256, 0, s>>>(b.q, b.xq8, b.sx, b.sum_x, 256);
                    drf::gemv_i8_dp4a<1><<<D/8, 256, 256, s>>>(b.W_big + 4*1048576, b.sw, b.xq8, b.sx, b.x, D, 256);
                    drf::quantize_act<false><<<1, 256, 0, s>>>(b.x, b.xq8, b.sx, b.sum_x, D);
                    if (use_i4_head)
                        drf::head_argmax_i4_dp4a<1><<<160, 256, D, s>>>(b.W4_big, b.sw, b.xq8, b.sum_x, b.vids, b.blk_v, b.blk_i, b.ticket, b.out_tok, V_head, D);
                    else
                        drf::head_argmax_i8_dp4a<1><<<160, 256, D, s>>>(b.W_big, b.sw, b.xq8, b.sx, b.vids, b.blk_v, b.blk_i, b.ticket, b.out_tok, V_head, D);
                }
            }, stream);
        };
        float t1 = run_k(std::integral_constant<int, 1>{});
        float t2 = run_k(std::integral_constant<int, 2>{});
        float t3 = run_k(std::integral_constant<int, 3>{});
        printf("  %-34s | VRAM=%5.1fM                | k=1:%6.1fu (%4.2fms) | k=2:%6.1fu (%4.2fms) | k=3:%6.1fu (%4.2fms)\n",
               name, vram, t1, t1/1000.f, t2, t2/1000.f, t3, t3/1000.f);
    };
    run_dspark("DSpark-Int8   (3L D2304 i8 + i8 32K)", false, 32768, false);
    run_dspark("DSpark-Int4   (3L D2304 i4 + i4 32K)", true,  32768, true);
    run_dspark("DSpark-Int4   (3L D2304 i4 + i4 16K)", true,  16384, true);

    CK(cudaStreamDestroy(stream));
    b.free_all();
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs, %.0f MB VRAM)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1048576.0);
    verify_kernels();
    bench_exl3_unembed();
    bench_kernels_and_heads();
    bench_full_bodies();
    return 0;
}
