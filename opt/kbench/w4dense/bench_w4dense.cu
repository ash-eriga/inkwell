// Full verification & interleaved GPU benchmark suite for group-wise W4/W5/W6 dense GEMVs
// on Pascal sm_61 (GTX 1060 6GB), head-to-head against shipped inkwell_exl3_gemv_s
// AND shipped end-to-end EXL3 dense chain (inkwell_exl3_had_in + inkwell_exl3_gemv_sf).
//
// Build: see opt/kbench/w4dense/build_and_run.sh
// Run:   flock /tmp/inkwell-gpu.lock opt/kbench/w4dense/bench_w4dense [/tmp/sgemv_dense_gold]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../../crates/inkwell-cuda/src/kernels.cu"
#include "w4dense_kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// Raw streaming uint4 read ceiling kernel (measures physical DRAM floor at exact byte size)
__global__ void __launch_bounds__(256, 8) stream_read_u4(
    const uint4* __restrict__ src, uint32_t* __restrict__ out, size_t n_u4)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (size_t i = idx; i < n_u4; i += stride) {
        uint4 v = __ldg(src + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) out[0] = acc;
}

// Deterministic PRNG
static uint32_t g_rng = 0x12345678u;
static inline uint32_t xrnd() {
    uint32_t x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng = x;
    return x;
}
static inline float rand_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((xrnd() & 0xFFFFFFu) / 16777216.0f);
}
static inline __half f2h_host(float v) { return __float2half(v); }
static inline float h2f_host(__half h) { return __half2float(h); }

// Shipped EXL3 dense split-K heuristic (INKWELL_SGEMV_DENSE_TGT = 32 default)
static int pick_exl3_nky(int tn, int tk, int tgt = 32) {
    const int nb = (tn + 7) / 8;
    int nky = 1;
    while (nb * nky < tgt && nky < 8 && nky * 2 <= tk) nky *= 2;
    return nky;
}

struct Exl3Shape {
    std::string name;
    int tk, tn, ka, hk;
    size_t nb;
    int k() const { return tk * 16; }
    int n_exl3() const { return tn * 16; }
    int n_real() const { return (name == "unembed") ? 200058 : tn * 16; }
};

struct BenchRes {
    float ctrl_min, ctrl_med;
    float cand_min, cand_med;
};

template <typename FCtrl, typename FCand>
static BenchRes bench_interleaved(FCtrl&& f_ctrl, FCand&& f_cand, int reps = 31) {
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    for (int i = 0; i < 4; ++i) { f_ctrl(); f_cand(); }
    CK(cudaDeviceSynchronize());

    std::vector<float> tc, tk;
    tc.reserve(reps); tk.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        float ms0 = 0.f, ms1 = 0.f;
        if ((i & 1) == 0) {
            CK(cudaEventRecord(ev0)); f_ctrl(); CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
            CK(cudaEventElapsedTime(&ms0, ev0, ev1));
            CK(cudaEventRecord(ev0)); f_cand(); CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
            CK(cudaEventElapsedTime(&ms1, ev0, ev1));
        } else {
            CK(cudaEventRecord(ev0)); f_cand(); CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
            CK(cudaEventElapsedTime(&ms1, ev0, ev1));
            CK(cudaEventRecord(ev0)); f_ctrl(); CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
            CK(cudaEventElapsedTime(&ms0, ev0, ev1));
        }
        tc.push_back(ms0 * 1e3f);
        tk.push_back(ms1 * 1e3f);
    }
    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    std::sort(tc.begin(), tc.end());
    std::sort(tk.begin(), tk.end());
    return { tc[0], tc[reps / 2], tk[0], tk[reps / 2] };
}

// ============================================================================
// CPU Packer & FP64 Reference (Row-Major + deterministic load-time N32 transpose)
// ============================================================================
struct PackedMatrix {
    int N, K, bits, G;
    bool asym;
    // Row-major arrays
    std::vector<uint32_t> wlo;    // [N, K/8]
    std::vector<uint32_t> whi;    // [N, K/32] for W5, [N, K/16] for W6
    std::vector<uint16_t> scales; // [N, K/G] for sym, [N, K/G, 2] for asym
    // N32-tiled arrays (exact load-time 32-row transpose)
    std::vector<uint4>    wlo_t;  // [nt, K/32, 32]
    std::vector<uint32_t> whi5_t; // [nt, K/32, 32] (W5)
    std::vector<uint2>    whi6_t; // [nt, K/32, 32] (W6)
    std::vector<uint16_t> s_t;    // [nt, K/G, 32] (sym) or [nt, K/G, 32, 2] (asym)
    // FP64 reference
    std::vector<double> ref_y;
    double ref_max_abs = 1.0;
    uint32_t ref_argmax = 0;

    size_t rm_bytes() const {
        return wlo.size() * 4 + whi.size() * 4 + scales.size() * 2;
    }
    size_t n32_bytes() const {
        return wlo_t.size() * 16 + whi5_t.size() * 4 + whi6_t.size() * 8 + s_t.size() * 2;
    }
};

static PackedMatrix build_and_ref(int N, int K, int bits, int G, bool asym, const std::vector<float>& x, bool need_rm = false) {
    PackedMatrix M;
    M.N = N; M.K = K; M.bits = bits; M.G = G; M.asym = asym;
    const int ng = K / G;
    const int k_u32 = K / 8;
    const int k_u4 = K / 32;
    const int nt = (N + 31) / 32;
    const uint32_t umax = (1u << bits) - 1u;
    const double zero_off = (double)(1u << (bits - 1));

    if (need_rm) {
        M.wlo.assign((size_t)N * k_u32, 0u);
        if (bits == 5) M.whi.assign((size_t)N * (K / 32), 0u);
        else if (bits == 6) M.whi.assign((size_t)N * (K / 16), 0u);
        M.scales.assign((size_t)N * ng * (asym ? 2 : 1), 0u);
    }
    M.wlo_t.assign((size_t)nt * k_u4 * 32, make_uint4(0, 0, 0, 0));
    if (bits == 5) M.whi5_t.assign((size_t)nt * k_u4 * 32, 0u);
    else if (bits == 6) M.whi6_t.assign((size_t)nt * k_u4 * 32, make_uint2(0, 0));
    M.s_t.assign((size_t)nt * ng * 32 * (asym ? 2 : 1), 0u);
    M.ref_y.assign(N, 0.0);

    std::vector<double> sxg(ng, 0.0);
    for (int g = 0; g < ng; ++g) {
        double s = 0.0;
        for (int j = 0; j < G; ++j) s += (double)x[g * G + j];
        sxg[g] = s;
    }

    g_rng = 0x9E3779B9u ^ (uint32_t)(N * 31 + K * 7 + bits * 131 + G + (asym ? 1 : 0));

    double max_abs = 0.0;
    double best_v = -1e300;
    uint32_t best_i = 0;

    std::vector<uint32_t> row_lo(k_u32);
    std::vector<uint32_t> row_hi(bits == 5 ? (K / 32) : (bits == 6 ? (K / 16) : 0));

    for (int r = 0; r < N; ++r) {
        std::fill(row_hi.begin(), row_hi.end(), 0u);
        const int tile = r >> 5;
        const int lane = r & 31;
        double row_acc = 0.0;

        for (int g = 0; g < ng; ++g) {
            __half h_sc = f2h_host(rand_uniform(0.002f, 0.04f));
            __half h_mn = asym ? f2h_host(rand_uniform(-0.35f, -0.02f)) : f2h_host(0.f);
            double sc = (double)h2f_host(h_sc);
            double bias = asym ? (double)h2f_host(h_mn) : (-zero_off * sc);

            if (asym) {
                uint16_t s0, s1;
                memcpy(&s0, &h_sc, 2); memcpy(&s1, &h_mn, 2);
                const size_t tidx = (((size_t)tile * ng + g) * 32 + lane) * 2;
                M.s_t[tidx + 0] = s0; M.s_t[tidx + 1] = s1;
                if (need_rm) {
                    M.scales[((size_t)r * ng + g) * 2 + 0] = s0;
                    M.scales[((size_t)r * ng + g) * 2 + 1] = s1;
                }
            } else {
                uint16_t s0;
                memcpy(&s0, &h_sc, 2);
                M.s_t[((size_t)tile * ng + g) * 32 + lane] = s0;
                if (need_rm) M.scales[(size_t)r * ng + g] = s0;
            }

            double g_dot = 0.0;
            const int w_per_g = G / 8;
            for (int j = 0; j < w_per_g; ++j) {
                const int col0 = g * G + j * 8;
                uint32_t p = 0u;
                uint32_t u[8];
                for (int t = 0; t < 8; ++t) {
                    u[t] = xrnd() & umax;
                    g_dot += (double)u[t] * (double)x[col0 + t];
                }
                for (int b = 0; b < 4; ++b) {
                    uint32_t byte_val = (u[b] & 15u) | ((u[4 + b] & 15u) << 4);
                    p |= (byte_val << (b * 8));
                }
                const int widx = g * w_per_g + j;
                row_lo[widx] = p;

                if (bits == 5) {
                    const int hidx = widx >> 2;
                    const int k_in_h = widx & 3;
                    uint32_t H = row_hi[hidx];
                    for (int b = 0; b < 4; ++b) {
                        H |= (((u[b] >> 4) & 1u) << (b * 8 + k_in_h));
                        H |= (((u[4 + b] >> 4) & 1u) << (b * 8 + 4 + k_in_h));
                    }
                    row_hi[hidx] = H;
                } else if (bits == 6) {
                    const int hidx = widx >> 1;
                    const int k_in_h = widx & 1;
                    uint32_t H = row_hi[hidx];
                    for (int b = 0; b < 4; ++b) {
                        H |= (((u[b] >> 4) & 3u) << (b * 8 + 2 * k_in_h));
                        H |= (((u[4 + b] >> 4) & 3u) << (b * 8 + 4 + 2 * k_in_h));
                    }
                    row_hi[hidx] = H;
                }
            }
            row_acc += sc * g_dot + bias * sxg[g];
        }

        // Write into N32 tiled arrays
        for (int u4 = 0; u4 < k_u4; ++u4) {
            const size_t tidx = ((size_t)tile * k_u4 + u4) * 32 + lane;
            M.wlo_t[tidx] = make_uint4(row_lo[u4 * 4 + 0], row_lo[u4 * 4 + 1], row_lo[u4 * 4 + 2], row_lo[u4 * 4 + 3]);
            if (bits == 5) M.whi5_t[tidx] = row_hi[u4];
            else if (bits == 6) M.whi6_t[tidx] = make_uint2(row_hi[u4 * 2 + 0], row_hi[u4 * 2 + 1]);
        }
        if (need_rm) {
            memcpy(M.wlo.data() + (size_t)r * k_u32, row_lo.data(), k_u32 * 4);
            if (!row_hi.empty()) memcpy(M.whi.data() + (size_t)r * row_hi.size(), row_hi.data(), row_hi.size() * 4);
        }

        M.ref_y[r] = row_acc;
        max_abs = std::max(max_abs, fabs(row_acc));
        if (row_acc > best_v) { best_v = row_acc; best_i = (uint32_t)r; }
    }
    M.ref_max_abs = (max_abs > 0.0) ? max_abs : 1.0;
    M.ref_argmax = best_i;
    return M;
}

static double check_output(const float* dy, const PackedMatrix& M) {
    std::vector<float> hy(M.N);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    CK(cudaMemcpy(hy.data(), dy, (size_t)M.N * 4, cudaMemcpyDeviceToHost));
    double err = 0.0;
    for (int i = 0; i < M.N; ++i) err = std::max(err, fabs((double)hy[i] - M.ref_y[i]));
    return err / M.ref_max_abs;
}

struct DevN32 {
    uint4* d_wlo = nullptr;
    void*  d_whi = nullptr;
    void*  d_sc  = nullptr;
    void upload(const PackedMatrix& M) {
        CK(cudaMalloc(&d_wlo, M.wlo_t.size() * 16));
        CK(cudaMemcpy(d_wlo, M.wlo_t.data(), M.wlo_t.size() * 16, cudaMemcpyHostToDevice));
        if (!M.whi5_t.empty()) {
            CK(cudaMalloc(&d_whi, M.whi5_t.size() * 4));
            CK(cudaMemcpy(d_whi, M.whi5_t.data(), M.whi5_t.size() * 4, cudaMemcpyHostToDevice));
        } else if (!M.whi6_t.empty()) {
            CK(cudaMalloc(&d_whi, M.whi6_t.size() * 8));
            CK(cudaMemcpy(d_whi, M.whi6_t.data(), M.whi6_t.size() * 8, cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&d_sc, M.s_t.size() * 2));
        CK(cudaMemcpy(d_sc, M.s_t.data(), M.s_t.size() * 2, cudaMemcpyHostToDevice));
    }
    void free_all() {
        if (d_wlo) cudaFree(d_wlo);
        if (d_whi) cudaFree(d_whi);
        if (d_sc)  cudaFree(d_sc);
        d_wlo = nullptr; d_whi = nullptr; d_sc = nullptr;
    }
};

struct DevRM {
    uint32_t* d_wlo = nullptr;
    uint32_t* d_whi = nullptr;
    void*     d_sc  = nullptr;
    void upload(const PackedMatrix& M) {
        CK(cudaMalloc(&d_wlo, M.wlo.size() * 4));
        CK(cudaMemcpy(d_wlo, M.wlo.data(), M.wlo.size() * 4, cudaMemcpyHostToDevice));
        if (!M.whi.empty()) {
            CK(cudaMalloc(&d_whi, M.whi.size() * 4));
            CK(cudaMemcpy(d_whi, M.whi.data(), M.whi.size() * 4, cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&d_sc, M.scales.size() * 2));
        CK(cudaMemcpy(d_sc, M.scales.data(), M.scales.size() * 2, cudaMemcpyHostToDevice));
    }
    void free_all() {
        if (d_wlo) cudaFree(d_wlo);
        if (d_whi) cudaFree(d_whi);
        if (d_sc)  cudaFree(d_sc);
        d_wlo = nullptr; d_whi = nullptr; d_sc = nullptr;
    }
};

struct N32Cfg {
    int nwarps;
    int wpt;
    int max_grid;
    bool stage_x;
};

static size_t n32_smem_f32(int K, int nwarps, int wpt, bool stage_x) {
    int red = (wpt > 1) ? ((nwarps / wpt) * wpt * 32) : 0;
    return (size_t)(48 + red + (stage_x ? K : 0)) * sizeof(float);
}
static size_t n32_smem_i16(int K, int nwarps, int wpt) {
    int red = (wpt > 1) ? ((nwarps / wpt) * wpt * 32) : 0;
    return (size_t)(96 + red) * sizeof(int) + (size_t)K * sizeof(int16_t);
}

template <int BITS, int G, bool ASYM>
static void launch_n32_f32(const DevN32& D, const float* dx, float* dy, int N, int K, N32Cfg cfg) {
    const int nt = (N + 31) / 32;
    const int tpb = cfg.nwarps / cfg.wpt;
    const int nb = std::min((nt + tpb - 1) / tpb, cfg.max_grid);
    const size_t sm = n32_smem_f32(K, cfg.nwarps, cfg.wpt, cfg.stage_x);

    if (cfg.nwarps == 16 && cfg.wpt == 16 && cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 16, 16, true><<<nb, 512, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 16 && cfg.wpt == 16 && !cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 16, 16, false><<<nb, 512, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 8 && cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 8, true><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 8 && !cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 8, false><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 4 && cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 4, true><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 4 && !cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 4, false><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 2 && cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 2, true><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 2 && !cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 2, false><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 1 && cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 1, true><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 1 && !cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 8, 1, false><<<nb, 256, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 4 && cfg.wpt == 4 && cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 4, 4, true><<<nb, 128, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 4 && cfg.wpt == 4 && !cfg.stage_x)
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 4, 4, false><<<nb, 128, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
    else
        w4d::gemv_wn_n32_f32<BITS, G, ASYM, 4, 1, true><<<nb, 128, sm>>>(D.d_wlo, D.d_whi, D.d_sc, dx, dy, N, K);
}

template <int G, bool ASYM>
static void launch_n32_dp2a(const DevN32& D, const float* dx, float* dy, int N, int K, N32Cfg cfg) {
    const int nt = (N + 31) / 32;
    const int tpb = cfg.nwarps / cfg.wpt;
    const int nb = std::min((nt + tpb - 1) / tpb, cfg.max_grid);
    const size_t sm = n32_smem_i16(K, cfg.nwarps, cfg.wpt);
    if (cfg.nwarps == 16 && cfg.wpt == 16)
        w4d::gemv_w4_n32_dp2a_i16<G, ASYM, 16, 16><<<nb, 512, sm>>>(D.d_wlo, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 8)
        w4d::gemv_w4_n32_dp2a_i16<G, ASYM, 8, 8><<<nb, 256, sm>>>(D.d_wlo, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 4)
        w4d::gemv_w4_n32_dp2a_i16<G, ASYM, 8, 4><<<nb, 256, sm>>>(D.d_wlo, D.d_sc, dx, dy, N, K);
    else if (cfg.nwarps == 8 && cfg.wpt == 2)
        w4d::gemv_w4_n32_dp2a_i16<G, ASYM, 8, 2><<<nb, 256, sm>>>(D.d_wlo, D.d_sc, dx, dy, N, K);
    else
        w4d::gemv_w4_n32_dp2a_i16<G, ASYM, 8, 1><<<nb, 256, sm>>>(D.d_wlo, D.d_sc, dx, dy, N, K);
}

struct FormatSummary {
    std::string label;
    int bits, G;
    bool asym, is_dp2a, is_rm;
    double bpw;
    float shape_min[7];
    float shape_med[7];
    float shape_gbs[7];
    float ctrl_s_med[7];
    float ctrl_full_med[7];
    float argmax_min = 0.f;
    float argmax_med = 0.f;
    float argmax_gbs = 0.f;
    double max_rel_err = 0.0;
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string dir = (argc > 1) ? argv[1] : "/tmp/sgemv_dense_gold";

    std::vector<Exl3Shape> raw_shapes;
    {
        char p[512]; snprintf(p, sizeof p, "%s/meta.txt", dir.c_str());
        FILE* f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", p); return 1; }
        char nm[64], full[512]; int tk, tn, ka, hk; long long bytes;
        while (fscanf(f, "%63s %d %d %d %d %lld %511s", nm, &tk, &tn, &ka, &hk, &bytes, full) == 7) {
            raw_shapes.push_back({nm, tk, tn, ka, hk, (size_t)bytes});
        }
        fclose(f);
    }

    const char* ORDER[7] = {"wq_du", "gate_attn", "wr_du", "wk_dv", "wv_dv", "wo_ud", "unembed"};
    std::vector<Exl3Shape> shapes;
    for (int i = 0; i < 7; ++i)
        for (auto& s : raw_shapes) if (s.name == ORDER[i]) shapes.push_back(s);
    if (shapes.size() != 7) { fprintf(stderr, "expected 7 shapes, found %zu\n", shapes.size()); return 1; }

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (sm_%d%d, %d SMs, %.1f GB/s mem peak)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6);

    const int MAX_N = 201088;
    const int MAX_K = 3072;
    float *d_x = nullptr, *d_xh = nullptr, *d_y = nullptr, *d_part = nullptr, *d_exl3_t = nullptr;
    uint16_t *d_suh = nullptr, *d_svh = nullptr;
    int *d_exl3_ctr = nullptr;
    uint32_t *d_ctr = nullptr, *d_blk_id = nullptr, *d_ticket = nullptr, *d_out_tok = nullptr, *d_sink = nullptr;
    float *d_blk_val = nullptr, *d_out_val = nullptr;

    CK(cudaMalloc(&d_x, MAX_K * 4));
    CK(cudaMalloc(&d_xh, MAX_K * 4));
    CK(cudaMalloc(&d_y, (size_t)MAX_N * 4));
    CK(cudaMalloc(&d_part, (size_t)8 * 3072 * 4));
    CK(cudaMalloc(&d_exl3_t, (size_t)8 * MAX_N * 4));
    CK(cudaMalloc(&d_suh, MAX_K * 2));
    CK(cudaMalloc(&d_svh, (size_t)MAX_N * 2));
    CK(cudaMalloc(&d_exl3_ctr, 4096 * 4));
    CK(cudaMemset(d_exl3_ctr, 0, 4096 * 4));
    CK(cudaMalloc(&d_ctr, 4096 * 4));
    CK(cudaMemset(d_ctr, 0, 4096 * 4));
    CK(cudaMalloc(&d_blk_val, 512 * 4));
    CK(cudaMalloc(&d_blk_id, 512 * 4));
    CK(cudaMalloc(&d_ticket, 4));
    CK(cudaMemset(d_ticket, 0, 4));
    CK(cudaMalloc(&d_out_tok, 4));
    CK(cudaMalloc(&d_out_val, 4));
    CK(cudaMalloc(&d_sink, 16));

    // Fill suh/svh with +1.0f in fp16 (0x3C00)
    {
        std::vector<uint16_t> ones(MAX_N, 0x3C00u);
        CK(cudaMemcpy(d_suh, ones.data(), MAX_K * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_svh, ones.data(), (size_t)MAX_N * 2, cudaMemcpyHostToDevice));
    }

    auto make_x = [](int K) {
        g_rng = 0x55AA1234u ^ (uint32_t)K;
        std::vector<float> x(K);
        for (int i = 0; i < K; ++i) {
            float g = 0.f;
            for (int t = 0; t < 4; ++t) g += rand_uniform(-1.f, 1.f);
            x[i] = g * 0.5f;
        }
        x[xrnd() % K] *= 6.f;
        return x;
    };
    const std::vector<float> hx_2304 = make_x(2304);
    const std::vector<float> hx_3072 = make_x(3072);

    auto load_exl3_dev = [&](const Exl3Shape& S) -> uint32_t* {
        char p[512]; snprintf(p, sizeof p, "%s/%s.bin", dir.c_str(), S.name.c_str());
        std::vector<uint32_t> h(S.nb / 4);
        FILE* f = fopen(p, "rb");
        if (!f || fread(h.data(), 1, S.nb, f) != S.nb) { fprintf(stderr, "failed read %s\n", p); exit(1); }
        fclose(f);
        uint32_t* d = nullptr;
        CK(cudaMalloc(&d, S.nb));
        CK(cudaMemcpy(d, h.data(), S.nb, cudaMemcpyHostToDevice));
        return d;
    };

    int total_fails = 0;

    // ========================================================================
    // PHASE 0: Baseline EXL3 (bare gemv_s vs shipped had_in+gemv_sf) & RAW STREAM ceiling
    // ========================================================================
    printf("\n==================== PHASE 0: EXL3 CONTROLS & RAW DRAM STREAM CEILING ====================\n");
    printf("%-10s %10s %10s %10s %10s %10s %9s\n",
           "shape", "EXL3_KB", "gemv_s(us)", "full(us)", "had_oh(us)", "stream(us)", "str_GB/s");
    float base_gemvs[7], base_full[7];
    for (int si = 0; si < 7; ++si) {
        const Exl3Shape& S = shapes[si];
        const int K = S.k();
        const auto& hx = (K == 2304) ? hx_2304 : hx_3072;
        CK(cudaMemcpy(d_x, hx.data(), K * 4, cudaMemcpyHostToDevice));
        uint32_t* d_exl3 = load_exl3_dev(S);
        const int exl3_nky = pick_exl3_nky(S.tn, S.tk, 32);
        const int nb = (S.tn + 7) / 8;

        auto run_gemvs = [&]() {
            inkwell_exl3_gemv_s<<<dim3(nb, exl3_nky, 1), 256>>>(
                d_exl3, d_x, d_exl3_t, S.tk, S.tn, S.ka, S.hk, 0);
        };
        auto run_full = [&]() {
            inkwell_exl3_had_in<<<(unsigned)(K / 128), 128, 512>>>(d_x, d_suh, d_xh, K);
            inkwell_exl3_gemv_sf<<<dim3(nb, exl3_nky, 1), 256>>>(
                d_exl3, d_xh, d_svh, d_y, d_exl3_t, d_exl3_ctr, S.tk, S.tn, S.ka, S.hk, 0);
        };
        BenchRes rb = bench_interleaved(run_gemvs, run_full, 31);
        base_gemvs[si] = rb.ctrl_med;
        base_full[si]  = rb.cand_med;

        // Raw stream bandwidth at exact W4 g128 sym byte count
        const size_t w4_bytes = (size_t)S.n_real() * (K / 2) + (size_t)S.n_real() * (K / 128) * 2;
        uint4* d_str = nullptr;
        CK(cudaMalloc(&d_str, w4_bytes + 16));
        CK(cudaMemset(d_str, 1, w4_bytes));
        const size_t n_u4 = w4_bytes / 16;
        auto run_str = [&]() { stream_read_u4<<<160, 256>>>(d_str, d_sink, n_u4); };
        BenchRes rs = bench_interleaved(run_gemvs, run_str, 31);
        double str_gbs = (double)(n_u4 * 16) / (rs.cand_med * 1e-6) / 1e9;
        printf("%-10s %10.1f %10.2f %10.2f %10.2f %10.2f %9.1f\n",
               S.name.c_str(), S.nb / 1024.0, rb.ctrl_med, rb.cand_med,
               rb.cand_med - rb.ctrl_med, rs.cand_med, str_gbs);

        CK(cudaFree(d_str));
        CK(cudaFree(d_exl3));
    }
    {
        double a6_s = 0, a6_f = 0;
        for (int i = 0; i < 6; ++i) { a6_s += base_gemvs[i]; a6_f += base_full[i]; }
        printf("EXL3 baseline total: bare gemv_s = %.3f ms (48x %.1f us + %.3f ms); shipped had_in+gemv_sf = %.3f ms (48x %.1f us + %.3f ms)\n",
               (48.0 * a6_s + base_gemvs[6]) * 1e-3, a6_s, base_gemvs[6] * 1e-3,
               (48.0 * a6_f + base_full[6]) * 1e-3, a6_f, base_full[6] * 1e-3);
    }

    // ========================================================================
    // PHASE 1: W4 g128 sym Unpack-Trick & Layout (Row-Major vs N32 Tiled) Ablation
    // ========================================================================
    printf("\n==================== PHASE 1: W4 g128 sym UNPACK & LAYOUT ABLATION ====================\n");
    printf("%-10s %-30s %10s %9s %9s %8s %8s %8s\n",
           "shape", "variant", "max_rel", "med(us)", "min(us)", "GB/s", "vs_gemvS", "vs_full");

    // Auto-tuned winning N32 configs per shape class (populated from Phase 1 measurements)
    N32Cfg best_n32[7] = {
        {8,  4, 160, true}, // 0: wq_du
        {8,  4, 160, true}, // 1: gate_attn (same shape as wq_du)
        {16, 16, 160, true},// 2: wr_du
        {8,  8, 160, true}, // 3: wk_dv
        {8,  8, 160, true}, // 4: wv_dv (same shape as wk_dv)
        {8,  4, 160, true}, // 5: wo_ud
        {8,  1, 160, true}, // 6: unembed
    };

    const int ABL_IDX[5] = {0, 2, 3, 5, 6}; // wq_du, wr_du, wk_dv, wo_ud, unembed
    for (int ii = 0; ii < 5; ++ii) {
        const int si_abl = ABL_IDX[ii];
        const Exl3Shape& S = shapes[si_abl];
        const int K = S.k(), N = S.n_real();
        const auto& hx = (K == 2304) ? hx_2304 : hx_3072;
        CK(cudaMemcpy(d_x, hx.data(), K * 4, cudaMemcpyHostToDevice));
        uint32_t* d_exl3 = load_exl3_dev(S);
        const int exl3_nky = pick_exl3_nky(S.tn, S.tk, 32);
        const int nb_e = (S.tn + 7) / 8;
        auto run_ctrl = [&]() {
            inkwell_exl3_gemv_s<<<dim3(nb_e, exl3_nky, 1), 256>>>(
                d_exl3, d_x, d_exl3_t, S.tk, S.tn, S.ka, S.hk, 0);
        };

        PackedMatrix M = build_and_ref(N, K, 4, 128, false, hx, true);
        DevRM Drm; Drm.upload(M);
        DevN32 Dn; Dn.upload(M);
        const double bytes_rm  = (double)M.rm_bytes()  + (double)K * 4.0 + (double)N * 4.0;
        const double bytes_n32 = (double)M.n32_bytes() + (double)K * 4.0 + (double)N * 4.0;

        float best_med_f32 = 1e30f;

        auto test_fn = [&](const char* label, auto&& run_cand, double bytes_moved, double tol = 5e-6) -> float {
            run_cand();
            double rel = check_output(d_y, M);
            if (rel > tol) { ++total_fails; printf("  [FAIL] %s %s rel=%.3e\n", S.name.c_str(), label, rel); }
            BenchRes r = bench_interleaved(run_ctrl, run_cand, 25);
            double gbs = bytes_moved / (r.cand_med * 1e-6) / 1e9;
            printf("%-10s %-30s %10.2e %9.2f %9.2f %8.1f %7.2fx %7.2fx\n",
                   S.name.c_str(), label, rel, r.cand_med, r.cand_min, gbs,
                   r.ctrl_med / r.cand_med, base_full[si_abl] / r.cand_med);
            return r.cand_med;
        };
        auto test_n32 = [&](const char* label, N32Cfg cfg) {
            float med = test_fn(label, [&]() { launch_n32_f32<4, 128, false>(Dn, d_x, d_y, N, K, cfg); }, bytes_n32);
            if (med < best_med_f32) { best_med_f32 = med; best_n32[si_abl] = cfg; }
        };

        const size_t sm_rm = (48 + K) * 4;
        const int nbx_rm = (N + 15) / 16;
        // Row-major unpack tricks (on wq_du and unembed)
        if (si_abl == 0 || si_abl == 6) {
            test_fn("RM_DUAL_I2F w8 r2 sk1 smem", [&]() {
                w4d::gemv_w4_f32<128, false, 2, 8, 0, true><<<dim3(nbx_rm, 1, 1), 256, sm_rm>>>(
                    Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
            }, bytes_rm);
            test_fn("RM_BFE_I2F  w8 r2 sk1 smem", [&]() {
                w4d::gemv_w4_f32<128, false, 2, 8, 1, true><<<dim3(nbx_rm, 1, 1), 256, sm_rm>>>(
                    Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
            }, bytes_rm);
            test_fn("RM_LOP3_MAG w8 r2 sk1 smem", [&]() {
                w4d::gemv_w4_f32<128, false, 2, 8, 2, true><<<dim3(nbx_rm, 1, 1), 256, sm_rm>>>(
                    Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
            }, bytes_rm);
            test_fn("RM_LOP3_FLD w8 r2 sk1 smem", [&]() {
                w4d::gemv_w4_f32<128, false, 2, 8, 3, true><<<dim3(nbx_rm, 1, 1), 256, sm_rm>>>(
                    Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
            }, bytes_rm, 2e-5);
            test_fn("RM_DP2A_I16 w8 r2 sk1 smem", [&]() {
                w4d::gemv_w4_dp2a_i16<128, false, 2, 8><<<dim3(nbx_rm, 1, 1), 256, 96*4 + K*2>>>(
                    Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
            }, bytes_rm, 2e-4);
        } else {
            test_fn("RM_DUAL_I2F w8 r2 sk1 smem", [&]() {
                w4d::gemv_w4_f32<128, false, 2, 8, 0, true><<<dim3(nbx_rm, 1, 1), 256, sm_rm>>>(
                    Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
            }, bytes_rm);
        }

        // N32-tiled geometry sweep (smem vs ldg, WPT=1/2/4/8/16, dp2a)
        if (N <= 1024) {
            test_n32("N32_F32 w16 wpt16 smem", {16, 16, 160, true});
            test_n32("N32_F32 w16 wpt16 ldg",  {16, 16, 160, false});
            test_n32("N32_F32 w8  wpt8  smem", {8,  8,  160, true});
            test_n32("N32_F32 w8  wpt8  ldg",  {8,  8,  160, false});
            test_n32("N32_F32 w4  wpt4  smem", {4,  4,  160, true});
            test_n32("N32_F32 w4  wpt4  ldg",  {4,  4,  160, false});
            test_fn("N32_DP2A w8 wpt8  smem", [&]() { launch_n32_dp2a<128, false>(Dn, d_x, d_y, N, K, {8, 8, 160, true}); }, bytes_n32, 2e-4);
            test_fn("N32_DP2A w16 wpt16 smem",[&]() { launch_n32_dp2a<128, false>(Dn, d_x, d_y, N, K, {16, 16, 160, true}); }, bytes_n32, 2e-4);
        } else if (N <= 4096) {
            test_n32("N32_F32 w8  wpt8  smem", {8, 8, 160, true});
            test_n32("N32_F32 w8  wpt8  ldg",  {8, 8, 160, false});
            test_n32("N32_F32 w8  wpt4  smem", {8, 4, 160, true});
            test_n32("N32_F32 w8  wpt4  ldg",  {8, 4, 160, false});
            test_n32("N32_F32 w4  wpt4  smem", {4, 4, 160, true});
            test_n32("N32_F32 w4  wpt4  ldg",  {4, 4, 160, false});
            test_n32("N32_F32 w8  wpt2  smem", {8, 2, 160, true});
            test_n32("N32_F32 w8  wpt2  ldg",  {8, 2, 160, false});
            test_fn("N32_DP2A w8 wpt4  smem", [&]() { launch_n32_dp2a<128, false>(Dn, d_x, d_y, N, K, {8, 4, 160, true}); }, bytes_n32, 2e-4);
            test_fn("N32_DP2A w8 wpt8  smem", [&]() { launch_n32_dp2a<128, false>(Dn, d_x, d_y, N, K, {8, 8, 160, true}); }, bytes_n32, 2e-4);
        } else {
            test_n32("N32_F32 w8 wpt1 g80 smem",  {8, 1, 80,  true});
            test_n32("N32_F32 w8 wpt1 g160 smem", {8, 1, 160, true});
            test_n32("N32_F32 w8 wpt1 g160 ldg",  {8, 1, 160, false});
            test_n32("N32_F32 w4 wpt1 g160 smem", {4, 1, 160, true});
            test_fn("N32_DP2A w8 wpt1 g160 smem",[&]() { launch_n32_dp2a<128, false>(Dn, d_x, d_y, N, K, {8, 1, 160, true}); }, bytes_n32, 2e-4);
        }

        Drm.free_all();
        Dn.free_all();
        CK(cudaFree(d_exl3));
    }
    best_n32[1] = best_n32[0]; // gate_attn == wq_du
    best_n32[4] = best_n32[3]; // wv_dv == wk_dv

    printf("\nSelected best N32 configs for Phase 2:\n");
    for (int i = 0; i < 7; ++i) {
        printf("  %-10s: nwarps=%2d wpt=%2d max_grid=%3d stage_x=%d\n",
               shapes[i].name.c_str(), best_n32[i].nwarps, best_n32[i].wpt, best_n32[i].max_grid, (int)best_n32[i].stage_x);
    }

    // ========================================================================
    // PHASE 2: Full 7-Shape Head-to-Head Matrix Across All Formats (N32 Tiled + RM comparison)
    // ========================================================================
    printf("\n==================== PHASE 2: FULL 7-SHAPE INTERLEAVED BENCHMARK (31 reps) ====================\n");

    auto pick_n32 = [&](int si, bool is_dp2a) -> N32Cfg {
        N32Cfg c = best_n32[si];
        if (is_dp2a) {
            c.stage_x = true;
            if (c.nwarps == 4) { c.nwarps = 8; c.wpt = std::min(8, c.wpt * 2); }
        }
        return c;
    };

    std::vector<FormatSummary> summaries;

    auto run_format = [&](auto bits_c, auto g_c, auto asym_c, const char* label, bool is_dp2a, bool is_rm) {
        constexpr int BITS = decltype(bits_c)::value;
        constexpr int G = decltype(g_c)::value;
        constexpr bool ASYM = decltype(asym_c)::value;
        FormatSummary fs;
        fs.label = label;
        fs.bits = BITS; fs.G = G; fs.asym = ASYM; fs.is_dp2a = is_dp2a; fs.is_rm = is_rm;
        fs.bpw = (double)BITS + (ASYM ? 32.0 : 16.0) / (double)G;
        printf("\n--- Format: %s (%.3f bpw) ---\n", label, fs.bpw);
        printf("%-10s %8s %10s %9s %9s %8s %9s %8s %8s\n",
               "shape", "bytes_KB", "max_rel", "med(us)", "min(us)", "GB/s", "ctrlS(us)", "vs_gemvS", "vs_full");

        for (int si = 0; si < 7; ++si) {
            const Exl3Shape& S = shapes[si];
            const int K = S.k(), N = S.n_real();
            const auto& hx = (K == 2304) ? hx_2304 : hx_3072;
            CK(cudaMemcpy(d_x, hx.data(), K * 4, cudaMemcpyHostToDevice));
            uint32_t* d_exl3 = load_exl3_dev(S);
            const int exl3_nky = pick_exl3_nky(S.tn, S.tk, 32);
            const int nb_e = (S.tn + 7) / 8;
            auto run_ctrl = [&]() {
                inkwell_exl3_gemv_s<<<dim3(nb_e, exl3_nky, 1), 256>>>(
                    d_exl3, d_x, d_exl3_t, S.tk, S.tn, S.ka, S.hk, 0);
            };

            PackedMatrix M = build_and_ref(N, K, BITS, G, ASYM, hx, is_rm);
            DevN32 Dn; DevRM Drm;
            if (is_rm) Drm.upload(M);
            else       Dn.upload(M);
            const size_t wbytes = is_rm ? M.rm_bytes() : M.n32_bytes();
            const double bytes_moved = (double)wbytes + (double)K * 4.0 + (double)N * 4.0;
            const N32Cfg cfg = pick_n32(si, is_dp2a);

            auto run_cand = [&]() {
                if (is_rm) {
                    const int nbx = (N + 15) / 16;
                    const size_t sm = (48 + K) * sizeof(float);
                    if constexpr (BITS == 4)
                        w4d::gemv_w4_f32<G, ASYM, 2, 8, 0, true><<<dim3(nbx, 1, 1), 256, sm>>>(
                            Drm.d_wlo, Drm.d_sc, d_x, d_y, d_part, d_ctr, N, K);
                } else if (is_dp2a) {
                    if constexpr (BITS == 4)
                        launch_n32_dp2a<G, ASYM>(Dn, d_x, d_y, N, K, cfg);
                } else {
                    launch_n32_f32<BITS, G, ASYM>(Dn, d_x, d_y, N, K, cfg);
                }
            };

            run_cand();
            double rel = check_output(d_y, M);
            fs.max_rel_err = std::max(fs.max_rel_err, rel);
            const double tol = is_dp2a ? 2e-4 : 5e-6;
            if (rel > tol) { ++total_fails; printf("  [FAIL] %s %s rel=%.3e\n", label, S.name.c_str(), rel); }

            BenchRes r = bench_interleaved(run_ctrl, run_cand, 31);
            double gbs = bytes_moved / (r.cand_med * 1e-6) / 1e9;
            fs.shape_min[si] = r.cand_min;
            fs.shape_med[si] = r.cand_med;
            fs.shape_gbs[si] = (float)gbs;
            fs.ctrl_s_med[si] = r.ctrl_med;
            fs.ctrl_full_med[si] = base_full[si];
            printf("%-10s %8.1f %10.2e %9.2f %9.2f %8.1f %9.2f %7.2fx %7.2fx\n",
                   S.name.c_str(), wbytes / 1024.0, rel, r.cand_med, r.cand_min, gbs,
                   r.ctrl_med, r.ctrl_med / r.cand_med, base_full[si] / r.cand_med);

            // Fused unembed + argmax
            if (si == 6 && !is_dp2a) {
                const int G_blks = 160;
                const size_t sm = (48 + K) * sizeof(float);
                auto run_argmax = [&]() {
                    if (is_rm) {
                        w4d::unembed_argmax_f32<BITS, G, ASYM, 8><<<G_blks, 256, sm>>>(
                            Drm.d_wlo, Drm.d_whi, Drm.d_sc, d_x, d_blk_val, d_blk_id, d_ticket, d_out_tok, d_out_val, N, K);
                    } else {
                        w4d::unembed_argmax_n32_f32<BITS, G, ASYM, 8><<<G_blks, 256, sm>>>(
                            Dn.d_wlo, Dn.d_whi, Dn.d_sc, d_x, d_blk_val, d_blk_id, d_ticket, d_out_tok, d_out_val, N, K);
                    }
                };
                run_argmax();
                CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
                uint32_t gpu_tok = 0; float gpu_val = 0.f;
                CK(cudaMemcpy(&gpu_tok, d_out_tok, 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&gpu_val, d_out_val, 4, cudaMemcpyDeviceToHost));
                double val_rel = fabs((double)gpu_val - M.ref_y[M.ref_argmax]) / M.ref_max_abs;
                if (gpu_tok != M.ref_argmax) {
                    ++total_fails;
                    printf("  [FAIL ARGMAX] %s got tok=%u (val=%.6f) want tok=%u (val=%.6f)\n",
                           label, gpu_tok, gpu_val, M.ref_argmax, M.ref_y[M.ref_argmax]);
                }
                BenchRes ra = bench_interleaved(run_ctrl, run_argmax, 31);
                double am_bytes = (double)wbytes + (double)K * 4.0;
                double am_gbs = am_bytes / (ra.cand_med * 1e-6) / 1e9;
                fs.argmax_min = ra.cand_min;
                fs.argmax_med = ra.cand_med;
                fs.argmax_gbs = (float)am_gbs;
                printf("%-10s %8.1f %10.2e %9.2f %9.2f %8.1f %9.2f %7.2fx %7.2fx  (tok=%u %s)\n",
                       "unemb+argm", wbytes / 1024.0, val_rel, ra.cand_med, ra.cand_min, am_gbs,
                       ra.ctrl_med, ra.ctrl_med / ra.cand_med, base_full[6] / ra.cand_med,
                       gpu_tok, (gpu_tok == M.ref_argmax ? "EXACT" : "DIFF"));
            }

            if (is_rm) Drm.free_all();
            else       Dn.free_all();
            CK(cudaFree(d_exl3));
        }
        summaries.push_back(fs);
    };

    run_format(std::integral_constant<int,4>{}, std::integral_constant<int,128>{}, std::false_type{}, "W4_g128_sym",    false, false);
    run_format(std::integral_constant<int,4>{}, std::integral_constant<int,128>{}, std::false_type{}, "W4_g128_sym_RM", false, true);
    run_format(std::integral_constant<int,4>{}, std::integral_constant<int,128>{}, std::true_type{},  "W4_g128_asym",   false, false);
    run_format(std::integral_constant<int,4>{}, std::integral_constant<int, 64>{}, std::false_type{}, "W4_g64_sym",     false, false);
    run_format(std::integral_constant<int,4>{}, std::integral_constant<int, 64>{}, std::true_type{},  "W4_g64_asym",    false, false);
    run_format(std::integral_constant<int,4>{}, std::integral_constant<int,128>{}, std::false_type{}, "W4_g128_dp2a",   true,  false);
    run_format(std::integral_constant<int,5>{}, std::integral_constant<int,128>{}, std::false_type{}, "W5_g128_sym",    false, false);
    run_format(std::integral_constant<int,5>{}, std::integral_constant<int,128>{}, std::true_type{},  "W5_g128_asym",   false, false);
    run_format(std::integral_constant<int,5>{}, std::integral_constant<int, 64>{}, std::false_type{}, "W5_g64_sym",     false, false);
    run_format(std::integral_constant<int,6>{}, std::integral_constant<int,128>{}, std::false_type{}, "W6_g128_sym",    false, false);
    run_format(std::integral_constant<int,6>{}, std::integral_constant<int,128>{}, std::true_type{},  "W6_g128_asym",   false, false);
    run_format(std::integral_constant<int,6>{}, std::integral_constant<int, 64>{}, std::false_type{}, "W6_g64_sym",     false, false);

    // ========================================================================
    // SUMMARY TABLE: Modeled ms/token (48 * attn_6 + unembed), STRICTLY same-clock interleaved
    // ========================================================================
    printf("\n==================== MODELED ms/token SUMMARY (48 * attn_6 + unembed, same-clock) ====================\n");
    printf("%-16s %6s %9s %8s %9s %9s %9s %8s %8s %9s\n",
           "format", "bpw", "attn6(us)", "48x(ms)", "unemb(ms)", "fus(ms)", "ctrlS(ms)", "vs_gemvS", "vs_full~", "save_full~");
    double had_ratio[7];
    for (int i = 0; i < 7; ++i) had_ratio[i] = (double)base_full[i] / (double)base_gemvs[i];

    for (auto& fs : summaries) {
        double a6 = 0.0, ca6 = 0.0, ca6_full = 0.0;
        for (int i = 0; i < 6; ++i) {
            a6 += fs.shape_med[i];
            ca6 += fs.ctrl_s_med[i];
            ca6_full += fs.ctrl_s_med[i] * had_ratio[i];
        }
        double fus_us = (fs.argmax_med > 0.f) ? fs.argmax_med : fs.shape_med[6];
        double tot_ms = (48.0 * a6 + fus_us) * 1e-3;
        double ctrl_s_ms = (48.0 * ca6 + fs.ctrl_s_med[6]) * 1e-3;
        double ctrl_f_ms = (48.0 * ca6_full + fs.ctrl_s_med[6] * had_ratio[6]) * 1e-3;
        printf("%-16s %6.3f %9.1f %8.3f %9.3f %9.3f %9.3f %7.2fx %7.2fx %+8.2f ms\n",
               fs.label.c_str(), fs.bpw, a6, 48.0 * a6 * 1e-3, fs.shape_med[6] * 1e-3,
               tot_ms, ctrl_s_ms, ctrl_s_ms / tot_ms, ctrl_f_ms / tot_ms, tot_ms - ctrl_f_ms);
    }

    printf("\nOVERALL VERIFICATION: %s (fails=%d)\n", total_fails == 0 ? "PASS" : "FAIL", total_fails);
    return total_fails ? 1 : 0;
}
