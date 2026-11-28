#include "kernels.cu"
extern "C" const unsigned char ink_ch2nopless_cubin_start[1] = {0};
extern "C" const unsigned char ink_ch2nopless_cubin_end[1] = {0};

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "gqa3_v2_kernel.inc"

#define F3D_SRC_N 4096
#define F3D_NLAYER 8

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        exit(2);
    }
}

__global__ void ink_f3d_quant_from_dump(const __half* __restrict__ kd, const __half* __restrict__ vd,
                                        InkKvQSlot slot, int src_n) {
    __shared__ float ek[128], ev[128], mn[128], mx[128];
    __shared__ unsigned char qk[128], qv[128], qklo[128];
    __shared__ float ksc_s[2], vsc_s[4], vmn_s[4];
    const int hk = blockIdx.y;
    const int d = threadIdx.x;
    const int q0 = blockIdx.x * 8;
    const int q1 = min(q0 + 8, slot.cap);
    for (int qrow = q0; qrow < q1; ++qrow) {
        const int sr = qrow % src_n;
        const size_t src = ((size_t)hk * src_n + sr) * 128 + d;
        ink_kvq_store_quant(slot, qrow, hk, d, __half2float(kd[src]), __half2float(vd[src]),
                            ek, ev, mn, mx, qk, qv, qklo, ksc_s, vsc_s, vmn_s);
    }
}

static bool read_file(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const size_t n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(n);
    f.read((char*)out.data(), (std::streamsize)n);
    return f.good() || f.eof();
}

struct DevLayer {
    InkKvQSlot slot{};
    float* q = nullptr;
    float* bias = nullptr;
    unsigned char* kq = nullptr;
    unsigned char* vq = nullptr;
    unsigned char* meta = nullptr;
};

static float time_kernel(const std::vector<std::function<void()>>& fns, int which, int reps) {
    std::vector<float> ts;
    for (int i = 0; i < reps; ++i) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a);
        fns[which]();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        ts.push_back(ms);
        cuda_check(cudaGetLastError(), "timed launch");
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

static inline void launch_smem3v2(const DevLayer& L, const dim3& gp, int chunk_len,
                                  const int* dpos, const float* mask, float* part, const int* dnkv,
                                  int nh, int q_s2, int b_s2, float scale,
                                  const unsigned char* qbuf, const float* qscales) {
    inkwell_attn_kvq_gqa3_v2_smem3<INK_KVQ_K8V4><<<gp, dim3(96), 2 * 6 * sizeof(InkF3dSRow)>>>(
        L.slot, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

template<int NST>
static inline void launch_rowv2(const DevLayer& L, const dim3& gp, int chunk_len,
                                const int* dpos, const float* mask, float* part, const int* dnkv,
                                int nh, int q_s2, int b_s2, float scale,
                                const unsigned char* qbuf, const float* qscales) {
    constexpr int NW = NST / 2;
    const size_t sm = (size_t)(NW * 3 * 128 + NW * 3 * 2 + NW * 3 * 16) * 4;
    inkwell_attn_kvq_gqa3_v2_row<INK_KVQ_K8V4, NST><<<gp, dim3(NST * 16), sm>>>(
        L.slot, dpos, L.q, L.bias, mask, part, nh, dnkv, q_s2, b_s2, scale,
        chunk_len, 1, 0, 1024, 1024, qbuf, qscales);
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    const int n_kv = argc > 3 ? atoi(argv[3]) : 65536, nh = 24, hkv = 8, hd = 128;
    const int q_s2 = 128, b_s2 = 1025;
    const float scale = 1.0f / sqrtf(128.f);
    const char* mode = argc > 1 ? argv[1] : "real";
    const int chunk_len = argc > 2 ? atoi(argv[2]) : 256;
    const int reps = getenv("F3D_REPS") ? atoi(getenv("F3D_REPS")) : 7;
    const bool uniform = strcmp(mode, "uniform") == 0;
    const std::string dumproot = getenv("F3D_DUMP") ? getenv("F3D_DUMP") :
        "/home/ash/.cache/inkwell-intkv/attn4k-v26-nocache/4096";
    const char* names[F3D_NLAYER] = {"l5p0", "l11p0", "l11p1", "l11p2", "l17p0", "l17p1", "l17p2", "l23p0"};
    if (chunk_len < 32 || chunk_len > 1024) {
        fprintf(stderr, "chunk_len must be in [32,1024]\n");
        return 2;
    }

    std::vector<DevLayer> layers(F3D_NLAYER);
    const size_t kbytes = (size_t)n_kv * 1024, vbytes = (size_t)n_kv * 512, mbytes = (size_t)n_kv * 96;
    const float tau = 1.f + 0.1f * logf((float)n_kv / 8192.f);
    for (int li = 0; li < F3D_NLAYER; ++li) {
        DevLayer& L = layers[li];
        cuda_check(cudaMalloc(&L.kq, kbytes), "malloc kq");
        cuda_check(cudaMalloc(&L.vq, vbytes), "malloc vq");
        cuda_check(cudaMalloc(&L.meta, mbytes), "malloc meta");
        cuda_check(cudaMalloc(&L.q, nh * hd * 4), "malloc q");
        cuda_check(cudaMalloc(&L.bias, nh * b_s2 * 4), "malloc bias");
        if (uniform) {
            cudaMemset(L.kq, 0, kbytes);
            cudaMemset(L.vq, 0, vbytes);
            cudaMemset(L.meta, 0, mbytes);
            cudaMemset(L.q, 0, nh * hd * 4);
            cudaMemset(L.bias, 0, nh * b_s2 * 4);
        } else {
            const std::string dir = dumproot + "/" + names[li];
            std::vector<unsigned char> kb, vb, qb, bb;
            if (!read_file(dir + "/k.f16", kb) || !read_file(dir + "/v.f16", vb) ||
                !read_file(dir + "/q.f32", qb) || !read_file(dir + "/bias.f32", bb)) {
                fprintf(stderr, "missing realistic dump in %s\n", dir.c_str());
                return 2;
            }
            __half *dk = nullptr, *dv = nullptr;
            cuda_check(cudaMalloc(&dk, kb.size()), "malloc dump k");
            cuda_check(cudaMalloc(&dv, vb.size()), "malloc dump v");
            cudaMemcpy(dk, kb.data(), kb.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(dv, vb.data(), vb.size(), cudaMemcpyHostToDevice);
            InkKvQSlot s{};
            s.kq = L.kq; s.vq = L.vq; s.meta = L.meta;
            s.cap = n_kv; s.hp_cap = 0; s.swa = 1024; s.sinks = 0; s.local = 0; s.is_swa = 0;
            s.fmt = INK_KVQ_K8V4; s.krow = 1024; s.vrow = 512; s.mrow = 96;
            dim3 qg((unsigned)((n_kv + 7) / 8), hkv);
            ink_f3d_quant_from_dump<<<qg, 128>>>(dk, dv, s, F3D_SRC_N);
            cuda_check(cudaGetLastError(), "quant launch");
            cuda_check(cudaDeviceSynchronize(), "quant sync");
            cudaFree(dk);
            cudaFree(dv);

            std::vector<float> hq((const float*)qb.data(), (const float*)qb.data() + nh * hd);
            for (float& x : hq) x *= tau;
            cudaMemcpy(L.q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
            const float* bd = (const float*)bb.data();
            std::vector<float> table(nh * b_s2, 0.f);
            const int pos4 = F3D_SRC_N - 1;
            for (int h = 0; h < nh; ++h) {
                for (int rp = 0; rp < b_s2; ++rp) {
                    const int j = pos4 - rp;
                    table[h * b_s2 + rp] = bd[h * F3D_SRC_N + j] * tau;
                }
            }
            cudaMemcpy(L.bias, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
        }
        L.slot.kq = L.kq;
        L.slot.vq = L.vq;
        L.slot.meta = L.meta;
        L.slot.cap = n_kv;
        L.slot.hp_cap = 0;
        L.slot.swa = 1024;
        L.slot.fmt = INK_KVQ_K8V4;
        L.slot.krow = 1024;
        L.slot.vrow = 512;
        L.slot.mrow = 96;
    }

    float *part = nullptr, *out = nullptr, *mask = nullptr, *qscales = nullptr;
    unsigned char* qbuf = nullptr;
    int *dpos = nullptr, *dnkv = nullptr;
    cuda_check(cudaMalloc(&part, (size_t)4096 * nh * (hd + 2) * 4), "malloc part");
    cuda_check(cudaMalloc(&out, nh * hd * 4), "malloc out");
    cuda_check(cudaMalloc(&mask, (n_kv + 1) * 4), "malloc mask");
    cudaMemset(mask, 0, (n_kv + 1) * 4);
    cuda_check(cudaMalloc(&qbuf, 8192), "malloc qbuf");
    qscales = (float*)(qbuf + 7168);
    cuda_check(cudaMalloc(&dpos, 4), "malloc pos");
    cuda_check(cudaMalloc(&dnkv, 4), "malloc nkv");
    const int hpos = n_kv - 1;
    cudaMemcpy(dpos, &hpos, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dnkv, &n_kv, 4, cudaMemcpyHostToDevice);

    const int max_chunks = (n_kv + chunk_len - 1) / chunk_len;
    const dim3 gp_split((unsigned)(max_chunks * 3), hkv);
    const dim3 gp_gqa((unsigned)max_chunks, hkv);
    const size_t smw = (3 * 128 + 2 * 3) * 4;

    auto quant_layer = [&](int li) {
        inkwell_kvq_quantq<<<dim3((unsigned)nh), 32>>>(layers[li].q, q_s2, nh, qbuf, 2);
    };
    auto launch_base = [&](int li) {
        inkwell_attn_kvq_splitw<INK_KVQ_K8V4, false, 2><<<gp_split, dim3(64), smw>>>(
            layers[li].slot, dpos, layers[li].q, layers[li].bias, mask, part,
            nh, dnkv, q_s2, b_s2, scale, chunk_len, 1, 3, 1,
            0, 1024, 1024, qbuf, qscales);
    };
    auto launch_a64 = [&](int li) { launch_rowv2<4>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_a128 = [&](int li) { launch_rowv2<8>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_a192 = [&](int li) { launch_rowv2<12>(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto launch_b96 = [&](int li) { launch_smem3v2(layers[li], gp_gqa, chunk_len, dpos, mask, part, dnkv, nh, q_s2, b_s2, scale, qbuf, qscales); };
    auto reduce = [&]() {
        inkwell_attn_decode_reduce_dyn<<<dim3((unsigned)nh), 128>>>(part, mask, out, nh, dnkv, hd, chunk_len);
    };

    std::vector<std::function<void(int)>> ilaunch = {launch_base, launch_a64, launch_a128, launch_a192, launch_b96};
    const char* inames[5] = {"splitw", "row64", "row128", "row192", "smem96"};
    double max_abs[5] = {}, sumsq[5] = {}, refsq = 0.0;
    std::vector<float> hnew(nh * hd);
    std::vector<std::vector<float>> refs(F3D_NLAYER);
    for (int li = 0; li < F3D_NLAYER; ++li) {
        quant_layer(li);
        launch_base(li);
        reduce();
        cuda_check(cudaDeviceSynchronize(), "baseline correctness sync");
        refs[li].resize(nh * hd);
        cudaMemcpy(refs[li].data(), out, nh * hd * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < nh * hd; ++i) refsq += (double)refs[li][i] * refs[li][i];
        for (int v = 1; v < 5; ++v) {
            ilaunch[v](li);
            reduce();
            cuda_check(cudaDeviceSynchronize(), "candidate correctness sync");
            cudaMemcpy(hnew.data(), out, nh * hd * 4, cudaMemcpyDeviceToHost);
            for (int i = 0; i < nh * hd; ++i) {
                const double df = (double)hnew[i] - refs[li][i];
                max_abs[v] = std::max(max_abs[v], fabs(df));
                sumsq[v] += df * df;
            }
            if (li == 0 && v == 1 && getenv("F3D_DEBUG")) {
                int mi = 0;
                for (int i = 1; i < nh * hd; ++i) if (fabsf(hnew[i] - refs[li][i]) > fabsf(hnew[mi] - refs[li][mi])) mi = i;
                printf("debug maxi=%d h=%d d=%d ref=%.8g new=%.8g\n", mi, mi / hd, mi % hd, refs[li][mi], hnew[mi]);
                for (int h = 0; h < 3; ++h) {
                    printf("debug h%d ref:", h);
                    for (int d = 0; d < 8; ++d) printf(" %.6g", refs[li][h * hd + d]);
                    printf(" new:");
                    for (int d = 0; d < 8; ++d) printf(" %.6g", hnew[h * hd + d]);
                    printf("\n");
                }
            }
        }
    }
    cuda_check(cudaGetLastError(), "correctness launch");

    std::vector<std::function<void()>> bench = {
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_base(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_a64(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_a128(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_a192(li); },
        [&] { for (int li = 0; li < F3D_NLAYER; ++li) launch_b96(li); },
    };
    for (auto& fn : bench) fn();
    cudaDeviceSynchronize();
    float bms[5] = {};
    for (int round = 0; round < 2; ++round) {
        for (int v = 0; v < 5; ++v) {
            const float t = time_kernel(bench, v, reps);
            bms[v] = bms[v] == 0.f ? t : std::min(bms[v], t);
        }
    }

    printf("mode=%s chunk=%d layers=%d nkv=%d\n", mode, chunk_len, F3D_NLAYER, n_kv);
    for (int v = 0; v < 5; ++v) {
        const double rel = v ? sqrt(sumsq[v] / std::max(1e-30, refsq)) : 0.0;
        printf("%s ms=%.3f maxabs=%.3g relL2=%.3g speedup=%.3fx\n", inames[v], bms[v], max_abs[v], rel, bms[0] / bms[v]);
    }
    bool ok = true;
    for (int v = 1; v < 5; ++v) ok = ok && max_abs[v] <= 1e-5;
    return ok ? 0 : 1;
}
