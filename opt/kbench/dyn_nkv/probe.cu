#include "kernels.cu"
extern "C" const unsigned char ink_ch2nopless_cubin_start[1] = {0};
extern "C" const unsigned char ink_ch2nopless_cubin_end[1] = {0};
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float frand(uint32_t& s, float scale) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return ((float)(s >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale;
}
static int bits_diff(const std::vector<float>& a, const std::vector<float>& b) {
    int d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t x, y; memcpy(&x, &a[i], 4); memcpy(&y, &b[i], 4); d += x != y;
    }
    return d;
}
static int bucket64(int n) { return (n + 63) & ~63; }

int main() {
    setbuf(stdout, nullptr);
    const int nh = 24, hkv = 8, hd = 128, cap = 4096;
    const int cases[] = {1, 63, 64, 65, 200, 319, 320, 500, 1024, 4096};
    const float scale = 1.0f / sqrtf((float)hd);
    printf("n_kv pad regime relbias_bitdiffs attn_bitdiffs err\n");
    int bad = 0;
    for (int raw : cases) {
        const int pad = bucket64(raw);
        const size_t ks = (size_t)hkv * pad * hd, kd = (size_t)hkv * cap * hd;
        std::vector<__half> hk_s(ks), hv_s(ks), hk_d(kd), hv_d(kd);
        std::vector<float> hq(nh * hd), hs_m(pad + 1, 0.f), hd_m(cap + 1, 0.f);
        std::vector<float> rp_s(pad, 0.f), rp_d(cap, 0.f);
        std::vector<int> rpi_s(pad, 0), rpi_d(cap, 0);
        std::vector<float> table(nh * (pad + 8)), sinks(nh);
        uint32_t rs = 0x9e3779b9u + (uint32_t)raw;
        std::vector<float> val(kd);
        for (size_t i = 0; i < kd; ++i) val[i] = frand(rs, 0.37f);
        for (int h = 0; h < hkv; ++h) for (int j = 0; j < cap; ++j) for (int d = 0; d < hd; ++d) {
            const size_t logical = ((size_t)h * cap + j) * hd + d;
            if (j < cap) { hk_d[((size_t)h * cap + j) * hd + d] = __float2half(val[logical]); }
            if (j < pad) { hk_s[((size_t)h * pad + j) * hd + d] = __float2half(val[logical]); }
        }
        for (size_t i = 0; i < kd; ++i) val[i] = frand(rs, 0.29f);
        for (int h = 0; h < hkv; ++h) for (int j = 0; j < cap; ++j) for (int d = 0; d < hd; ++d) {
            const size_t logical = ((size_t)h * cap + j) * hd + d;
            if (j < cap) { hv_d[((size_t)h * cap + j) * hd + d] = __float2half(val[logical]); }
            if (j < pad) { hv_s[((size_t)h * pad + j) * hd + d] = __float2half(val[logical]); }
        }
        for (size_t i = 0; i < hq.size(); ++i) hq[i] = frand(rs, 0.43f);
        for (size_t i = 0; i < table.size(); ++i) table[i] = frand(rs, 0.07f);
        for (size_t i = 0; i < sinks.size(); ++i) sinks[i] = frand(rs, 0.03f);
        for (int j = raw; j < pad; ++j) hs_m[j] = -INFINITY;
        for (int j = 0; j < raw; ++j) {
            rpi_s[j] = raw - 1 - j; rpi_d[j] = raw - 1 - j;
        }
        // The dynamic input mask is all zero: padded positions are never read.
        void *KS, *VS, *KD, *VD;
        float *Q, *MS, *MD, *RPS, *RPD, *TB, *SN, *BS, *BD, *OS, *OD;
        int *DN;
        cudaMalloc(&KS, ks * 2); cudaMalloc(&VS, ks * 2);
        cudaMalloc(&KD, kd * 2); cudaMalloc(&VD, kd * 2);
        cudaMalloc(&Q, hq.size() * 4); cudaMalloc(&MS, hs_m.size() * 4); cudaMalloc(&MD, hd_m.size() * 4);
        cudaMalloc(&RPS, rpi_s.size() * 4); cudaMalloc(&RPD, rpi_d.size() * 4);
        cudaMalloc(&TB, table.size() * 4); cudaMalloc(&SN, sinks.size() * 4);
        cudaMalloc(&BS, (size_t)nh * pad * 4); cudaMalloc(&BD, (size_t)nh * cap * 4);
        cudaMalloc(&OS, nh * hd * 4); cudaMalloc(&OD, nh * hd * 4); cudaMalloc(&DN, 4);
        cudaMemcpy(KS, hk_s.data(), ks * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(VS, hv_s.data(), ks * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(KD, hk_d.data(), kd * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(VD, hv_d.data(), kd * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(Q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(MS, hs_m.data(), hs_m.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(MD, hd_m.data(), hd_m.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(RPS, rpi_s.data(), rpi_s.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(RPD, rpi_d.data(), rpi_d.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(TB, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(SN, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(DN, &raw, 4, cudaMemcpyHostToDevice);
        cudaStream_t st; cudaStreamCreate(&st);
        ink_launch_relbias_gather(TB, SN, RPS, BS, pad, nh, (int)(pad + 8), st);
        ink_launch_relbias_gather_dyn(TB, SN, RPD, DN, BD, cap, nh, (int)(pad + 8), st);
        const long long ks1 = hd * 2, ks2s = (long long)pad * hd * 2, ks2d = (long long)cap * hd * 2;
        ink_launch_attn_decode(KS, VS, Q, BS, MS, OS, nh, hkv, pad, hd,
            ks1, ks2s, ks1, ks2s, hd, pad, scale, 1, st);
        const int use_split = pad >= 320, allow_gqa = pad < 320;
        ink_launch_attn_decode_dyn(KD, VD, Q, BD, MD, OD, nh, hkv, DN, hd, cap,
            ks1, ks2d, ks1, ks2d, hd, cap, scale, 1, use_split, allow_gqa, 32, 320, st);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaStreamSynchronize(st);
        int rb = -1, ad = -1;
        if (err == cudaSuccess) {
            std::vector<float> hbs((size_t)nh * pad), hbd((size_t)nh * cap), ho(nh * hd), hn(nh * hd);
            cudaMemcpy(hbs.data(), BS, hbs.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hbd.data(), BD, hbd.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(ho.data(), OS, ho.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hn.data(), OD, hn.size() * 4, cudaMemcpyDeviceToHost);
            rb = 0;
            for (int h = 0; h < nh; ++h) for (int j = 0; j < raw; ++j) {
                uint32_t x, y;
                memcpy(&x, &hbs[(size_t)h * pad + j], 4);
                memcpy(&y, &hbd[(size_t)h * cap + j], 4);
                rb += x != y;
            }
            ad = bits_diff(ho, hn);
        }
        printf("%5d %4d %c %d %d %s\n", raw, pad, use_split ? 'S' : (allow_gqa ? 'G' : 'O'), rb, ad, cudaGetErrorString(err));
        bad += rb != 0 || ad != 0 || err != cudaSuccess;
        cudaStreamDestroy(st);
        cudaFree(KS); cudaFree(VS); cudaFree(KD); cudaFree(VD); cudaFree(Q); cudaFree(MS); cudaFree(MD);
        cudaFree(RPS); cudaFree(RPD); cudaFree(TB); cudaFree(SN); cudaFree(BS); cudaFree(BD); cudaFree(OS); cudaFree(OD); cudaFree(DN);
    }
    return bad ? 1 : 0;
}
