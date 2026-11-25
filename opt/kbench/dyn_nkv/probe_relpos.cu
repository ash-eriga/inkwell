#include "kernels.cu"
extern "C" const unsigned char ink_ch2nopless_cubin_start[1] = {0};
extern "C" const unsigned char ink_ch2nopless_cubin_end[1] = {0};
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static float frand(uint32_t& s, float scale) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return ((float)(s >> 8) * (1.0f / 8388608.0f) - 1.0f) * scale;
}

int main() {
    setbuf(stdout, nullptr);
    const int nh = 24, cap = 4096, ext = 4104, rel_extent = 4095, swa = 1024;
    std::vector<float> table((size_t)nh * ext), sinks(nh), zeros(cap, 0.0f);
    uint32_t rs = 0x51f15eedu;
    for (auto& x : table) x = frand(rs, 0.071f);
    for (auto& x : sinks) x = frand(rs, 0.031f);
    std::vector<int> rp(cap, 0);
    float *TB, *SN, *RP, *YA, *YB;
    int *DP, *DN;
    cudaMalloc(&TB, table.size() * 4); cudaMalloc(&SN, sinks.size() * 4);
    cudaMalloc(&RP, rp.size() * 4); cudaMalloc(&YA, (size_t)nh * cap * 4); cudaMalloc(&YB, (size_t)nh * cap * 4);
    cudaMalloc(&DP, 4); cudaMalloc(&DN, 4);
    cudaMemcpy(TB, table.data(), table.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(SN, sinks.data(), sinks.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t st; cudaStreamCreate(&st);
    printf("mode pos nkv bitdiffs err\n");
    int bad = 0;
    auto run = [&](int mode, int pos) {
        const int nkv = mode == 0 ? pos + 1 : std::min(pos + 1, swa);
        for (int j = 0; j < nkv; ++j) {
            int x;
            if (mode == 1) {
                x = (pos - j) % swa;
                if (x < 0) x += swa;
            } else if (mode == 2) {
                x = nkv - 1 - j;
            } else {
                x = pos - j;
            }
            rp[j] = std::min(x, rel_extent);
        }
        for (int j = nkv; j < cap; ++j) rp[j] = 0;
        cudaMemcpy(RP, rp.data(), rp.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(DP, &pos, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(DN, &nkv, 4, cudaMemcpyHostToDevice);
        ink_launch_relbias_gather_dyn(TB, SN, RP, DN, YA, cap, nh, ext, st);
        ink_launch_relbias_gather_dynpos(TB, SN, DP, DN, YB, cap, nh, ext, mode, swa, rel_extent, st);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaStreamSynchronize(st);
        int diffs = -1;
        if (err == cudaSuccess) {
            std::vector<float> a((size_t)nh * cap), b((size_t)nh * cap);
            cudaMemcpy(a.data(), YA, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), YB, b.size() * 4, cudaMemcpyDeviceToHost);
            diffs = 0;
            for (int h = 0; h < nh; ++h) for (int j = 0; j < nkv; ++j) {
                uint32_t x, y;
                const size_t i = (size_t)h * cap + j;
                memcpy(&x, &a[i], 4); memcpy(&y, &b[i], 4);
                diffs += x != y;
            }
        }
        printf("%d %d %d %d %s\n", mode, pos, nkv, diffs, cudaGetErrorString(err));
        bad += diffs != 0 || err != cudaSuccess;
    };
    for (int pos : {0, 62, 63, 64, 318, 499, 1023, 4095}) run(0, pos);
    for (int pos : {5, 63, 1023, 1024, 1025, 2047, 4095}) run(1, pos);
    for (int pos : {5, 63, 1023, 1024, 1025, 2047, 4095}) run(2, pos);
    cudaStreamDestroy(st);
    cudaFree(TB); cudaFree(SN); cudaFree(RP); cudaFree(YA); cudaFree(YB); cudaFree(DP); cudaFree(DN);
    return bad ? 1 : 0;
}
