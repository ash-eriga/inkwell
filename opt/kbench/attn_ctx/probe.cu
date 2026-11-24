// Context-length probe for inkwell_attn_decode (the shipped decode attention).
// Checks whether the launch succeeds at each n_kv and times one global-attention
// visit (24 q heads, 8 kv heads, hd 128, F16 K/V) with CUDA events.
#include "kernels.cu"
#include <vector>
#include <algorithm>

int main() {
    const int nh = 24, hkv = 8, hd = 128;
    const int nkvs[] = {1024, 4096, 8192, 12160, 12224, 16384, 32768};
    for (int n_kv : nkvs) {
        size_t kvb = (size_t)hkv * n_kv * hd * 2;
        void *K, *V; float *Q, *B, *Mk, *O;
        cudaMalloc(&K, kvb); cudaMalloc(&V, kvb);
        cudaMalloc(&Q, nh * hd * 4); cudaMalloc(&B, (size_t)nh * n_kv * 4);
        cudaMalloc(&Mk, (n_kv + 1) * 4); cudaMalloc(&O, nh * hd * 4);
        cudaMemset(K, 0, kvb); cudaMemset(V, 0, kvb); cudaMemset(Q, 0, nh * hd * 4);
        cudaMemset(B, 0, (size_t)nh * n_kv * 4); cudaMemset(Mk, 0, (n_kv + 1) * 4);
        cudaGetLastError();
        cudaStream_t st; cudaStreamCreate(&st);
        auto launch = [&] {
            ink_launch_attn_decode(K, V, Q, B, Mk, O, nh, hkv, n_kv, hd,
                hd * 2, (long long)n_kv * hd * 2, hd * 2, (long long)n_kv * hd * 2,
                hd, n_kv, 1.0f / sqrtf((float)hd), 1, st);
        };
        launch();
        cudaError_t le = cudaGetLastError();
        cudaError_t se = cudaStreamSynchronize(st);
        std::vector<float> ms;
        if (le == cudaSuccess && se == cudaSuccess) {
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            for (int r = 0; r < 15; ++r) {
                cudaEventRecord(a, st); launch(); cudaEventRecord(b, st); cudaEventSynchronize(b);
                float t; cudaEventElapsedTime(&t, a, b); ms.push_back(t);
            }
            std::sort(ms.begin(), ms.end());
        }
        printf("n_kv=%6d smem=%6zu B launch=%s sync=%s median=%.3f ms/visit\n", n_kv,
               (size_t)(n_kv + hd) * 4, cudaGetErrorString(le), cudaGetErrorString(se),
               ms.empty() ? -1.0f : ms[ms.size() / 2]);
        cudaFree(K); cudaFree(V); cudaFree(Q); cudaFree(B); cudaFree(Mk); cudaFree(O);
        cudaStreamDestroy(st);
    }
    return 0;
}
