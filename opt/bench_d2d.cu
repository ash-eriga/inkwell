// device-to-device copy bandwidth ceiling for sm_61 (pure C, no C++ stdlib)
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

int main() {
    const size_t bytes = 512u << 20;
    void *a, *b;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMemset(a, 1, bytes);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice); // warmup
    cudaDeviceSynchronize();
    float best_ms = 1e9f;
    for (int i = 0; i < 10; ++i) {
        cudaEventRecord(e0);
        cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
        printf("d2d: %.1f GB/s\n", 2.0 * bytes / ms / 1e6);
    }
    printf("best: %.1f GB/s\n", 2.0 * bytes / best_ms / 1e6);
    return 0;
}
