#include <cstdint>
template<int MODE> __global__ void __launch_bounds__(32) khi(const uint32_t* in, float* out, int n) {
    uint32_t acc=0;
    for (int i=threadIdx.x;i<n;i+=32) {
        uint32_t u=in[i];
        uint32_t t=(u>>16)*0xd12du;
        uint32_t p=u*0x83dcu+t;
        acc += __dp4a(p,0x01010101u,0x4b000000u);
    }
    if (acc==12345) out[0]=acc;
}
template __global__ void khi<0>(const uint32_t*, float*, int);
