#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Hardware __dp4a (IDP4A) is present on Pascal GP102/104/106/107 (sm_61)
// and Volta+ (sm_70+; sm_75+ also carry INT8 Tensor Cores).
// Maxwell (sm_5x), GP100 (sm_60), and Tegra X2 (sm_62) lack native dp4a.
static inline bool ink_gpu_has_dp4a() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0, maj = 0, min = 0;
        if (cudaGetDevice(&dev) == cudaSuccess
            && cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess
            && cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess) {
            const int sm = maj * 10 + min;
            cached = (sm == 61 || sm >= 70) ? 1 : 0;
        } else {
            cached = 0;
        }
    }
    return cached != 0;
}

static inline bool ink_pf_int8_enabled() {
    const char* q8e = getenv("INKWELL_PF_INT8");
    const bool has_dp4a = ink_gpu_has_dp4a();
    if (q8e == nullptr || q8e[0] == '\0') {
        return has_dp4a;
    }
    if (q8e[0] == '0') {
        return false;
    }
    if (!has_dp4a) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "inkwell: INKWELL_PF_INT8=%s requested on GPU without hardware dp4a; falling back to FP32\n", q8e);
        }
        return false;
    }
    return q8e[0] == '1';
}

static inline int ink_kvq_dp4a_mode(bool fmt_supports_dp4a) {
    if (!fmt_supports_dp4a) return 0;
    const char* dp4a_env = getenv("INKWELL_KVQ_DP4A");
    const bool has_dp4a = ink_gpu_has_dp4a();
    if (dp4a_env == nullptr || dp4a_env[0] == '\0') {
        return has_dp4a ? 2 : 0;
    }
    const int req = atoi(dp4a_env);
    if (req <= 0) return 0;
    if (!has_dp4a) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "inkwell: INKWELL_KVQ_DP4A=%s requested on GPU without hardware dp4a; falling back to FP32\n", dp4a_env);
        }
        return 0;
    }
    return req > 4 ? 4 : req;
}
