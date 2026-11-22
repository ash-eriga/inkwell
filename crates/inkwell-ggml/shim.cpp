// C shim over ggml + ggml-cuda: tensor/graph construction for the
// Rust-side host engine. Weight data stays in ggml cuda buffers.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>

extern "C" {

ggml_context* ink_ctx_new() {
    ggml_init_params p = { GGML_DEFAULT_N_THREADS*0 + 1u<<20, nullptr, true };
    p.mem_size = (size_t)64 << 20;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    return ggml_init(p);
}
void ink_ctx_free(ggml_context* c) { ggml_free(c); }

ggml_tensor* ink_tensor(ggml_context* c, int type, int nd, const int64_t* ne) {
    switch (nd) {
        case 1: return ggml_new_tensor_1d(c, (ggml_type)type, ne[0]);
        case 2: return ggml_new_tensor_2d(c, (ggml_type)type, ne[0], ne[1]);
        case 3: return ggml_new_tensor_3d(c, (ggml_type)type, ne[0], ne[1], ne[2]);
        default: return ggml_new_tensor_4d(c, (ggml_type)type, ne[0], ne[1], ne[2], ne[3]);
    }
}
ggml_tensor* ink_view_1d(ggml_context* c, ggml_tensor* a, int64_t n, int64_t off) {
    return ggml_view_1d(c, a, n, off);
}
ggml_tensor* ink_view_2d(ggml_context* c, ggml_tensor* a, int64_t n0, int64_t n1,
                         int64_t row_stride, int64_t off) {
    return ggml_view_2d(c, a, n0, n1, row_stride, off);
}
ggml_tensor* ink_view_3d(ggml_context* c, ggml_tensor* a, int64_t n0, int64_t n1, int64_t n2,
                         int64_t s1, int64_t s2, int64_t off) {
    return ggml_view_3d(c, a, n0, n1, n2, s1, s2, off);
}

ggml_tensor* ink_view_4d(ggml_context* c, ggml_tensor* a, int64_t n0, int64_t n1, int64_t n2, int64_t n3,
                         int64_t s1, int64_t s2, int64_t s3, int64_t off) {
    return ggml_view_4d(c, a, n0, n1, n2, n3, s1, s2, s3, off);
}
ggml_tensor* ink_set_name(ggml_tensor* t, const char* name) { ggml_set_name(t, name); return t; }
void ink_set_output(ggml_tensor* t) { ggml_set_output(t); }
void ink_set_input(ggml_tensor* t) { t->flags |= GGML_TENSOR_FLAG_INPUT; }

// ---- ops ----
ggml_tensor* ink_mul_mat(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_mul_mat(c, a, b); }
ggml_tensor* ink_mul(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_mul(c, a, b); }
ggml_tensor* ink_add(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_add(c, a, b); }
ggml_tensor* ink_sub(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_sub(c, a, b); }
ggml_tensor* ink_div(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_div(c, a, b); }
ggml_tensor* ink_rms_norm(ggml_context* c, ggml_tensor* a, float eps) { return ggml_rms_norm(c, a, eps); }
ggml_tensor* ink_scale(ggml_context* c, ggml_tensor* a, float s) { return ggml_scale(c, a, s); }
ggml_tensor* ink_scale_bias(ggml_context* c, ggml_tensor* a, float s, float b) { return ggml_scale_bias(c, a, s, b); }
ggml_tensor* ink_sigmoid(ggml_context* c, ggml_tensor* a) { return ggml_sigmoid(c, a); }
ggml_tensor* ink_tanh(ggml_context* c, ggml_tensor* a) { return ggml_tanh(c, a); }
ggml_tensor* ink_sqr(ggml_context* c, ggml_tensor* a) { return ggml_sqr(c, a); }
ggml_tensor* ink_sqrt(ggml_context* c, ggml_tensor* a) { return ggml_sqrt(c, a); }
ggml_tensor* ink_log(ggml_context* c, ggml_tensor* a) { return ggml_log(c, a); }
ggml_tensor* ink_clamp(ggml_context* c, ggml_tensor* a, float lo, float hi) { return ggml_clamp(c, a, lo, hi); }
ggml_tensor* ink_silu(ggml_context* c, ggml_tensor* a) { return ggml_silu(c, a); }
ggml_tensor* ink_gelu(ggml_context* c, ggml_tensor* a) { return ggml_gelu(c, a); }
ggml_tensor* ink_ssm_conv(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_ssm_conv(c, a, b); }
ggml_tensor* ink_get_rows(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_get_rows(c, a, b); }
ggml_tensor* ink_soft_max_ext(ggml_context* c, ggml_tensor* a, ggml_tensor* mask, float scale, float max_bias) {
    return ggml_soft_max_ext(c, a, mask, scale, max_bias);
}
ggml_tensor* ink_concat(ggml_context* c, ggml_tensor* a, ggml_tensor* b, int dim) { return ggml_concat(c, a, b, dim); }
ggml_tensor* ink_cont(ggml_context* c, ggml_tensor* a) { return ggml_cont(c, a); }
ggml_tensor* ink_permute(ggml_context* c, ggml_tensor* a, int a0, int a1, int a2, int a3) { return ggml_permute(c, a, a0, a1, a2, a3); }
ggml_tensor* ink_reshape_2d(ggml_context* c, ggml_tensor* a, int64_t n0, int64_t n1) { return ggml_reshape_2d(c, a, n0, n1); }
ggml_tensor* ink_reshape_3d(ggml_context* c, ggml_tensor* a, int64_t n0, int64_t n1, int64_t n2) { return ggml_reshape_3d(c, a, n0, n1, n2); }
ggml_tensor* ink_reshape_4d(ggml_context* c, ggml_tensor* a, int64_t n0, int64_t n1, int64_t n2, int64_t n3) { return ggml_reshape_4d(c, a, n0, n1, n2, n3); }
ggml_tensor* ink_transpose(ggml_context* c, ggml_tensor* a) { return ggml_transpose(c, a); }
ggml_tensor* ink_repeat(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_repeat(c, a, b); }
ggml_tensor* ink_pad(ggml_context* c, ggml_tensor* a, int p0, int p1, int p2, int p3) { return ggml_pad(c, a, p0, p1, p2, p3); }
ggml_tensor* ink_fill(ggml_context* c, ggml_tensor* a, float v) { return ggml_fill(c, a, v); }
ggml_tensor* ink_sum_rows(ggml_context* c, ggml_tensor* a) { return ggml_sum_rows(c, a); }
ggml_tensor* ink_cast(ggml_context* c, ggml_tensor* a, int type) { return ggml_cast(c, a, (ggml_type)type); }
ggml_tensor* ink_flash_attn_ext(ggml_context* c, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float scale, float max_bias, float logit_softcap) {
    return ggml_flash_attn_ext(c, q, k, v, mask, scale, max_bias, logit_softcap);
}
void ink_flash_attn_ext_add_sinks(ggml_tensor* t, ggml_tensor* sinks) { ggml_flash_attn_ext_add_sinks(t, sinks); }
void ink_flash_attn_ext_set_n_kv_max(ggml_tensor* t, int32_t n) { ggml_flash_attn_ext_set_n_kv_max(t, n); }

ggml_tensor* ink_mul_mat_id(ggml_context* c, ggml_tensor* as, ggml_tensor* b, ggml_tensor* ids) { return ggml_mul_mat_id(c, as, b, ids); }
ggml_tensor* ink_argsort_top_k(ggml_context* c, ggml_tensor* a, int k) { return ggml_argsort_top_k(c, a, k); }
ggml_tensor* ink_set_rows(ggml_context* c, ggml_tensor* a, ggml_tensor* b, ggml_tensor* idx) { return ggml_set_rows(c, a, b, idx); }
ggml_tensor* ink_cpy(ggml_context* c, ggml_tensor* a, ggml_tensor* b) { return ggml_cpy(c, a, b); }
ggml_tensor* ink_dup(ggml_context* c, ggml_tensor* a) { return ggml_dup(c, a); }
ggml_tensor* ink_softplus(ggml_context* c, ggml_tensor* a) { return ggml_softplus(c, a); }
ggml_tensor* ink_gelu_quick(ggml_context* c, ggml_tensor* a) { return ggml_gelu_quick(c, a); }
ggml_tensor* ink_add_id(ggml_context* c, ggml_tensor* a, ggml_tensor* b, ggml_tensor* ids) { return ggml_add_id(c, a, b, ids); }

// ---- backend / buffers / graph compute ----
ggml_backend_t ink_backend_cuda(int dev) { return ggml_backend_cuda_init(dev); }
ggml_backend_buffer_type_t ink_cuda_buft(int dev) { return ggml_backend_cuda_buffer_type(dev); }
ggml_backend_buffer_type_t ink_cpu_buft() { return ggml_backend_cpu_buffer_type(); }
ggml_backend_buffer_type_t ink_cuda_host_buft() { return ggml_backend_cuda_host_buffer_type(); }
ggml_backend_buffer_t ink_buf_alloc(ggml_backend_buffer_type_t buft, size_t bytes) {
    return ggml_backend_buft_alloc_buffer(buft, bytes);
}
void* ink_buf_base(ggml_backend_buffer_t b) { return ggml_backend_buffer_get_base(b); }
int ink_tensor_debug(ggml_tensor* t) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    fprintf(stderr, "TDBG op=%d view_src=%p buf=%p data=%p flags=%d\n",
        (int)t->op, (void*)t->view_src, (void*)buf, t->data, (int)t->flags);
    return (int)t->op;
}
void ink_buf_set_usage(ggml_backend_buffer_t b, int usage) { ggml_backend_buffer_set_usage(b, (enum ggml_backend_buffer_usage)usage); }
void ink_buf_set(ggml_backend_buffer_t b, size_t off, const void* src, size_t n) {
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // write via tensor-less path: ggml buffers don't expose set; use cudaMemcpy via tensor set instead
    memcpy((char*)ggml_backend_buffer_get_base(b) + off, src, n); // host bufs only
}
void ink_tensor_set(ggml_tensor* t, const void* src, size_t off, size_t n) {
    ggml_backend_tensor_set(t, src, off, n);
}
void ink_tensor_get(ggml_tensor* t, void* dst, size_t off, size_t n) {
    ggml_backend_tensor_get(t, dst, off, n);
}
void ink_tensor_assign(ggml_tensor* t, ggml_backend_buffer_t buf, size_t off) {
    t->buffer = buf;
    t->data = (char*)ggml_backend_buffer_get_base(buf) + off;
}
ggml_gallocr_t ink_gallocr(ggml_backend_buffer_type_t buft) { return ggml_gallocr_new(buft); }
ggml_cgraph* ink_graph_new(ggml_context* c, int64_t size) { return ggml_new_graph_custom(c, size, false); }
void ink_build_forward(ggml_cgraph* g, ggml_tensor* out) { ggml_build_forward_expand(g, out); }
bool ink_alloc_graph(ggml_gallocr_t galloc, ggml_cgraph* g) { return ggml_gallocr_alloc_graph(galloc, g); }
int ink_compute(ggml_backend_t backend, ggml_cgraph* g) { return (int)ggml_backend_graph_compute(backend, g); }
void ink_backend_free(ggml_backend_t b) { ggml_backend_free(b); }
size_t ink_tensor_nbytes(ggml_tensor* t) { return ggml_nbytes(t); }
int64_t ink_ne(ggml_tensor* t, int i) { return t->ne[i]; }
int64_t ink_nb(ggml_tensor* t, int i) { return t->nb[i]; }

// ---- scheduler (cuda + cpu split for expert offload) ----
ggml_backend_t ink_backend_cpu(int threads) {
    ggml_backend_t b = ggml_backend_cpu_init();
    if (threads > 0) ggml_backend_cpu_set_n_threads(b, threads);
    return b;
}
ggml_backend_sched_t ink_sched_new(ggml_backend_t* backends, ggml_backend_buffer_type_t* bufts, int n, size_t graph_size) {
    return ggml_backend_sched_new(backends, bufts, n, graph_size, false, true);
}
void ink_sched_set_tensor_backend(ggml_backend_sched_t s, ggml_tensor* t, ggml_backend_t b) {
    ggml_backend_sched_set_tensor_backend(s, t, b);
}
void ink_sched_reset(ggml_backend_sched_t s) { ggml_backend_sched_reset(s); }
int ink_sched_compute(ggml_backend_sched_t s, ggml_cgraph* g) {
    return (int)ggml_backend_sched_graph_compute(s, g);
}
ggml_backend_buffer_t ink_alloc_ctx_tensors_buft(ggml_context* c, ggml_backend_buffer_type_t buft) {
    return ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
}
ggml_backend_buffer_t ink_alloc_ctx_tensors(ggml_context* c, ggml_backend_t b) {
    return ggml_backend_alloc_ctx_tensors(c, b);
}
// dequantize `nelems` of `type` from src into dst (f32); returns 0 ok, -1 no traits
int ink_dequant(int type, const void* src, float* dst, int64_t nelems) {
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type)type);
    if (!tt || !tt->to_float) return -1;
    tt->to_float(src, dst, nelems);
    return 0;
}
} // extern "C"
