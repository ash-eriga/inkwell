//! Rust bindings for the ggml shim: build ggml graphs (all host-side
//! orchestration in Rust) and run them on the CUDA backend.
#![allow(non_snake_case)]
#![expect(missing_docs, reason = "extern decls mirror ggml.h signatures")]
#![expect(clippy::missing_safety_doc, reason = "raw FFI surface; safe wrappers come with the runner")]

use std::ffi::{c_char, c_int, c_void};

/// ggml type ids matching `ggml_type`.
pub const F32: i32 = 0;
pub const F16: i32 = 1;
pub const Q8_0: i32 = 8;
pub const Q6_K: i32 = 14;
pub const IQ2_XXS: i32 = 16;

pub extern "C" {
    #[allow(dead_code)]
    fn ink_ctx_new() -> *mut c_void;
    fn ink_ctx_free(c: *mut c_void);
    fn ink_tensor(c: *mut c_void, ty: c_int, nd: c_int, ne: *const i64) -> *mut c_void;
    fn ink_view_1d(c: *mut c_void, a: *mut c_void, n: i64, off: i64) -> *mut c_void;
    fn ink_view_2d(c: *mut c_void, a: *mut c_void, n0: i64, n1: i64, rs: i64, off: i64) -> *mut c_void;
    fn ink_view_3d(c: *mut c_void, a: *mut c_void, n0: i64, n1: i64, n2: i64, s1: i64, s2: i64, off: i64) -> *mut c_void;
    fn ink_set_name(t: *mut c_void, name: *const c_char) -> *mut c_void;
    fn ink_set_output(t: *mut c_void);
    fn ink_mul_mat(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_mul(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_add(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_sub(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_div(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_rms_norm(c: *mut c_void, a: *mut c_void, eps: f32) -> *mut c_void;
    fn ink_scale(c: *mut c_void, a: *mut c_void, s: f32) -> *mut c_void;
    fn ink_scale_bias(c: *mut c_void, a: *mut c_void, s: f32, b: f32) -> *mut c_void;
    fn ink_sigmoid(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_tanh(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_sqr(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_sqrt(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_log(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_clamp(c: *mut c_void, a: *mut c_void, lo: f32, hi: f32) -> *mut c_void;
    fn ink_silu(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_gelu(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_ssm_conv(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_get_rows(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_soft_max_ext(c: *mut c_void, a: *mut c_void, mask: *mut c_void, scale: f32, mb: f32) -> *mut c_void;
    fn ink_concat(c: *mut c_void, a: *mut c_void, b: *mut c_void, dim: c_int) -> *mut c_void;
    fn ink_cont(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_permute(c: *mut c_void, a: *mut c_void, a0: c_int, a1: c_int, a2: c_int, a3: c_int) -> *mut c_void;
    fn ink_reshape_2d(c: *mut c_void, a: *mut c_void, n0: i64, n1: i64) -> *mut c_void;
    fn ink_reshape_3d(c: *mut c_void, a: *mut c_void, n0: i64, n1: i64, n2: i64) -> *mut c_void;
    fn ink_reshape_4d(c: *mut c_void, a: *mut c_void, n0: i64, n1: i64, n2: i64, n3: i64) -> *mut c_void;
    fn ink_transpose(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_repeat(c: *mut c_void, a: *mut c_void, b: *mut c_void) -> *mut c_void;
    fn ink_pad(c: *mut c_void, a: *mut c_void, p0: c_int, p1: c_int, p2: c_int, p3: c_int) -> *mut c_void;
    fn ink_fill(c: *mut c_void, a: *mut c_void, v: f32) -> *mut c_void;
    fn ink_sum_rows(c: *mut c_void, a: *mut c_void) -> *mut c_void;
    fn ink_cast(c: *mut c_void, a: *mut c_void, ty: c_int) -> *mut c_void;
    fn ink_flash_attn_ext(c: *mut c_void, q: *mut c_void, k: *mut c_void, v: *mut c_void, mask: *mut c_void, scale: f32, mb: f32, lsc: f32) -> *mut c_void;
    fn ink_flash_attn_ext_add_sinks(t: *mut c_void, sinks: *mut c_void);
    fn ink_flash_attn_ext_set_n_kv_max(t: *mut c_void, n: i32);
    fn ink_backend_cuda(dev: c_int) -> *mut c_void;
    fn ink_cuda_buft(dev: c_int) -> *mut c_void;
    fn ink_cpu_buft() -> *mut c_void;
    fn ink_cuda_host_buft() -> *mut c_void;
    fn ink_buf_alloc(buft: *mut c_void, bytes: usize) -> *mut c_void;
    fn ink_buf_base(b: *mut c_void) -> *mut c_void;
    fn ink_buf_set(b: *mut c_void, off: usize, src: *const c_void, n: usize);
    fn ink_tensor_set(t: *mut c_void, src: *const c_void, off: usize, n: usize);
    fn ink_tensor_get(t: *mut c_void, dst: *mut c_void, off: usize, n: usize);
    fn ink_tensor_assign(t: *mut c_void, buf: *mut c_void, off: usize);
    fn ink_gallocr(buft: *mut c_void) -> *mut c_void;
    fn ink_graph_new(c: *mut c_void) -> *mut c_void;
    fn ink_build_forward(g: *mut c_void, out: *mut c_void);
    fn ink_alloc_graph(galloc: *mut c_void, g: *mut c_void) -> bool;
    fn ink_compute(backend: *mut c_void, g: *mut c_void) -> c_int;
    fn ink_backend_free(b: *mut c_void);
    fn ink_tensor_nbytes(t: *mut c_void) -> usize;
    fn ink_ne(t: *mut c_void, i: c_int) -> i64;
    fn ink_nb(t: *mut c_void, i: c_int) -> i64;
}

/// Probe: create a no-alloc ctx + cuda backend, compute a tiny
/// rms_norm graph on device. Returns output row or empty on failure.
#[must_use]
pub fn smoke() -> Vec<f32> {
    unsafe {
        let ctx = ink_ctx_new();
        let backend = ink_backend_cuda(0);
        assert!(!backend.is_null(), "cuda backend");
        // weights buffer on device
        let buft = ink_cuda_buft(0);
        let buf = ink_buf_alloc(buft, 4096 * 4 * 2);
        assert!(!buf.is_null(), "buf alloc");
        let x_t = ink_tensor(ctx, F32, 1, [4096i64].as_ptr());
        let w_t = ink_tensor(ctx, F32, 1, [4096i64].as_ptr());
        ink_tensor_assign(x_t, buf, 0);
        ink_tensor_assign(w_t, buf, 4096 * 4);
        let host_x: Vec<f32> = (0..4096).map(|i| i as f32 * 0.5 - 1000.0).collect();
        let host_w: Vec<f32> = vec![2.0; 4096];
        ink_tensor_set(x_t, host_x.as_ptr().cast(), 0, host_x.len() * 4);
        ink_tensor_set(w_t, host_w.as_ptr().cast(), 0, host_w.len() * 4);
        let y = ink_rms_norm(ctx, x_t, 1e-6);
        let out = ink_mul(ctx, y, w_t);
        ink_set_output(out);
        let g = ink_graph_new(ctx);
        ink_build_forward(g, out);
        let galloc = ink_gallocr(buft);
        assert!(ink_alloc_graph(galloc, g), "alloc graph");
        assert_eq!(ink_compute(backend, g), 0);
        let mut v = vec![0.0f32; 4096];
        ink_tensor_get(out, v.as_mut_ptr().cast(), 0, v.len() * 4);
        ink_backend_free(backend);
        ink_ctx_free(ctx);
        v
    }
}
