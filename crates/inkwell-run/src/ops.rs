//! Safe-ish wrappers over the raw `inkwell-ggml` FFI: owned context,
//! tensors, buffers, backend and graph compute.
use std::ffi::c_void;

/// Opaque ggml object pointer.
#[derive(Debug, Copy, Clone)]
pub struct Ptr(pub *mut c_void);

impl Ptr {
    /// Null check.
    #[must_use]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}



/// The whole compute context: ggml context + cuda backend + gallocr.
#[derive(Debug)]
pub struct Rt {
    ctx: Ptr,
    backend: Ptr,
    buft: Ptr,
    galloc: Ptr,
}

impl Rt {
    /// Init on CUDA device 0.
    ///
    /// # Panics
    /// If backend init fails.
    #[must_use]
    pub fn cuda() -> Self {
        unsafe {
            let ctx = Ptr(inkwell_ggml::ink_ctx_new());
            let backend = Ptr(inkwell_ggml::ink_backend_cuda(0));
            assert!(!backend.is_null(), "cuda backend init");
            let buft = Ptr(inkwell_ggml::ink_cuda_buft(0));
            let galloc = Ptr(inkwell_ggml::ink_gallocr(buft.0));
            Self { ctx, backend, buft, galloc }
        }
    }

    /// Host pinned buffer type (for experts streamed over `PCIe`).
    #[must_use]
    pub fn host_buft() -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_cuda_host_buft() })
    }

    /// Device buffer type.
    #[must_use]
    pub fn device_buft(&self) -> Ptr {
        self.buft
    }

    /// Allocate a ggml buffer of `bytes` on `buft`.
    ///
    /// # Panics
    /// On allocation failure.
    #[must_use]
    pub fn buf(&self, buft: Ptr, bytes: usize) -> Buf {
        let b = unsafe { inkwell_ggml::ink_buf_alloc(buft.0, bytes) };
        assert!(!b.is_null(), "buf alloc {bytes}");
        Buf(Ptr(b))
    }

    /// New graph.
    #[must_use]
    pub fn graph(&self) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_graph_new(self.ctx.0, 32768) })
    }

    /// Alloc + compute a graph.
    ///
    /// # Panics
    /// On alloc or compute failure.
    pub fn compute(&self, graph: Ptr) {
        unsafe {
            assert!(inkwell_ggml::ink_alloc_graph(self.galloc.0, graph.0), "alloc graph");
            assert_eq!(inkwell_ggml::ink_compute(self.backend.0, graph.0), 0, "compute");
        }
    }

    /// The ggml context.
    #[must_use]
    pub fn ctx(&self) -> Ptr {
        self.ctx
    }
}

impl Drop for Rt {
    fn drop(&mut self) {
        unsafe {
            inkwell_ggml::ink_backend_free(self.backend.0);
            inkwell_ggml::ink_ctx_free(self.ctx.0);
        }
    }
}

/// A ggml buffer (device or host-pinned).
#[derive(Debug)]
#[derive(Clone, Copy)]
pub struct Buf(pub Ptr);

impl Buf {
    /// Point `t`'s data at `self[offset..]`.
    pub fn assign(&self, t: Ptr, offset: usize) {
        unsafe { inkwell_ggml::ink_tensor_assign(t.0, self.0 .0, offset) };
    }
}

/// Tensor creation + ops, all taking `rt`.
///
/// `c` = the ggml context pointer.
#[expect(clippy::too_many_arguments)]
#[expect(clippy::many_single_char_names, reason = "ggml arg names are short by convention")]
pub mod t {
    use super::Ptr;

    /// Create tensor `ne[..]` of ggml `ty` (use ggml type ids).
    ///
    /// # Panics
    /// If `ne.len() > i32::MAX`.
    #[must_use]
    pub fn new(c: Ptr, ty: i32, ne: &[i64]) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_tensor(c.0, ty, i32::try_from(ne.len()).expect("nd"), ne.as_ptr()) })
    }

    macro_rules! op2 {
        ($name:ident, $ffi:ident) => {
            #[must_use]
            pub fn $name(c: Ptr, a: Ptr, b: Ptr) -> Ptr {
                Ptr(unsafe { inkwell_ggml::$ffi(c.0, a.0, b.0) })
            }
        };
    }
    op2!(mul_mat, ink_mul_mat);
    /// `mul_mat` fast-path: Q8_0 weights route to the dp4a i8 GEMV custom op;
    /// everything else falls back to the ggml kernel.
    #[must_use]
    pub fn mmq(c: Ptr, w: Ptr, x: Ptr) -> Ptr {
        unsafe {
            let wt = inkwell_ggml::ink_tensor_type(w.0);
            if wt == 8 || wt == 14 {
                let nrows = inkwell_ggml::ink_tensor_ne(w.0, 1);
                let mut args = [w.0, x.0];
                Ptr(inkwell_ggml::ink_custom_4d(c.0, 0, nrows, 1, 1, 1,
                    args.as_mut_ptr(), 2, crate::runner::dense_cb_addr(), std::ptr::null_mut()))
            } else {
                mul_mat(c, w, x)
            }
        }
    }
    op2!(mul, ink_mul);
    op2!(add, ink_add);
    op2!(sub, ink_sub);
    op2!(div, ink_div);
    op2!(ssm_conv, ink_ssm_conv);
    op2!(get_rows, ink_get_rows);
    op2!(repeat, ink_repeat);
    op2!(cpy, ink_cpy);
    /// concat on `dim`.
    #[must_use]
    pub fn concat(c: Ptr, a: Ptr, b: Ptr, dim: i32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_concat(c.0, a.0, b.0, dim) })
    }
    /// `mul_mat_id` expert gather.
    #[must_use]
    pub fn mul_mat_id(c: Ptr, as_: Ptr, b: Ptr, ids: Ptr) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_mul_mat_id(c.0, as_.0, b.0, ids.0) })
    }
    /// `add_id`.
    #[must_use]
    pub fn add_id(c: Ptr, a: Ptr, b: Ptr, ids: Ptr) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_add_id(c.0, a.0, b.0, ids.0) })
    }
    /// `set_rows` (dst, src, idx).
    #[must_use]
    pub fn set_rows(c: Ptr, a: Ptr, b: Ptr, idx: Ptr) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_set_rows(c.0, a.0, b.0, idx.0) })
    }
    /// `argsort_top_k`.
    #[must_use]
    pub fn argsort_top_k(c: Ptr, a: Ptr, k: i32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_argsort_top_k(c.0, a.0, k) })
    }

    macro_rules! op1 {
        ($name:ident, $ffi:ident) => {
            #[must_use]
            pub fn $name(c: Ptr, a: Ptr) -> Ptr {
                Ptr(unsafe { inkwell_ggml::$ffi(c.0, a.0) })
            }
        };
    }
    op1!(sigmoid, ink_sigmoid);
    op1!(tanh, ink_tanh);
    op1!(sqr, ink_sqr);
    op1!(sqrt, ink_sqrt);
    op1!(log, ink_log);
    op1!(silu, ink_silu);
    op1!(gelu, ink_gelu);
    op1!(gelu_quick, ink_gelu_quick);
    op1!(softplus, ink_softplus);
    /// `sum_rows`.
    #[must_use]
    pub fn sum_rows(c: Ptr, a: Ptr) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_sum_rows(c.0, a.0) })
    }
    /// `dup`.
    #[must_use]
    pub fn dup(c: Ptr, a: Ptr) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_dup(c.0, a.0) })
    }
    op1!(cont, ink_cont);
    op1!(transpose, ink_transpose);

    /// `rms_norm`.
    #[must_use]
    pub fn rms_norm(c: Ptr, a: Ptr, eps: f32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_rms_norm(c.0, a.0, eps) })
    }
    /// `scale`.
    #[must_use]
    pub fn scale(c: Ptr, a: Ptr, s: f32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_scale(c.0, a.0, s) })
    }
    /// `scale_bias`.
    #[must_use]
    pub fn scale_bias(c: Ptr, a: Ptr, s: f32, b: f32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_scale_bias(c.0, a.0, s, b) })
    }
    /// `clamp`.
    #[must_use]
    pub fn clamp(c: Ptr, a: Ptr, lo: f32, hi: f32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_clamp(c.0, a.0, lo, hi) })
    }
    /// fill a (dest-shaped) tensor with `v`.
    #[must_use]
    pub fn fill(c: Ptr, a: Ptr, v: f32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_fill(c.0, a.0, v) })
    }
    /// permute axes.
    #[must_use]
    pub fn permute(c: Ptr, a: Ptr, a0: i32, a1: i32, a2: i32, a3: i32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_permute(c.0, a.0, a0, a1, a2, a3) })
    }
    /// reshape 2d.
    #[must_use]
    pub fn reshape2(c: Ptr, a: Ptr, n0: i64, n1: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_reshape_2d(c.0, a.0, n0, n1) })
    }
    /// reshape 3d.
    #[must_use]
    pub fn reshape3(c: Ptr, a: Ptr, n0: i64, n1: i64, n2: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_reshape_3d(c.0, a.0, n0, n1, n2) })
    }
    /// reshape 4d.
    #[must_use]
    pub fn reshape4(c: Ptr, a: Ptr, n0: i64, n1: i64, n2: i64, n3: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_reshape_4d(c.0, a.0, n0, n1, n2, n3) })
    }
    /// view 1d (offset in elements).
    #[must_use]
    pub fn view1(c: Ptr, a: Ptr, n: i64, off: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_view_1d(c.0, a.0, n, off) })
    }
    /// view 2d.
    #[must_use]
    pub fn view2(c: Ptr, a: Ptr, n0: i64, n1: i64, row_stride: i64, off: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_view_2d(c.0, a.0, n0, n1, row_stride, off) })
    }
    /// view 3d.
    #[must_use]
    pub fn view3(c: Ptr, a: Ptr, n0: i64, n1: i64, n2: i64, s1: i64, s2: i64, off: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_view_3d(c.0, a.0, n0, n1, n2, s1, s2, off) })
    }
    pub fn view4(c: Ptr, a: Ptr, n0: i64, n1: i64, n2: i64, n3: i64, s1: i64, s2: i64, s3: i64, off: i64) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_view_4d(c.0, a.0, n0, n1, n2, n3, s1, s2, s3, off) })
    }
    /// pad: [p0 p1 p2 p3] extra elements per dim.
    #[must_use]
    pub fn pad(c: Ptr, a: Ptr, p0: i32, p1: i32, p2: i32, p3: i32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_pad(c.0, a.0, p0, p1, p2, p3) })
    }
    /// `cast`.
    #[must_use]
    pub fn cast(c: Ptr, a: Ptr, ty: i32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_cast(c.0, a.0, ty) })
    }
    /// `soft_max_ext`.
    #[must_use]
    pub fn soft_max(c: Ptr, a: Ptr, mask: Ptr, scale: f32, mb: f32) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_soft_max_ext(c.0, a.0, mask.0, scale, mb) })
    }
    /// `flash_attn_ext` + sinks.
    #[must_use]
    pub fn flash_attn(c: Ptr, q: Ptr, k: Ptr, v: Ptr, mask: Ptr, sinks: Ptr, scale: f32, mb: f32, lsc: f32) -> Ptr {
        let t = Ptr(unsafe { inkwell_ggml::ink_flash_attn_ext(c.0, q.0, k.0, v.0, mask.0, scale, mb, lsc) });
        if !sinks.is_null() {
            unsafe { inkwell_ggml::ink_flash_attn_ext_add_sinks(t.0, sinks.0) };
        }
        t
    }
    /// name (debug).
    #[must_use]
    pub fn named(t: Ptr, name: &std::ffi::CStr) -> Ptr {
        Ptr(unsafe { inkwell_ggml::ink_set_name(t.0, name.as_ptr()) })
    }
    /// mark output.
    pub fn output(t: Ptr) {
        unsafe { inkwell_ggml::ink_set_output(t.0) };
    }
    /// `nbytes`.
    #[must_use]
    pub fn nbytes(t: Ptr) -> usize {
        unsafe { inkwell_ggml::ink_tensor_nbytes(t.0) }
    }
    /// host → tensor.
    ///
    /// # Safety
    /// `src` must be `n` readable bytes.
    pub unsafe fn set(t: Ptr, src: *const u8, off: usize, n: usize) {
        unsafe { inkwell_ggml::ink_tensor_set(t.0, src.cast(), off, n) };
    }
    /// tensor → host.
    ///
    /// # Safety
    /// `dst` must be `n` writable bytes.
    pub unsafe fn get(t: Ptr, dst: *mut u8, off: usize, n: usize) {
        unsafe { inkwell_ggml::ink_tensor_get(t.0, dst.cast(), off, n) };
    }
    /// `build_forward_expand`.
    pub fn build_forward(g: Ptr, out: Ptr) {
        unsafe { inkwell_ggml::ink_build_forward(g.0, out.0) };
    }
    /// ne[i].
    #[must_use]
    pub fn ne(t: Ptr, i: i32) -> i64 {
        unsafe { inkwell_ggml::ink_ne(t.0, i) }
    }
    /// nb[i].
    #[must_use]
    pub fn nb(t: Ptr, i: i32) -> i64 {
        unsafe { inkwell_ggml::ink_nb(t.0, i) }
    }
}
