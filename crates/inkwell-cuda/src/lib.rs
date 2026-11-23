//! FFI bindings for the inkwell CUDA kernels + a minimal device
//! buffer wrapper over `libcudart`.
//!
//! GPU kernels stay CUDA (`sm_61`); only host orchestration is Rust.

use std::ffi::c_void;



#[link(name = "cudart")]
extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaGetDeviceCount(c: *mut i32) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: *mut c_void) -> i32;
    fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(s: *mut c_void, e: *mut c_void, flags: u32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaHostAlloc(pp: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostRegister(p: *mut c_void, bytes: usize, flags: u32) -> i32;
    fn cudaHostGetDevicePointer(dst: *mut *mut c_void, src: *mut c_void, flags: u32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

extern "C" {
    fn ink_launch_dequant_iq2_xxs(vx: *const c_void, y: *mut f32, nblocks: i64, stream: *mut c_void);
    fn ink_launch_dequant_iq1_s(vx: *const c_void, y: *mut f32, nblocks: i64, stream: *mut c_void);
    fn ink_launch_mmvq_nvfp4(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_iq1_s(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xxs(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xxs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xxs_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q2_0_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q2_0_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q8_0_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q8_0_i8(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_q6_k_i8(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_attn_decode(K: *const c_void, V: *const c_void, Q: *const f32, bias: *const f32, mask: *const f32, out: *mut f32, nh: i64, hkv: i64, n_kv: i64, hd: i64, k_s1: i64, k_s2: i64, v_s1: i64, v_s2: i64, q_s2: i64, b_s2: i64, scale: f32, kv16: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_s_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xs_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q5_k_i8(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_iq1_s_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_nvfp4_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_nvfp4_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_situ_pair(gu: *const f32, sgu: *const f32, hid: *mut f32, beta: f32, lin_beta: f32, ncols: i64, nslots: i64, stream: *mut c_void);
    fn ink_launch_exl3_gemv(trel: *const c_void, suh: *const c_void, svh: *const c_void, x: *const f32, y: *mut f32, k: i64, n: i64, ka: i64, halfk: i64, scratch: *mut f32, stream: *mut c_void);
    fn ink_launch_exl3_moe(dg: *const c_void, du: *const c_void, dd: *const c_void, ids: *const i32, x: *const f32, xh: *mut f32, gu: *mut f32, hid: *mut f32, dn: *mut f32, w: *const f32, out: *mut f32, nslots: i32, stream: *mut c_void, staging: *mut c_void, sdescs_g: *mut c_void, sdescs_u: *mut c_void, sdescs_d: *mut c_void, on_dev: i32);
    fn ink_launch_exl3_moe_cached(dg: *const c_void, du: *const c_void, dd: *const c_void, ids: *const i32, x: *const f32, xh: *mut f32, gu: *mut f32, hid: *mut f32, dn: *mut f32, w: *const f32, out: *mut f32, nslots: i32, stream: *mut c_void, cache: *mut c_void, tag: *mut i32, map: *mut i32, miss: *mut i32, rr: *mut i32, k: i32, slot_proj: i64, sdescs_g: *mut c_void, sdescs_u: *mut c_void, sdescs_d: *mut c_void);
    fn ink_launch_exl3_stage(dg: *const c_void, du: *const c_void, dd: *const c_void, ids: *const i32, staging: *mut c_void, sg: *mut c_void, su: *mut c_void, sd: *mut c_void, stream: *mut c_void);
    fn ink_launch_conv_step(x_in: *const f32, state: *mut f32, region_off: i32, w: *const f32, dyn_p: *const f32, ch: i32, n_heads: i32, scale: f32, out: *mut f32, stream: *mut c_void);
    fn ink_launch_expert_wsum(dn: *const f32, w: *const f32, sd: *const f32, out: *mut f32, ncols: i64, nslots: i64, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_s_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq4_xs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_swiglu_pair(y: *mut f32, gu: *const f32, nslots: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_q2_0(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn inkwell_rmsnorm_launch(x: *const f32, w: *const f32, y: *mut f32, n: i32, eps: f32);
    fn inkwell_clip_launch(x: *mut f32, scale: *const f32, dim: i32, rows: i32, thr: f32);
    fn inkwell_gate_mul_launch(y: *mut f32, a: *const f32, b: *const f32, n: i32);
    fn inkwell_softcap_launch(x: *mut f32, n: i32, cap: f32);
    fn inkwell_silu_launch(x: *mut f32, n: i32, beta: f32);
    fn inkwell_add_launch(y: *mut f32, a: *const f32, b: *const f32, n: i32);
}

/// Returns the CUDA device count (probe for the runtime).
#[must_use]
pub fn device_count() -> i32 {
    let mut c = 0;
    unsafe { cudaGetDeviceCount(&raw mut c) };
    c
}

/// `rms_norm(x)*w` (w optional); x and y may alias.
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "dim < i32::MAX")]
pub fn rmsnorm(x: &DevBuf, w: Option<&DevBuf>, y: &mut DevBuf, n: usize, eps: f32) {
    let wp = w.map_or(std::ptr::null(), |b| b.ptr.cast_const());
    unsafe { inkwell_rmsnorm_launch(x.ptr, wp, y.ptr, n as i32, eps) };
    DevBuf::sync();
}

/// L2-clip rows of `dim` to `thr`, optional per-channel scale.
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "dims < i32::MAX")]
pub fn clip(x: &mut DevBuf, scale: Option<&DevBuf>, dim: usize, rows: usize, thr: f32) {
    let sp = scale.map_or(std::ptr::null(), |b| b.ptr.cast_const());
    unsafe { inkwell_clip_launch(x.ptr, sp, dim as i32, rows as i32, thr) };
    DevBuf::sync();
}

/// y = a * sigmoid(b).
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "n < i32::MAX")]
pub fn gate_mul(y: &mut DevBuf, a: &DevBuf, b: &DevBuf, n: usize) {
    unsafe { inkwell_gate_mul_launch(y.ptr, a.ptr, b.ptr, n as i32) };
    DevBuf::sync();
}

/// logits softcap in place.
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "n < i32::MAX")]
pub fn softcap(x: &mut DevBuf, cap: f32) {
    unsafe { inkwell_softcap_launch(x.ptr, x.len as i32, cap) };
    DevBuf::sync();
}

/// `x *= sigmoid(beta*x)` in place (SITU silu family).
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "n < i32::MAX")]
pub fn silu(x: &mut DevBuf, beta: f32) {
    unsafe { inkwell_silu_launch(x.ptr, x.len as i32, beta) };
    DevBuf::sync();
}

/// y = a + b.
#[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "n < i32::MAX")]
pub fn add(y: &mut DevBuf, a: &DevBuf, b: &DevBuf) {
    unsafe { inkwell_add_launch(y.ptr, a.ptr, b.ptr, a.len as i32) };
    DevBuf::sync();
}


/// GEMV `y[r] = dequant(W[r,:]) . x` for IQ2_XXS weights.
/// `w` = raw quant rows, `x` = f32 activations, `y` = `nrows` out.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn mmvq_iq2_xxs(w: &DevBuf, x: &DevBuf, y: &mut DevBuf, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_iq2_xxs(w.ptr.cast_const().cast(), x.ptr, y.ptr, nrows as i64, ncols as i64, stream()) };
    DevBuf::sync();
}

/// GEMV `y[r] = dequant(W[r,:]) . x` for Q2_0 weights.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn mmvq_q2_0(w: &DevBuf, x: &DevBuf, y: &mut DevBuf, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_q2_0(w.ptr.cast_const().cast(), x.ptr, y.ptr, nrows as i64, ncols as i64, stream()) };
    DevBuf::sync();
}

/// Register host memory for zero-copy device access; returns a
/// device-visible pointer. `flags` = `cudaHostRegisterMapped` (2).
///
/// # Panics
/// On registration failure.
#[must_use]
pub fn host_register_mapped(host: &mut [u8]) -> *mut c_void {
    assert_eq!(unsafe { cudaHostRegister(host.as_mut_ptr().cast(), host.len(), 2) }, 0, "cudaHostRegister");
    let mut dptr = std::ptr::null_mut();
    assert_eq!(unsafe { cudaHostGetDevicePointer(&raw mut dptr, host.as_mut_ptr().cast(), 0) }, 0);
    dptr
}

/// Zero-copy mapping of read-only host memory (e.g. an mmap'd GGUF).
///
/// # Panics
/// On registration failure.
#[must_use]
pub fn host_register_mapped_ro(host: &[u8]) -> *mut c_void {
    let p = host.as_ptr() as *mut c_void; // const-cast: CUDA only reads
    assert_eq!(unsafe { cudaHostRegister(p, host.len(), 2) }, 0, "cudaHostRegister");
    let mut dptr = std::ptr::null_mut();
    assert_eq!(unsafe { cudaHostGetDevicePointer(&raw mut dptr, p, 0) }, 0);
    dptr
}

/// GEMV with weights read zero-copy from registered host memory.
///
/// # Safety
/// `w` must be a device pointer from [`host_register_mapped`] covering
/// `nrows` full rows of quant data.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_xxs_host(w: *mut c_void, x: &DevBuf, y: &mut DevBuf, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_iq2_xxs(w, x.ptr, y.ptr, nrows as i64, ncols as i64, stream()) };
    DevBuf::sync();
}

/// Batched expert GEMV: `y[slot][r] = dequant(W[ids[slot]][r,:]) . x`.
/// `w` may be a device ptr or a `host_register_mapped*` host pointer.
///
/// # Safety
/// `w` and `ids` must be device-accessible pointers.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_xxs_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_xxs_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// Batched expert GEMV for Q2_0 weights.
///
/// # Safety
/// See [`mmvq_iq2_xxs_moe`].
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_q2_0_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_q2_0_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// Fused `silu(gate)*up` for gate_up pairs; `gu` = `[nslots][2*ncols]`.
///
/// # Panics
/// On launch failure.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn swiglu_pair(y: &mut DevBuf, gu: &DevBuf, nslots: usize, ncols: usize) {
    unsafe { ink_launch_swiglu_pair(y.ptr, gu.ptr, nslots as i64, ncols as i64, stream()) };
    DevBuf::sync();
}

/// GEMV for NVFP4 weights (36 bytes per 64 values).
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn mmvq_nvfp4(w: &DevBuf, x: &DevBuf, y: &mut DevBuf, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_nvfp4(w.ptr.cast_const().cast(), x.ptr, y.ptr, nrows as i64, ncols as i64, stream()) };
    DevBuf::sync();
}

/// Dequantize IQ1_S blocks (50 bytes per 256 values) on GPU.
#[expect(clippy::cast_possible_wrap, reason = "nblocks < i64::MAX")]
pub fn dequant_iq1_s(src: &DevBuf, dst: &mut DevBuf) {
    let nblocks = (dst.len / 256) as i64;
    unsafe { ink_launch_dequant_iq1_s(src.ptr.cast_const().cast(), dst.ptr, nblocks, stream()) };
    DevBuf::sync();
}

/// GEMV for IQ1_S weights.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn mmvq_iq1_s(w: &DevBuf, x: &DevBuf, y: &mut DevBuf, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_iq1_s(w.ptr.cast_const().cast(), x.ptr, y.ptr, nrows as i64, ncols as i64, stream()) };
    DevBuf::sync();
}

/// Dequantize IQ2_XXS blocks (66 bytes per 256 values) on GPU.
/// `src` = raw quant bytes, `dst` = `nblocks*256` floats.
#[expect(clippy::cast_possible_wrap, reason = "nblocks < i64::MAX")]
pub fn dequant_iq2_xxs(src: &DevBuf, dst: &mut DevBuf) {
    let nblocks = (dst.len / 256) as i64;
    unsafe { ink_launch_dequant_iq2_xxs(src.ptr.cast_const().cast(), dst.ptr, nblocks, stream()) };
    DevBuf::sync();
}

/// Kernels are launched through `extern "C"` wrappers in
/// `kernels.cu` (`__global__` isn't callable across FFI).
#[derive(Debug)]
pub struct DevBuf {
    ptr: *mut f32,

    /// element count
    pub len: usize,
}

unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl DevBuf {
    /// Allocate `len` floats on device.
    ///
    /// # Panics
    /// On `cudaMalloc` failure.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&raw mut ptr, len * 4) }, 0, "cudaMalloc");
        Self { ptr: ptr.cast(), len }
    }

    /// Allocate `bytes` raw bytes on device (quant payloads).
    ///
    /// # Panics
    /// On `cudaMalloc` failure.
    #[must_use]
    pub fn new_bytes(bytes: usize) -> Self {
        Self::try_new_bytes(bytes).expect("cudaMalloc")
    }

    /// Fallible variant of `new_bytes` (returns `None` on OOM).
    pub fn try_new_bytes(bytes: usize) -> Option<Self> {
        let mut ptr = std::ptr::null_mut();
        if unsafe { cudaMalloc(&raw mut ptr, bytes) } != 0 {
            return None;
        }
        Some(Self { ptr: ptr.cast(), len: bytes / 4 })
    }

    /// Zero-fill the buffer.
    ///
    /// # Panics
    /// On memset error.
    pub fn zero(&self) {
        assert_eq!(unsafe { cudaMemset(self.ptr.cast(), 0, self.len * 4) }, 0);
    }

    /// Copy raw bytes → device.
    ///
    /// # Panics
    /// On copy error.
    /// Copy `self` (device) → `dst` (host), then sync.
    ///
    /// # Panics
    /// On `cudaMemcpy` failure.
    /// Raw device pointer for kernels that take `*mut c_void`.
    #[must_use]
    pub fn dev_ptr(&self) -> *mut c_void {
        self.ptr.cast()
    }

    /// Blocking D2H read of one u64 (debug only).
    pub fn read_u64(&self) -> u64 {
        let mut b = [0u8; 8];
        self.read_bytes(&mut b);
        u64::from_le_bytes(b)
    }

    pub fn read_bytes(&self, dst: &mut [u8]) {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if std::env::var("INKWELL_MEMCPY_DEBUG").is_ok() {
            eprintln!("MC read_bytes len={} n={}", dst.len(), N.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        }
        let rc = unsafe { cudaMemcpy(dst.as_mut_ptr().cast(), self.ptr.cast::<std::ffi::c_void>(), dst.len(), 2) }; // DtoH
        assert_eq!(rc, 0, "cudaMemcpy DtoH");
    }

    pub fn write_bytes(&mut self, src: &[u8]) {
        if std::env::var("INKWELL_MEMCPY_DEBUG").is_ok() {
            eprintln!("MC write_bytes len={}", src.len());
        }
        assert_eq!(unsafe { cudaMemcpy(self.ptr.cast(), src.as_ptr().cast(), src.len(), 1) }, 0);
    }

    /// Copy host slice → device.
    ///
    /// # Panics
    /// On length mismatch or copy error.
    pub fn write(&mut self, src: &[f32]) {
        if std::env::var("INKWELL_MEMCPY_DEBUG").is_ok() {
            eprintln!("MC write len={}", src.len() * 4);
        }
        assert_eq!(src.len(), self.len);
        assert_eq!(unsafe { cudaMemcpy(self.ptr.cast(), src.as_ptr().cast(), src.len() * 4, 1) }, 0);
    }

    /// Async HtoD byte upload on stream 0 (caller syncs).
    /// Same as [`Self::write_bytes_async`] on the side stream.
    pub fn write_bytes_async_side(&self, src: &[u8]) {
        let rc = unsafe { cudaMemcpyAsync(self.ptr.cast(), src.as_ptr().cast(), src.len(), 1, side_stream()) };
        assert_eq!(rc, 0, "cudaMemcpyAsync HtoD side");
    }
    pub fn write_bytes_async(&self, src: &[u8]) {
        let rc = unsafe { cudaMemcpyAsync(self.ptr.cast(), src.as_ptr().cast(), src.len(), 1, stream()) };
        assert_eq!(rc, 0, "cudaMemcpyAsync HtoD");
    }

    /// Copy device → host vec.
    ///
    /// # Panics
    /// On copy error.
    #[must_use]
    pub fn read(&self) -> Vec<f32> {
        if std::env::var("INKWELL_MEMCPY_DEBUG").is_ok() {
            eprintln!("MC read len={}", self.len * 4);
        }
        let mut v = vec![0.0f32; self.len];
        assert_eq!(unsafe { cudaMemcpy(v.as_mut_ptr().cast(), self.ptr.cast(), self.len * 4, 2) }, 0);
        v
    }

    /// Sync device, panic on kernel error.
    ///
    /// # Panics
    /// On pending CUDA error.
    pub fn sync() {
        let e = unsafe { cudaDeviceSynchronize() };
        assert_eq!(e, 0, "cuda sync {e}");
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.ptr.cast()) };
    }
}

/// Batched IQ1_S expert GEMV.
///
/// # Safety
/// See [`mmvq_iq2_xxs_moe`].
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq1_s_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq1_s_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// Batched NVFP4 expert GEMV.
///
/// # Safety
/// See [`mmvq_iq2_xxs_moe`].
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_nvfp4_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_nvfp4_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// SITU gate·up activation with optional per-expert scales.
///
/// # Panics
/// On launch failure.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn situ_pair(hid: &mut DevBuf, gu: &DevBuf, sgu: Option<&DevBuf>, beta: f32, lin_beta: f32, ncols: usize, nslots: usize) {
    let s = sgu.map_or(std::ptr::null(), |b| b.ptr);
    unsafe { ink_launch_situ_pair(gu.ptr, s, hid.ptr, beta, lin_beta, ncols as i64, nslots as i64, stream()) };
    DevBuf::sync();
}

/// Weighted sum across expert slots: `out[c] = Σ w[s]·sd[s]·dn[s][c]`.
///
/// # Panics
/// On launch failure.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub fn expert_wsum(out: &mut DevBuf, dn: &DevBuf, w: &DevBuf, sd: Option<&DevBuf>, ncols: usize, nslots: usize) {
    let s = sd.map_or(std::ptr::null(), |b| b.ptr);
    unsafe { ink_launch_expert_wsum(dn.ptr, w.ptr, s, out.ptr, ncols as i64, nslots as i64, stream()) };
    DevBuf::sync();
}

/// Batched IQ2_XS expert GEMV.
///
/// # Safety
/// See [`mmvq_iq2_xxs_moe`].
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_xs_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_xs_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// Batched IQ2_S expert GEMV.
///
/// # Safety
/// See [`mmvq_iq2_xxs_moe`].
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_s_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_s_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// Batched IQ4_XS expert GEMV.
///
/// # Safety
/// See [`mmvq_iq2_xxs_moe`].
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq4_xs_moe(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq4_xs_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
    DevBuf::sync();
}

/// `cudaHostAlloc` pinned+mapped buffer (UVA → same ptr on device on Pascal).
///
/// # Panics
/// On allocation failure.
#[must_use]
pub fn host_alloc_mapped(bytes: usize) -> PinnedBuf {
    let mut p: *mut c_void = std::ptr::null_mut();
    // cudaHostAllocDefault — UVA makes it device-addressable.
    let rc = unsafe { cudaHostAlloc(&mut p, bytes, 0) };
    assert_eq!(rc, 0, "cudaHostAlloc({bytes}) = {rc}");
    let mut dp: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { cudaHostGetDevicePointer(&mut dp, p, 0) };
    assert_eq!(rc, 0, "cudaHostGetDevicePointer = {rc}");
    PinnedBuf { host: p.cast(), dev: dp }
}

/// Pinned host buffer usable as a device pointer (zero-copy over PCIe).
#[derive(Debug)]
pub struct PinnedBuf {
    /// Host-accessible pointer.
    pub host: *mut u8,
    /// Device-accessible pointer to the same memory.
    pub dev: *mut c_void,
}
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}


// ---- async variants: no device sync, for chained launches ----

/// Same as [`mmvq_iq2_xxs_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_xxs_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_xxs_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// int8-dp4a variant of [`mmvq_iq2_xxs_moe_async`]; async.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_xxs_moe_i8_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_xxs_moe_i8(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Same as [`mmvq_q2_0_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_q2_0_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_q2_0_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Same as [`mmvq_iq1_s_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq1_s_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq1_s_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// DP4A variant of q2_0 (~0.05% error).
/// # Safety
/// See [`mmvq_nvfp4_moe`].
pub unsafe fn mmvq_q2_0_moe_i8_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_q2_0_moe_i8(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}
/// Single-tensor dense q8_0 GEMV (dp4a). Zeroes `y` internally.
/// # Safety
/// `w`/`x`/`y` are device pointers.
pub unsafe fn mmvq_q8_0_i8_async(w: *const c_void, x: *const c_void, y: *mut f32, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_q8_0_i8(w, x.cast(), y, nrows as i64, ncols as i64, stream()) };
}
/// Single-tensor dense Q6_K GEMV (dp4a). Zeroes `y` internally.
/// # Safety
/// `w`/`x`/`y` are device pointers.
pub unsafe fn mmvq_q6_k_i8_async(w: *const c_void, x: *const c_void, y: *mut f32, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_q6_k_i8(w, x.cast(), y, nrows as i64, ncols as i64, stream()) };
}

/// MoE IQ2_XS GEMV (dp4a, opt-in).
/// # Safety
/// `wp`/`x`/`y` are device pointers.
pub unsafe fn mmvq_iq2_xs_moe_i8_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_xs_moe_i8(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// MoE IQ2_S GEMV (dp4a).
/// # Safety
/// `wp`/`x`/`y` are device pointers.
pub unsafe fn mmvq_iq2_s_moe_i8_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_s_moe_i8(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Fused decode attention (Pascal): scores+softmax+V-mul in one launch per head.
/// # Safety
/// All pointers are device pointers; strides are in elements.
#[expect(clippy::too_many_arguments)]
pub unsafe fn attn_decode_async(k: *const c_void, v: *const c_void, q: *const c_void, bias: *const c_void, mask: *const c_void, out: *mut f32, nh: i64, hkv: i64, n_kv: i64, hd: i64, k_s1: i64, k_s2: i64, v_s1: i64, v_s2: i64, q_s2: i64, b_s2: i64, scale: f32, kv16: bool) {
    unsafe { ink_launch_attn_decode(k.cast(), v.cast(), q.cast(), bias.cast(), mask.cast(), out, nh, hkv, n_kv, hd, k_s1, k_s2, v_s1, v_s2, q_s2, b_s2, scale, i32::from(kv16), stream()) };
}
/// Single-tensor dense Q5_K GEMV (dp4a). Zeroes `y` internally.
/// # Safety
/// `w`/`x`/`y` are device pointers.
pub unsafe fn mmvq_q5_k_i8_async(w: *const c_void, x: *const c_void, y: *mut f32, nrows: usize, ncols: usize) {
    unsafe { ink_launch_mmvq_q5_k_i8(w, x.cast(), y, nrows as i64, ncols as i64, stream()) };
}
/// DP4A dense GEMV for q8_0: x quantized to int8 per 32-val block (same
/// precision class as ggml's q8_1 x-quant path).
/// # Safety
/// `wp` is a device array of weight pointers; `y` gets `nslots*nrows` floats.
pub unsafe fn mmvq_q8_0_moe_i8_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_q8_0_moe_i8(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}
/// DP4A variant: x quantized to int8 per 64-val block (~0.1% error).
/// # Safety
/// See [`mmvq_nvfp4_moe`].
pub unsafe fn mmvq_nvfp4_moe_i8_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_nvfp4_moe_i8(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}
/// Same as [`mmvq_nvfp4_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_nvfp4_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_nvfp4_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Same as [`mmvq_iq2_xs_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_xs_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_xs_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Same as [`mmvq_iq2_s_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq2_s_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq2_s_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Same as [`mmvq_iq4_xs_moe`] but does not synchronize; caller must sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
pub unsafe fn mmvq_iq4_xs_moe_async(wp: *const c_void, x: *const c_void, y: &mut DevBuf, nrows: usize, ncols: usize, x_stride: usize, nslots: usize) {
    unsafe { ink_launch_mmvq_iq4_xs_moe(wp.cast(), x.cast(), y.ptr, nrows as i64, ncols as i64, x_stride as i64, nslots as i32, stream()) };
}

/// Async `situ_pair` — no sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
/// # Safety
/// `sgu` may be null; pointers valid for `ncols*nslots` elements.
pub unsafe fn situ_pair_async(hid: &mut DevBuf, gu: &DevBuf, sgu: *const c_void, beta: f32, lin_beta: f32, ncols: usize, nslots: usize) {
    unsafe { ink_launch_situ_pair(gu.ptr, sgu.cast(), hid.ptr, beta, lin_beta, ncols as i64, nslots as i64, stream()) };
}

/// Async `expert_wsum` — no sync.
#[expect(clippy::cast_possible_wrap, reason = "dims < i64::MAX")]
/// # Safety
/// `sd` may be null; `w` valid for `nslots`, `out` for `ncols`.
pub unsafe fn expert_wsum_async(out: *mut c_void, dn: &DevBuf, w: *const c_void, sd: *const c_void, ncols: usize, nslots: usize) {
    unsafe { ink_launch_expert_wsum(dn.ptr, w.cast(), sd.cast(), out.cast(), ncols as i64, nslots as i64, stream()) };
}

/// Free/total VRAM in bytes.
#[must_use]
pub fn mem_info() -> (usize, usize) {
    let mut f = 0usize;
    let mut t = 0usize;
    assert_eq!(unsafe { cudaMemGetInfo(&raw mut f, &raw mut t) }, 0, "cudaMemGetInfo");
    (f, t)
}

/// Async HtoD copy on stream 0 (ordered with kernels).
///
/// # Safety
/// `dst`/`src` valid for `bytes`.
pub unsafe fn memcpy_h2d_async(dst: *mut c_void, src: *const c_void, bytes: usize) {
    let rc = unsafe { cudaMemcpyAsync(dst, src, bytes, 1, stream()) };
    assert_eq!(rc, 0, "cudaMemcpyAsync HtoD");
}
/// Secondary stream + event for overlapping MoE expert promotion with compute.
fn side_stream() -> *mut c_void {
    use std::sync::OnceLock;
    static S: OnceLock<usize> = OnceLock::new();
    *S.get_or_init(|| {
        let mut s = std::ptr::null_mut();
        assert_eq!(unsafe { cudaStreamCreateWithFlags(&mut s, 1) }, 0);
        s as usize
    }) as *mut c_void
}
fn side_event() -> *mut c_void {
    use std::sync::OnceLock;
    static E: OnceLock<usize> = OnceLock::new();
    *E.get_or_init(|| {
        let mut e = std::ptr::null_mut();
        assert_eq!(unsafe { cudaEventCreateWithFlags(&mut e, 2) }, 0);
        e as usize
    }) as *mut c_void
}
/// Async HtoD copy on the side stream.
///
/// # Safety
/// `dst`/`src` valid for `bytes`.
pub unsafe fn memcpy_h2d_async_side(dst: *mut c_void, src: *const c_void, bytes: usize) {
    let rc = unsafe { cudaMemcpyAsync(dst, src, bytes, 1, side_stream()) };
    assert_eq!(rc, 0, "cudaMemcpyAsync HtoD side");
}
/// Record the side stream's event and make stream 0 wait on it.
///
/// # Safety
/// Stream/event plumbing only.
pub unsafe fn side_sync_to_main() {
    let s = side_stream();
    let e = side_event();
    assert_eq!(unsafe { cudaEventRecord(e, s) }, 0);
    assert_eq!(unsafe { cudaStreamWaitEvent(std::ptr::null_mut(), e, 0) }, 0);
}

extern "C" {
    fn ink_launch_moe_resolve(ids: *const i32, map: *const i32,
        lru: *mut u64, stamp: *const u64,
        arena: u64, host: u64, esz: i64,
        ptrs: *mut u64, missq: *mut i32, nslots: i32, stream: *mut c_void);
    fn ink_launch_moe_alloc(missq: *const i32, map: *mut i32,
        lru: *mut u64, stamp: *mut u64, rev: *mut i32,
        arena: u64, host: u64, esz: i64,
        ptrs: *mut u64, work: *mut u64, k: i32, stream: *mut c_void);
    fn ink_launch_moe_gather(missq: *const i32, work: *const u64, esz: i64,
        stats: *mut u64, stream: *mut c_void);
    fn cudaMemsetAsync(dst: *mut c_void, v: i32, bytes: usize, stream: *mut c_void) -> i32;
    fn cudaMemset(dst: *mut c_void, v: i32, bytes: usize) -> i32;
}

/// Set the stream all `*_async` helpers below launch on.
///
/// # Safety
/// `s` must outlive all launches (use the ggml CUDA stream).
static INK_STREAM: std::sync::atomic::AtomicPtr<c_void> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
pub unsafe fn set_stream(s: *mut c_void) {
    INK_STREAM.store(s, std::sync::atomic::Ordering::Relaxed);
}
/// EXL3 trellis GEMV: y = svh . H (What^T . H (suh . x)) for one weight.
///
/// # Safety
/// Raw device pointers; launched on the ggml cuda stream.
pub unsafe fn exl3_gemv_async(trel: *const c_void, suh: *const c_void, svh: *const c_void,
    x: *const f32, y: *mut f32, k: usize, n: usize, ka: usize, halfk: bool, scratch: *mut f32) {
    unsafe { ink_launch_exl3_gemv(trel, suh, svh, x, y, k as i64, n as i64, ka as i64, halfk as i64, scratch, stream()) };
}
/// EXL3 MoE per-slot expert chain (gate+up -> situ -> down -> wsum).
///
/// # Safety
/// desc tables are pinned host-mapped u64[128][4] arrays.
pub unsafe fn conv_step_async(x_in: *const f32, state: *mut f32, region_off: i32,
    w: *const f32, dyn_p: *const f32, ch: i32, n_heads: i32, scale: f32, out: *mut f32) {
    unsafe { ink_launch_conv_step(x_in, state, region_off, w, dyn_p, ch, n_heads, scale, out, stream()) };
}

pub unsafe fn exl3_moe_async(dg: *const c_void, du: *const c_void, dd: *const c_void,
    ids: *const i32, x: *const f32, xh: *mut f32, gu: *mut f32, hid: *mut f32, dn: *mut f32,
    w: *const f32, out: *mut f32, nslots: i32,
    staging: *mut c_void, sdescs_g: *mut c_void, sdescs_u: *mut c_void, sdescs_d: *mut c_void,
    on_dev: bool) {
    unsafe { ink_launch_exl3_moe(dg, du, dd, ids, x, xh, gu, hid, dn, w, out, nslots, stream(), staging, sdescs_g, sdescs_u, sdescs_d, i32::from(on_dev)) };
}

/// Cached variant: per-layer VRAM cache of expert blobs; copies only misses.
///
/// # Safety
/// `cache` holds `k` slots × 3 projs × EXL3_STAGE_PROJ bytes; `tag` has `k` i32
/// (init to 0xFF); `map`/`miss` have `nslots` i32; `rr` is one i32.
pub unsafe fn exl3_moe_cached_async(dg: *const c_void, du: *const c_void, dd: *const c_void,
    ids: *const i32, x: *const f32, xh: *mut f32, gu: *mut f32, hid: *mut f32, dn: *mut f32,
    w: *const f32, out: *mut f32, nslots: i32,
    cache: *mut c_void, tag: *mut i32, map: *mut i32, miss: *mut i32, rr: *mut i32, k: i32,
    slot_proj: i64, sdescs_g: *mut c_void, sdescs_u: *mut c_void, sdescs_d: *mut c_void) {
    unsafe { ink_launch_exl3_moe_cached(dg, du, dd, ids, x, xh, gu, hid, dn, w, out, nslots, stream(), cache, tag, map, miss, rr, k, slot_proj, sdescs_g, sdescs_u, sdescs_d) };
}
fn stream() -> *mut c_void { INK_STREAM.load(std::sync::atomic::Ordering::Relaxed) }

/// ids[s] -> ptrs[s] via device map (misses pushed to missq).
///
/// # Safety
/// `ids`/`map`/`ptrs`/`missq` are device pointers; `host` may be a host-mapped pointer.
pub unsafe fn moe_resolve_async(ids: *const i32, map: *const i32,
    lru: *mut u64, stamp: *const u64,
    arena: u64, host: u64, esz: i64,
    ptrs: *mut u64, missq: *mut i32, nslots: usize) {
    unsafe {
        ink_launch_moe_resolve(ids, map, lru, stamp, arena, host, esz, ptrs, missq,
            nslots as i32, stream());
    }
}

/// Pick LRU victims for queued misses on device; writes ptrs + copy work items.
///
/// # Safety
/// All pointer args are device pointers.
pub unsafe fn moe_alloc_async(missq: *const i32, map: *mut i32,
    lru: *mut u64, stamp: *mut u64, rev: *mut i32,
    arena: u64, host: u64, esz: i64,
    ptrs: *mut u64, work: *mut u64, k: usize) {
    unsafe {
        ink_launch_moe_alloc(missq, map, lru, stamp, rev, arena, host, esz,
            ptrs, work, k as i32, stream());
    }
}

/// Copy queued misses host->VRAM slot (zero-copy read), one block each.
///
/// # Safety
/// `missq`/`work` are device pointers; work srcs may be host-mapped.
/// Gather enqueued on the side stream (overlaps next GEMVs on the main stream).
///
/// # Safety
/// `missq`/`work` device-valid; call [`mark_dispatch_ready`] first.
pub unsafe fn moe_gather_async(missq: *const i32, work: *const u64, esz: i64, stats: *mut u64) {
    unsafe { ink_launch_moe_gather(missq, work, esz, stats, stream()) }
}

/// Gather on the side stream (caller joins with [`wait_gather_done`]).
///
/// # Safety
/// Same contract as [`moe_gather_async`].
pub unsafe fn moe_gather_side_async(missq: *const i32, work: *const u64, esz: i64, stats: *mut u64) {
    unsafe { ink_launch_moe_gather(missq, work, esz, stats, side_stream()) }
}

/// Async device memset on the default stream.
///
/// # Safety
/// `dst` is a device pointer.
pub unsafe fn memset_dev_async(dst: *mut c_void, v: i32, bytes: usize) {
    unsafe { cudaMemsetAsync(dst, v, bytes, stream()) };
}

/// cudaDeviceSynchronize + report last error (debug only).
///
/// # Safety
/// Synchronizes the device.
pub unsafe fn sync_check(tag: &str) {
    extern "C" { fn cudaDeviceSynchronize() -> i32; fn cudaGetLastError() -> i32; }
    unsafe {
        cudaDeviceSynchronize();
        let e = cudaGetLastError();
        if e != 0 { eprintln!("CUDAERR@{tag}: {e}"); }
    }
}


fn gather_ready_event() -> *mut c_void {
    use std::sync::OnceLock;
    static E: OnceLock<usize> = OnceLock::new();
    *E.get_or_init(|| { let mut e = std::ptr::null_mut(); assert_eq!(unsafe { cudaEventCreateWithFlags(&mut e, 2) }, 0); e as usize }) as *mut c_void
}
fn gather_done_event() -> *mut c_void {
    use std::sync::OnceLock;
    static E: OnceLock<usize> = OnceLock::new();
    *E.get_or_init(|| { let mut e = std::ptr::null_mut(); assert_eq!(unsafe { cudaEventCreateWithFlags(&mut e, 2) }, 0); e as usize }) as *mut c_void
}
/// Stream0 → side: the gather inputs (missq/work) are ready.
///
/// # Safety
/// Event plumbing only.
pub unsafe fn mark_dispatch_ready() {
    let e = gather_ready_event();
    assert_eq!(unsafe { cudaEventRecord(e, stream()) }, 0);
    assert_eq!(unsafe { cudaStreamWaitEvent(side_stream(), e, 0) }, 0);
}
/// Side → stream0 (next visit): all gather writes are visible.
///
/// # Safety
/// Event plumbing only.
pub unsafe fn mark_gather_done() {
    assert_eq!(unsafe { cudaEventRecord(gather_done_event(), side_stream()) }, 0);
}
/// Make the main stream wait for the previous visit's gathers.
///
/// # Safety
/// Event plumbing only.
pub unsafe fn wait_gather_done() {
    assert_eq!(unsafe { cudaStreamWaitEvent(stream(), gather_done_event(), 0) }, 0);
}

/// Launch only the expert staging kernel (pinned→VRAM DMA), for benching.
///
/// # Safety
/// Pointers must be valid device/mapped buffers.
pub unsafe fn launch_stage_only(descs_dev: *const c_void, ids: *const i32,
    staging: *mut c_void, sdescs: *mut c_void) {
    unsafe { ink_launch_exl3_stage(descs_dev, descs_dev, descs_dev, ids, staging, sdescs, sdescs, sdescs, stream()) };
}
