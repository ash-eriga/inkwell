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
    fn cudaDeviceSynchronize() -> i32;
}

extern "C" {
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
#[expect(clippy::cast_possible_truncation, reason = "dim < i32::MAX")]
pub fn rmsnorm(x: &DevBuf, w: Option<&DevBuf>, y: &mut DevBuf, n: usize, eps: f32) {
    let wp = w.map_or(std::ptr::null(), |b| b.ptr.cast_const());
    unsafe { inkwell_rmsnorm_launch(x.ptr, wp, y.ptr, n as i32, eps) };
    DevBuf::sync();
}

/// L2-clip rows of `dim` to `thr`, optional per-channel scale.
#[expect(clippy::cast_possible_truncation, reason = "dims < i32::MAX")]
pub fn clip(x: &mut DevBuf, scale: Option<&DevBuf>, dim: usize, rows: usize, thr: f32) {
    let sp = scale.map_or(std::ptr::null(), |b| b.ptr.cast_const());
    unsafe { inkwell_clip_launch(x.ptr, sp, dim as i32, rows as i32, thr) };
    DevBuf::sync();
}

/// y = a * sigmoid(b).
#[expect(clippy::cast_possible_truncation, reason = "n < i32::MAX")]
pub fn gate_mul(y: &mut DevBuf, a: &DevBuf, b: &DevBuf, n: usize) {
    unsafe { inkwell_gate_mul_launch(y.ptr, a.ptr, b.ptr, n as i32) };
    DevBuf::sync();
}

/// logits softcap in place.
#[expect(clippy::cast_possible_truncation, reason = "n < i32::MAX")]
pub fn softcap(x: &mut DevBuf, cap: f32) {
    unsafe { inkwell_softcap_launch(x.ptr, x.len as i32, cap) };
    DevBuf::sync();
}

/// `x *= sigmoid(beta*x)` in place (SITU silu family).
#[expect(clippy::cast_possible_truncation, reason = "n < i32::MAX")]
pub fn silu(x: &mut DevBuf, beta: f32) {
    unsafe { inkwell_silu_launch(x.ptr, x.len as i32, beta) };
    DevBuf::sync();
}

/// y = a + b.
#[expect(clippy::cast_possible_truncation, reason = "n < i32::MAX")]
pub fn add(y: &mut DevBuf, a: &DevBuf, b: &DevBuf) {
    unsafe { inkwell_add_launch(y.ptr, a.ptr, b.ptr, a.len as i32) };
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

impl DevBuf {
    /// Allocate `len` floats on device.
    ///
    /// # Panics
    /// On `cudaMalloc` failure.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let mut ptr = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut ptr, len * 4) }, 0, "cudaMalloc");
        Self { ptr: ptr.cast(), len }
    }

    /// Copy host slice → device.
    ///
    /// # Panics
    /// On length mismatch or copy error.
    pub fn write(&mut self, src: &[f32]) {
        assert_eq!(src.len(), self.len);
        assert_eq!(unsafe { cudaMemcpy(self.ptr.cast(), src.as_ptr().cast(), src.len() * 4, 1) }, 0);
    }

    /// Copy device → host vec.
    ///
    /// # Panics
    /// On copy error.
    #[must_use]
    pub fn read(&self) -> Vec<f32> {
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
