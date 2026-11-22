use std::ffi::c_void;
use inkwell_cuda as _;
extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn ink_launch_exl3_dp4a_dbg(trel: *const c_void, xh: *const f32,
        s: *mut f32, sx: *mut f32, tk: i64, tn: i64, ka: i64, halfk: i64, stream: *mut c_void);
}
fn main() {
    let trel = std::fs::read("/tmp/wq_trellis.bin").unwrap();
    let k = 2304usize; let n = 3072usize;
    // xh = had_in(x*suh) — reuse kernel path? simpler: CPU compute xh via hadamard
    // load xh computed by python into /tmp/xh.npy
    let xh = npy("/tmp/xh.npy");
    unsafe {
        let mut d_t = std::ptr::null_mut(); cudaMalloc(&raw mut d_t, trel.len());
        cudaMemcpy(d_t, trel.as_ptr().cast(), trel.len(), 1);
        let mut d_x = std::ptr::null_mut(); cudaMalloc(&raw mut d_x, k * 4);
        cudaMemcpy(d_x, xh.as_ptr().cast(), k * 4, 1);
        let mut d_s = std::ptr::null_mut(); cudaMalloc(&raw mut d_s, n * 4);
        let mut d_sx = std::ptr::null_mut(); cudaMalloc(&raw mut d_sx, n * 4);
        ink_launch_exl3_dp4a_dbg(d_t, d_x.cast(), d_s.cast(), d_sx.cast(), 144, 192, 3, 0, std::ptr::null_mut());
        cudaDeviceSynchronize();
        let mut s = vec![0f32; n]; let mut sx = vec![0f32; n];
        cudaMemcpy(s.as_mut_ptr().cast(), d_s, n * 4, 2);
        cudaMemcpy(sx.as_mut_ptr().cast(), d_sx, n * 4, 2);
        eprintln!("s[0..8]={:?}", &s[..8]);
        eprintln!("sx[0..8]={:?}", &sx[..8]);
    }
}
fn npy(p: &str) -> Vec<f32> {
    let d = std::fs::read(p).unwrap();
    let hlen = u16::from_le_bytes([d[8], d[9]]) as usize;
    d[10 + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
