//! GPU exl3_gemv on REAL wq_du bytes vs testlib yref.
use inkwell_cuda as _;
use std::ffi::c_void;
extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn ink_launch_exl3_gemv(trel: *const c_void, suh: *const c_void, svh: *const c_void,
        x: *const f32, y: *mut f32, k: i64, n: i64, ka: i64, halfk: i64,
        scratch: *mut f32, stream: *mut c_void);
}
fn npy_f32(p: &str) -> Vec<f32> {
    let d = std::fs::read(p).unwrap();
    let hlen = u16::from_le_bytes([d[8], d[9]]) as usize;
    d[10 + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn main() {
    let trel = std::fs::read("/tmp/wq_trellis.bin").unwrap();
    let suh = std::fs::read("/tmp/wq_suh.bin").unwrap();
    let svh = std::fs::read("/tmp/wq_svh.bin").unwrap();
    let x = npy_f32("/tmp/x_in.npy");
    let yref = npy_f32("/tmp/yref_wq.npy");
    let k = 2304usize; let n = 3072usize;
    unsafe {
        let mut d_t = std::ptr::null_mut(); cudaMalloc(&raw mut d_t, trel.len());
        cudaMemcpy(d_t, trel.as_ptr().cast(), trel.len(), 1);
        let mut d_u = std::ptr::null_mut(); cudaMalloc(&raw mut d_u, suh.len());
        cudaMemcpy(d_u, suh.as_ptr().cast(), suh.len(), 1);
        let mut d_v = std::ptr::null_mut(); cudaMalloc(&raw mut d_v, svh.len());
        cudaMemcpy(d_v, svh.as_ptr().cast(), svh.len(), 1);
        let mut d_x = std::ptr::null_mut(); cudaMalloc(&raw mut d_x, k * 4);
        cudaMemcpy(d_x, x.as_ptr().cast(), k * 4, 1);
        let mut d_s = std::ptr::null_mut(); cudaMalloc(&raw mut d_s, (k + n) * 4);
        let mut d_y = std::ptr::null_mut(); cudaMalloc(&raw mut d_y, n * 4);
        ink_launch_exl3_gemv(d_t, d_u, d_v, d_x.cast(), d_y.cast(), k as i64, n as i64, 3, 0, d_s.cast(), std::ptr::null_mut());
        cudaDeviceSynchronize();
        let mut y = vec![0f32; n];
        cudaMemcpy(y.as_mut_ptr().cast(), d_y, n * 4, 2);
        let md = (0..n).map(|i| (y[i] as f64 - yref[i] as f64).abs()).fold(0f64, f64::max);
        let dot: f64 = (0..n).map(|i| (y[i] as f64) * (yref[i] as f64)).sum();
        let na: f64 = y.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        let nb: f64 = yref.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        eprintln!("REAL wq: maxdiff={md:.4} cos={:.5} gpu={:?} ref={:?}", dot / (na.sqrt() * nb.sqrt()), &y[..6], &yref[..6]);
    }
}
