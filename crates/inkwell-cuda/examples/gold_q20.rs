use inkwell_cuda as _;
use std::ffi::c_void;
extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn ink_launch_mmvq_q2_0_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
}
fn f16(b: u16) -> f32 {
    let s = ((b >> 15) & 1) as f32; let e = ((b >> 10) & 31) as i32; let m = (b & 1023) as f32;
    if e == 0 { return if m == 0.0 { 0.0 } else { (1.0 - 2.0 * s) * (m / 1024.0) * f32::powi(2.0, -14) }; }
    (1.0 - 2.0 * s) * (1.0 + m / 1024.0) * f32::powi(2.0, e - 15)
}
fn main() {
    const NR: i64 = 8; const NC: i64 = 64; // 1 block/row
    let mut w = vec![0u8; (NR as usize) * 18];
    // deterministic pattern
    for (i, v) in w.iter_mut().enumerate() { *v = (i * 7 + 3) as u8 & 0x3F; }
    let x: Vec<f32> = (0..NC).map(|i| (i as f32) * 0.01 - 0.3).collect();
    // CPU ref: byte bi -> vals bi*4..+3, q=(byte>>2j)&3 -1
    let mut yref = vec![0f32; NR as usize];
    for r in 0..NR as usize {
        let row = &w[r * 18..(r + 1) * 18];
        let d = f16(u16::from_le_bytes([row[0], row[1]]));
        let mut acc = 0f32;
        for bi in 0..16 {
            let byte = row[2 + bi];
            for j in 0..4 { acc += d * (((byte >> (j * 2)) & 3) as i32 - 1) as f32 * x[bi * 4 + j]; }
        }
        yref[r] = acc;
    }
    unsafe {
        let mut dw = std::ptr::null_mut(); cudaMalloc(&raw mut dw, w.len());
        cudaMemcpy(dw, w.as_ptr().cast(), w.len(), 1);
        let mut dx = std::ptr::null_mut(); cudaMalloc(&raw mut dx, NC as usize * 4);
        cudaMemcpy(dx, x.as_ptr().cast(), x.len() * 4, 1);
        let mut dy = std::ptr::null_mut(); cudaMalloc(&raw mut dy, NR as usize * 4);
        let wp = [dw as *const c_void];
        let mut dwp = std::ptr::null_mut(); cudaMalloc(&raw mut dwp, 8);
        cudaMemcpy(dwp, wp.as_ptr().cast(), 8, 1);
        ink_launch_mmvq_q2_0_moe(dwp.cast(), dx.cast(), dy.cast(), NR, NC, NC, 1, std::ptr::null_mut());
        cudaDeviceSynchronize();
        let mut y = vec![0f32; NR as usize];
        cudaMemcpy(y.as_mut_ptr().cast(), dy, y.len() * 4, 2);
        let mut md = 0f32;
        for i in 0..NR as usize { md = md.max((y[i] - yref[i]).abs()); }
        println!("gpu {:?}\ncpu {:?}\nmaxdiff {md:.6}", &y[..4], &yref[..4]);
    }
}
