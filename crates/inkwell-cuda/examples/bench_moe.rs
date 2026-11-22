use inkwell_cuda as _;

use std::ffi::c_void;
use std::time::Instant;

extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn ink_launch_mmvq_iq2_xxs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q2_0_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq1_s_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_nvfp4_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_s_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq4_xs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
}

fn alloc(bytes: usize) -> *mut c_void {
    unsafe {
        let mut p = std::ptr::null_mut();
        assert_eq!(cudaMalloc(&raw mut p, bytes), 0);
        p
    }
}

fn main() {
    const NR: i64 = 2304; // gu rows (2*ff) and down rows = embd
    const NC: i64 = 2304; // max cols (down=1152 also tested)
    const NS: i32 = 8;
    let cases: [(&str, unsafe extern "C" fn(*const *const c_void, *const f32, *mut f32, i64, i64, i64, i32, *mut c_void), usize); 7] = [
        ("iq2_xxs", ink_launch_mmvq_iq2_xxs_moe, 66 * 256 / 256), // bytes per 64 vals
        ("q2_0", ink_launch_mmvq_q2_0_moe, 18),
        ("iq1_s", ink_launch_mmvq_iq1_s_moe, 50 * 256 / 256),
        ("nvfp4", ink_launch_mmvq_nvfp4_moe, 36),
        ("iq2_xs", ink_launch_mmvq_iq2_xs_moe, 74 * 256 / 256),
        ("iq2_s", ink_launch_mmvq_iq2_s_moe, 82 * 256 / 256),
        ("iq4_xs", ink_launch_mmvq_iq4_xs_moe, 136 * 256 / 256),
    ];
    unsafe {
        for (name, f, b64) in cases {
            for &nc in &[2304i64, 1152] {
                let row_b = (nc as usize / 64) * b64;
                let mut wp = Vec::new();
                for _ in 0..NS {
                    let p = alloc((NR as usize) * row_b);
                    let junk = vec![0x5Au8; (NR as usize) * row_b];
                    cudaMemcpy(p, junk.as_ptr().cast(), junk.len(), 1);
                    wp.push(p as *const c_void);
                }
                let x = alloc((nc as usize) * 4 * NS as usize);
                let y = alloc((NR as usize) * 4 * NS as usize);
                let wpa = alloc(NS as usize * 8);
                cudaMemcpy(wpa, wp.as_ptr().cast(), NS as usize * 8, 1);
                // warmup
                for _ in 0..5 { f(wpa.cast(), x.cast(), y.cast(), NR, nc, nc, NS, std::ptr::null_mut()); }
                cudaDeviceSynchronize();
                let t = Instant::now();
                let iters = 50;
                for _ in 0..iters { f(wpa.cast(), x.cast(), y.cast(), NR, nc, nc, NS, std::ptr::null_mut()); }
                cudaDeviceSynchronize();
                let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
                let mib = (NS as usize * NR as usize * row_b) as f64 / 1048576.0;
                println!("{name:8} ncols={nc:5} {ms:7.3} ms  ({mib:.1} MiB -> {:.1} GB/s)", mib / ms);
                for p in wp { cudaFree(p as *mut c_void); }
                cudaFree(x); cudaFree(y); cudaFree(wpa);
            }
        }
    }
}
