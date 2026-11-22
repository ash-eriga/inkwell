//! GOLD_EXL3: CPU reference (ported from exllamav3 tests/testlib/trellis.py)
//! vs inkwell_exl3_* kernels — mul1 codebook, integer and half-integer K.
use inkwell_cuda as _;
use std::ffi::c_void;
extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn ink_launch_exl3_dbg_had(x: *const f32, suh: *const c_void, out: *mut f32, k: i64, stream: *mut c_void);
    fn ink_launch_exl3_dbg(trel: *const c_void, out: *mut f32, ka: i64, halfk: i64, stream: *mut c_void);
    fn ink_launch_exl3_gemv(trel: *const c_void, suh: *const c_void, svh: *const c_void,
        x: *const f32, y: *mut f32, k: i64, n: i64, ka: i64, halfk: i64,
        scratch: *mut f32, stream: *mut c_void);
}

fn f16(b: u16) -> f64 {
    let s = ((b >> 15) & 1) as f64; let e = ((b >> 10) & 31) as i32; let m = (b & 1023) as f64;
    if e == 0 { return if m == 0.0 { 0.0 } else { (1.0 - 2.0 * s) * (m / 1024.0) * f64::powi(2.0, -14) }; }
    if e == 31 { return f64::INFINITY; }
    (1.0 - 2.0 * s) * (1.0 + m / 1024.0) * f64::powi(2.0, e - 15)
}
fn to_f16(v: f64) -> u16 { // round-to-nearest-even
    if v == 0.0 { return 0; }
    let sign = if v < 0.0 { 0x8000u16 } else { 0 };
    let mut a = v.abs();
    if a >= 65504.0 { return sign | 0x7bff; }
    let mut e = 0i32;
    while a >= 2.0 { a /= 2.0; e += 1; }
    while a < 1.0 { a *= 2.0; e -= 1; }
    if e > 15 { return sign | 0x7bff; }
    if e >= -14 {
        let mut m = (a - 1.0) * 1024.0;
        let mut mi = m.round() as u32;
        if mi == 1024 { mi = 0; e += 1; if e > 15 { return sign | 0x7bff; } }
        return sign | (((e + 15) as u16) << 10) | mi as u16;
    }
    // subnormal
    let m = (a * f64::powi(2.0, 24)).round() as u32;
    sign | m.min(1023) as u16
}
fn dec_mul1(w: u32) -> f64 {
    let x = w.wrapping_mul(0x83DCD12D);
    let bsum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
    let h = f16((0x6400 + bsum) as u16); // fp16 bits 0x6400 = 1024.0; s = bytesum + 0x6400
    f16(to_f16(h * f16(0x1eee) + f16(0xc931)))
}
fn step_widths(ka: usize, halfk: bool) -> Vec<usize> {
    (0..256).map(|i| ka + if halfk { ((0xAAAAusize >> (i & 15)) & 1) } else { 0 }).collect()
}
fn tile_perm() -> [usize; 256] {
    let mut p = [0usize; 256];
    for t in 0..32 { for j in 0..8 {
        let r = (t % 4) * 2 + (j & 1) + if j & 2 > 0 { 8 } else { 0 };
        let c = t / 4 + if j & 4 > 0 { 8 } else { 0 };
        p[t * 8 + j] = r * 16 + c;
    }}
    p
}
// ring bits MSB-first from packed u16 tile (u16[2j]=low half of word j)
fn unpack_tile(words16: &[u16], ka: usize, halfk: bool) -> Vec<u16> {
    let d = step_widths(ka, halfk);
    let r: usize = d.iter().sum();
    let nw = r / 32;
    let mut bits = vec![0u8; r];
    for j in 0..nw {
        let w = (words16[2 * j] as u32) | ((words16[2 * j + 1] as u32) << 16);
        for b in 0..32 { bits[32 * j + b] = ((w >> (31 - b)) & 1) as u8; }
    }
    let s: Vec<usize> = d.iter().scan(0, |a, &x| { *a += x; Some(*a) }).collect();
    (0..256).map(|i| {
        let mut v = 0u16;
        for b in 0..16 { v |= (bits[(s[i] - 16 + b + r) % r] as u16) << (15 - b); }
        v
    }).collect()
}
fn sylvester(n: usize) -> Vec<Vec<f64>> {
    let mut h = vec![vec![1.0f64]];
    while h.len() < n {
        let m = h.len();
        let mut n2 = vec![vec![0.0; 2 * m]; 2 * m];
        for i in 0..m { for j in 0..m {
            n2[i][j] = h[i][j]; n2[i][m + j] = h[i][j];
            n2[m + i][j] = h[i][j]; n2[m + i][m + j] = -h[i][j];
        }}
        h = n2;
    }
    h
}

fn run(tk: usize, tn: usize, ka: usize, halfk: bool, tag: &str) {
    let u16pt = if halfk { 16 * ka + 8 } else { 16 * ka }; // uint16 per tile = 16K
    let ntiles = tk * tn;
    let mut trel = vec![0u16; ntiles * u16pt];
    let mut st = 0x9E3779B97F4A7C15u64;
    for v in trel.iter_mut() {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17; // xorshift64
        *v = st as u16;
    }
    let perm = tile_perm();
    // decode -> What[k][n]
    let k = tk * 16; let n = tn * 16;
    let mut what = vec![vec![0f64; n]; k];
    for t in 0..ntiles {
        let tk_i = t / tn; let tn_i = t % tn;
        let w = unpack_tile(&trel[t * u16pt..(t + 1) * u16pt], ka, halfk);
        for i in 0..256 {
            let el = perm[i]; let r = el / 16; let c = el % 16;
            what[tk_i * 16 + r][tn_i * 16 + c] = dec_mul1(w[i] as u32);
        }
    }
    // suh[k], svh[n] as fp16
    let suh: Vec<u16> = (0..k).map(|i| to_f16(0.5 + (i % 13) as f64 * 0.03)).collect();
    let svh: Vec<u16> = (0..n).map(|i| to_f16(0.4 + (i % 11) as f64 * 0.05)).collect();
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 0.5).collect();
    // W = diag(suh) H What H diag(svh); y = x @ W
    let h = sylvester(128);
    let mut w1 = vec![vec![0f64; n]; k];
    for kb in 0..k / 128 {
        for i in 0..128 { for j in 0..n {
            let mut a = 0.0;
            for l in 0..128 { a += h[i][l] * what[kb * 128 + l][j]; }
            w1[kb * 128 + i][j] = a;
        }}
    }
    let mut w2 = vec![vec![0f64; n]; k];
    for i in 0..k { for nb in 0..n / 128 {
        for j in 0..128 {
            let mut a = 0.0;
            for l in 0..128 { a += w1[i][nb * 128 + l] * h[l][j]; }
            w2[i][nb * 128 + j] = a;
        }
    }}
    let mut yref = vec![0f64; n];
    for j in 0..n {
        let mut a = 0.0;
        for i in 0..k { a += x[i] as f64 * w2[i][j] * f16(suh[i]) * f16(svh[j]); }
        a /= 128.0; // kernel uses H/sqrt(128) on both sides
        yref[j] = a;
    }
    eprintln!("{tag} dbg: what00={} w1_00={} w2_00={} maxw2={}", what[0][0], w1[0][0], w2[0][0],
        w2.iter().flatten().fold(0f64, |a,&b| a.max(b.abs())));
    // stage-1: decode tile0 on CPU
    let cpu_dec: Vec<f64> = unpack_tile(&trel[..u16pt], ka, halfk).iter().map(|&w| dec_mul1(w as u32)).collect();
    unsafe {
        let mut dt = std::ptr::null_mut(); cudaMalloc(&raw mut dt, trel.len() * 2);
        cudaMemcpy(dt, trel.as_ptr().cast(), trel.len() * 2, 1);
        let mut dout = std::ptr::null_mut(); cudaMalloc(&raw mut dout, 256 * 4);
        ink_launch_exl3_dbg(dt, dout.cast(), ka as i64, halfk as i64, std::ptr::null_mut());
        cudaDeviceSynchronize();
        let mut gdec = vec![0f32; 256];
        cudaMemcpy(gdec.as_mut_ptr().cast(), dout, 256 * 4, 2);
        let dmd = (0..256).map(|i| (gdec[i] as f64 - cpu_dec[i]).abs()).fold(0f64, f64::max);
        eprintln!("{tag} decode-tile0 maxdiff {dmd:.6} gpu {:?} cpu {:?}", &gdec[..4], &cpu_dec[..4]);
        // stage-2: xh = H*(suh*x)/sqrt(128)
        let mut du2 = std::ptr::null_mut(); cudaMalloc(&raw mut du2, k * 2);
        cudaMemcpy(du2, suh.as_ptr().cast(), k * 2, 1);
        let mut dxh = std::ptr::null_mut(); cudaMalloc(&raw mut dxh, k * 4);
        cudaMemcpy(dxh, x.as_ptr().cast(), k * 4, 1);
        let mut dxo = std::ptr::null_mut(); cudaMalloc(&raw mut dxo, k * 4);
        ink_launch_exl3_dbg_had(dxh.cast(), du2, dxo.cast(), k as i64, std::ptr::null_mut());
        cudaDeviceSynchronize();
        let mut gxh = vec![0f32; k];
        cudaMemcpy(gxh.as_mut_ptr().cast(), dxo, k * 4, 2);
        let h128 = sylvester(128);
        let mut cxh = vec![0f64; k];
        for blk in 0..k / 128 { for i in 0..128 {
            let mut a = 0.0;
            for l in 0..128 { a += h128[i][l] * f16(suh[blk * 128 + l]) * x[blk * 128 + l] as f64; }
            cxh[blk * 128 + i] = a / 128f64.sqrt();
        }}
        let xmd = (0..k).map(|i| (gxh[i] as f64 - cxh[i]).abs()).fold(0f64, f64::max);
        eprintln!("{tag} xh maxdiff {xmd:.6} gpu {:?} cpu {:?}", &gxh[..4], &cxh[..4]);
        cudaMemcpy(dt, trel.as_ptr().cast(), trel.len() * 2, 1);
        let mut du = std::ptr::null_mut(); cudaMalloc(&raw mut du, k * 2);
        cudaMemcpy(du, suh.as_ptr().cast(), k * 2, 1);
        let mut dv = std::ptr::null_mut(); cudaMalloc(&raw mut dv, n * 2);
        cudaMemcpy(dv, svh.as_ptr().cast(), n * 2, 1);
        let mut dx = std::ptr::null_mut(); cudaMalloc(&raw mut dx, k * 4);
        cudaMemcpy(dx, x.as_ptr().cast(), k * 4, 1);
        let mut dy = std::ptr::null_mut(); cudaMalloc(&raw mut dy, n * 4);
        let mut dsc = std::ptr::null_mut(); cudaMalloc(&raw mut dsc, (k + n) * 4);
        ink_launch_exl3_gemv(dt, du, dv, dx.cast(), dy.cast(), k as i64, n as i64,
            ka as i64, halfk as i64, dsc.cast(), std::ptr::null_mut());
        cudaDeviceSynchronize();
        let mut y = vec![0f32; n];
        cudaMemcpy(y.as_mut_ptr().cast(), dy, n * 4, 2);
        let mut md = 0f64;
        let mut rel = 0f64;
        for i in 0..n {
            md = md.max((y[i] as f64 - yref[i]).abs());
            rel = rel.max((y[i] as f64 - yref[i]).abs() / yref[i].abs().max(1e-9));
        }
        println!("{tag}: k={k} n={n} ka={ka} halfk={halfk} gpu {:?} cpu {:?} maxdiff {md:.6} rel {rel:.4}", &y[..4], &yref[..4]);
    }
}

fn main() {
    run(8, 8, 4, false, "K4-int-128");
    run(8, 8, 2, false, "K2-int-128");
    run(8, 8, 2, true, "K2.5-half-128"); // ka=2 -> K=2.5
    run(8, 8, 1, true, "K1.5-half-128"); // ka=1 -> K=1.5
    run(9, 9, 3, true, "K3.5-144x144"); // ka=3 -> K=3.5, attn-sized
}
