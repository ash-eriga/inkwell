use inkwell_cuda as _;

use std::ffi::c_void;
use std::time::Instant;

extern "C" {
    fn cudaMalloc(p: *mut *mut c_void, bytes: usize) -> i32;
    fn cudaFree(p: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn ink_launch_mmvq_iq2_xxs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xxs(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xxs_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q2_0_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q2_0_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq1_s_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq1_s_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_nvfp4_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_nvfp4_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_attn_decode(K: *const f32, V: *const f32, Q: *const f32, bias: *const f32, mask: *const f32, out: *mut f32, nh: i64, hkv: i64, n_kv: i64, hd: i64, k_s1: i64, k_s2: i64, v_s1: i64, v_s2: i64, q_s2: i64, b_s2: i64, scale: f32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_s_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xs_moe_i8(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_xs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq2_s_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_iq4_xs_moe(wp: *const *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, x_stride: i64, nslots: i32, stream: *mut c_void);
    fn ink_launch_mmvq_q8_0_i8(w: *const c_void, x: *const f32, y: *mut f32, nrows: i64, ncols: i64, stream: *mut c_void);
}

fn alloc(bytes: usize) -> *mut c_void {
    unsafe {
        let mut p = std::ptr::null_mut();
        assert_eq!(cudaMalloc(&raw mut p, bytes), 0);
        p
    }
}

fn main() {
    if std::env::var("GOLD_ATTN").is_ok() { gold_attn(); return; }
    if std::env::var("GOLD_Q80").is_ok() { gold_q80(); return; }
    const NR: i64 = 2304; // gu rows (2*ff) and down rows = embd
    const NC: i64 = 2304; // max cols (down=1152 also tested)
    const NS: i32 = 8;
    let cases: [(&str, unsafe extern "C" fn(*const *const c_void, *const f32, *mut f32, i64, i64, i64, i32, *mut c_void), usize); 13] = [
        ("iq2_xxs", ink_launch_mmvq_iq2_xxs_moe, 66 * 256 / 256), // bytes per 64 vals
        ("iq2xxs_i8", ink_launch_mmvq_iq2_xxs_moe_i8, 66),
        ("q2_0", ink_launch_mmvq_q2_0_moe, 18),
        ("q2_0_i8", ink_launch_mmvq_q2_0_moe_i8, 18),
        ("iq1_s", ink_launch_mmvq_iq1_s_moe, 50 * 256 / 256),
        ("iq1_s_i8", ink_launch_mmvq_iq1_s_moe_i8, 50 * 256 / 256),
        ("nvfp4", ink_launch_mmvq_nvfp4_moe, 36),
        ("nvfp4_i8", ink_launch_mmvq_nvfp4_moe_i8, 36),
        ("iq2_xs", ink_launch_mmvq_iq2_xs_moe, 74 * 256 / 256),
        ("iq2xs_i8", ink_launch_mmvq_iq2_xs_moe_i8, 74),
        ("iq2s_i8", ink_launch_mmvq_iq2_s_moe_i8, 82),
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
                let xhost: Vec<f32> = (0..nc as usize * NS as usize).map(|i| ((i * 37 % 977) as f32 - 400.) * 0.001).collect();
                cudaMemcpy(x, xhost.as_ptr().cast(), xhost.len() * 4, 1);
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
                if name == "iq2xxs_i8" && nc == 2304 {
                    let yref = alloc((NR as usize) * 4 * NS as usize);
                    let ynew = alloc((NR as usize) * 4 * NS as usize);
                    {   // sanity: fp-moe slot0 vs dense-fp (unchanged reference kernel)
                        let yd = alloc((NR as usize) * 4);
                        ink_launch_mmvq_iq2_xxs(wp[0], x.cast(), yd.cast(), NR, nc, std::ptr::null_mut());
                        ink_launch_mmvq_iq2_xxs_moe(wpa.cast(), x.cast(), yref.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                        cudaDeviceSynchronize();
                        let mut a2 = vec![0f32; NR as usize]; let mut b2 = vec![0f32; NR as usize];
                        cudaMemcpy(a2.as_mut_ptr().cast(), yref, NR as usize * 4, 2);
                        cudaMemcpy(b2.as_mut_ptr().cast(), yd, NR as usize * 4, 2);
                        let mut md2 = 0f32;
                        for i in 0..NR as usize { md2 = md2.max((a2[i]-b2[i]).abs()); }
                        println!("  fp-moe vs fp-dense maxdiff={:.3e}", md2);
                        cudaFree(yd);
                    }
                    ink_launch_mmvq_iq2_xxs_moe(wpa.cast(), x.cast(), yref.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    ink_launch_mmvq_iq2_xxs_moe_i8(wpa.cast(), x.cast(), ynew.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    cudaDeviceSynchronize();
                    let n = (NR * NS as i64) as usize;
                    let mut a = vec![0f32; n]; let mut b = vec![0f32; n];
                    cudaMemcpy(a.as_mut_ptr().cast(), yref, n * 4, 2);
                    cudaMemcpy(b.as_mut_ptr().cast(), ynew, n * 4, 2);
                    let mut md = 0f32; let mut denom = 0f32;
                    for i in 0..n { md = md.max((a[i]-b[i]).abs()); denom = denom.max(a[i].abs()); }
                    println!("  iq2xxs_i8 vs fp maxdiff={:.3e} (max |y|={:.2})", md, denom);
                    cudaFree(yref); cudaFree(ynew);
                }
                if name == "iq1_s_i8" && nc == 2304 {
                    let yref = alloc((NR as usize) * 4 * NS as usize);
                    let ynew = alloc((NR as usize) * 4 * NS as usize);
                    ink_launch_mmvq_iq1_s_moe(wpa.cast(), x.cast(), yref.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    ink_launch_mmvq_iq1_s_moe_i8(wpa.cast(), x.cast(), ynew.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    cudaDeviceSynchronize();
                    let n = (NR * NS as i64) as usize;
                    let mut a = vec![0f32; n]; let mut b = vec![0f32; n];
                    cudaMemcpy(a.as_mut_ptr().cast(), yref, n * 4, 2);
                    cudaMemcpy(b.as_mut_ptr().cast(), ynew, n * 4, 2);
                    let mut md = 0f32; let mut denom = 0f32;
                    for i in 0..n { md = md.max((a[i]-b[i]).abs()); denom = denom.max(a[i].abs()); }
                    println!("  iq1_s_i8 vs fp maxdiff={:.3e} (max |y|={:.2})", md, denom);
                    cudaFree(yref); cudaFree(ynew);
                }
                if name == "q2_0_i8" && nc == 2304 {
                    let yref = alloc((NR as usize) * 4 * NS as usize);
                    let ynew = alloc((NR as usize) * 4 * NS as usize);
                    ink_launch_mmvq_q2_0_moe(wpa.cast(), x.cast(), yref.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    ink_launch_mmvq_q2_0_moe_i8(wpa.cast(), x.cast(), ynew.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    cudaDeviceSynchronize();
                    let n = (NR * NS as i64) as usize;
                    let mut a = vec![0f32; n]; let mut b = vec![0f32; n];
                    cudaMemcpy(a.as_mut_ptr().cast(), yref, n * 4, 2);
                    cudaMemcpy(b.as_mut_ptr().cast(), ynew, n * 4, 2);
                    let mut md = 0f32; let mut denom = 0f32;
                    for i in 0..n { md = md.max((a[i]-b[i]).abs()); denom = denom.max(a[i].abs()); }
                    println!("  q2_0_i8 vs fp maxdiff={:.3e} (max |y|={:.2})", md, denom);
                    cudaFree(yref); cudaFree(ynew);
                }
                if name == "nvfp4_i8" && nc == 2304 {
                    // correctness vs fp version: rerun both into separate y, compare
                    let yref = alloc((NR as usize) * 4 * NS as usize);
                    let ynew = alloc((NR as usize) * 4 * NS as usize);
                    ink_launch_mmvq_nvfp4_moe(wpa.cast(), x.cast(), yref.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    ink_launch_mmvq_nvfp4_moe_i8(wpa.cast(), x.cast(), ynew.cast(), NR, nc, nc, NS, std::ptr::null_mut());
                    cudaDeviceSynchronize();
                    let n = (NR * NS as i64) as usize;
                    let mut a = vec![0f32; n]; let mut b = vec![0f32; n];
                    cudaMemcpy(a.as_mut_ptr().cast(), yref, n * 4, 2);
                    cudaMemcpy(b.as_mut_ptr().cast(), ynew, n * 4, 2);
                    let mut md = 0f32; let mut denom = 0f32;
                    for i in 0..n { md = md.max((a[i]-b[i]).abs()); denom = denom.max(a[i].abs()); }
                    println!("  nvfp4_i8 vs fp maxdiff={:.3e} (max |y|={:.2})", md, denom);
                    cudaFree(yref); cudaFree(ynew);
                }
                for p in wp { cudaFree(p as *mut c_void); }
                cudaFree(x); cudaFree(y); cudaFree(wpa);
            }
        }
    }
}

fn gold_attn() {
    let (nh, hkv, n_kv, hd) = (24i64, 8i64, 192i64, 128i64);
    let k: Vec<f32> = (0..hd * n_kv * hkv).map(|i| ((i * 7) % 13) as f32 * 0.01 - 0.06).collect();
    let v: Vec<f32> = (0..hd * n_kv * hkv).map(|i| ((i * 11) % 17) as f32 * 0.01 - 0.05).collect();
    let q: Vec<f32> = (0..hd * nh).map(|i| ((i * 5) % 19) as f32 * 0.01 - 0.05).collect();
    let bias: Vec<f32> = (0..n_kv * nh).map(|i| ((i * 3) % 23) as f32 * 0.001 - 0.01).collect();
    let mask: Vec<f32> = (0..=n_kv).map(|j| if j < n_kv / 2 || j == n_kv { 0.0 } else { f32::NEG_INFINITY }).collect();
    let mut out = vec![0f32; (hd * nh) as usize];
    unsafe {
        let up = |d: &Vec<f32>| -> *mut c_void {
            let mut p: *mut c_void = std::ptr::null_mut();
            assert_eq!(cudaMalloc(&raw mut p, d.len() * 4), 0);
            cudaMemcpy(p, d.as_ptr().cast(), d.len() * 4, 1);
            p
        };
        let dk = up(&k); let dv = up(&v); let dq = up(&q); let db = up(&bias); let dm = up(&mask);
        let mut do_: *mut c_void = std::ptr::null_mut();
        assert_eq!(cudaMalloc(&raw mut do_, (hd * nh * 4) as usize), 0);
        ink_launch_attn_decode(dk.cast(), dv.cast(), dq.cast(), db.cast(), dm.cast(), do_.cast(),
            nh, hkv, n_kv, hd, hd, hd * n_kv, hd, hd * n_kv, hd, n_kv, 1.0 / (hd as f32).sqrt(), std::ptr::null_mut());
        cudaDeviceSynchronize();
        cudaMemcpy(out.as_mut_ptr().cast(), do_, (hd * nh * 4) as usize, 2);
    }
    let gqa = nh / hkv;
    let scale = 1.0 / (hd as f32).sqrt();
    for h in 0..nh {
        let hk = h / gqa;
        let mut mx = mask[n_kv as usize];
        for j in 0..n_kv {
            let mut a = 0f32;
            for d in 0..hd { a += q[(d + h * hd) as usize] * k[(d + j * hd + hk * hd * n_kv) as usize]; }
            let s = a * scale + bias[(j + h * n_kv) as usize] + mask[j as usize];
            mx = mx.max(s);
        }
        let mut den = (mask[n_kv as usize] - mx).exp();
        for j in 0..n_kv {
            let mut a = 0f32;
            for d in 0..hd { a += q[(d + h * hd) as usize] * k[(d + j * hd + hk * hd * n_kv) as usize]; }
            den += (a * scale + bias[(j + h * n_kv) as usize] + mask[j as usize] - mx).exp();
        }
        for d in 0..hd {
            let mut o = 0f32;
            for j in 0..n_kv {
                let mut a = 0f32;
                for dd in 0..hd { a += q[(dd + h * hd) as usize] * k[(dd + j * hd + hk * hd * n_kv) as usize]; }
                let s = a * scale + bias[(j + h * n_kv) as usize] + mask[j as usize];
                o += (s - mx).exp() / den * v[(d + j * hd + hk * hd * n_kv) as usize];
            }
            let e = (out[(d + h * hd) as usize] - o).abs();
            assert!(e < 1e-3, "h{h} d{d}: got {} want {o} (e={e})", out[(d + h * hd) as usize]);
        }
    }
    println!("attn gold ok");
}


// CPU-ref vs kernel for dense q8_0_i8: replicates per-32 x-quant (sc=max/127, rint)
// then y[r] = sum_b d_b * sx_b * (xi . qs). block_q8_0 = {u16 d; i8 qs[32]}.
fn gold_q80() {
    let (nrows, ncols) = (256i64, 2304i64);
    let nblk = (ncols / 32) as usize;
    let mut w = vec![0u8; (nrows as usize) * nblk * 34];
    for (i, b) in w.iter_mut().enumerate() {
        *b = (((i * 7 + 3) % 251) as u8).wrapping_add(1);
    }
    // keep d a small positive fp16 (~2^-6 pattern), qs bytes arbitrary above
    for r in 0..(nrows as usize) {
        for b in 0..nblk {
            let o = (r * nblk + b) * 34;
            w[o] = 0x00; w[o + 1] = 0x2C; // fp16 ~0.09375
        }
    }
    let x: Vec<f32> = (0..ncols).map(|i| ((i * 13) % 29) as f32 * 0.01 - 0.14).collect();
    let mut y = vec![0f32; nrows as usize];
    let mut dx: *mut c_void = std::ptr::null_mut();
    unsafe {
        let dw = alloc(w.len());
        cudaMemcpy(dw, w.as_ptr().cast(), w.len(), 1);
        dx = alloc(x.len() * 4);
        cudaMemcpy(dx, x.as_ptr().cast(), x.len() * 4, 1);
        let dy = alloc(nrows as usize * 4);
        ink_launch_mmvq_q8_0_i8(dw.cast(), dx.cast(), dy.cast(), nrows, ncols, std::ptr::null_mut());
        cudaDeviceSynchronize();
        cudaMemcpy(y.as_mut_ptr().cast(), dy, nrows as usize * 4, 2);
    }
    // CPU ref with identical x-quant + integer dot
    let mut sx = vec![0f32; nblk];
    let mut xi = vec![0i32; ncols as usize];
    for b in 0..nblk {
        let mut m = 0f32;
        for i in 0..32 { m = m.max(x[b * 32 + i].abs()); }
        sx[b] = m / 127.0;
        for i in 0..32 {
            xi[b * 32 + i] = if sx[b] > 0.0 { (x[b * 32 + i] / sx[b]).round() as i32 } else { 0 };
        }
    }
    let dval = f16_to_f32(0x2C00);
    let mut maxrel = 0f32;
    for r in 0..(nrows as usize) {
        let mut ref_ = 0f32;
        for b in 0..nblk {
            let mut isum = 0i64;
            for i in 0..32 {
                isum += xi[b * 32 + i] as i64 * (w[(r * nblk + b) * 34 + 2 + i] as i8) as i64;
            }
            ref_ += dval * sx[b] * isum as f32;
        }
        let rel = (y[r] - ref_).abs() / ref_.abs().max(1.0);
        if rel > maxrel { maxrel = rel; }
    }
    assert!(maxrel < 5e-3, "q8_0 gold maxrel={maxrel}");
    println!("q8_0 gold ok (maxrel {maxrel:.2e})");
    // timing at realistic nrows
    let nr2 = 4608i64;
    let w2 = vec![0u8; (nr2 as usize) * nblk * 34];
    let dw2 = alloc(w2.len());
    unsafe {
        cudaMemcpy(dw2, w2.as_ptr().cast(), w2.len(), 1);
        let dy2 = alloc(nr2 as usize * 4);
        ink_launch_mmvq_q8_0_i8(dw2.cast(), dx.cast(), dy2.cast(), nr2, ncols, std::ptr::null_mut());
        cudaDeviceSynchronize();
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            ink_launch_mmvq_q8_0_i8(dw2.cast(), dx.cast(), dy2.cast(), nr2, ncols, std::ptr::null_mut());
        }
        cudaDeviceSynchronize();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 200.0;
        let mib = (nr2 as f64) * (nblk as f64) * 34.0 / 1048576.0;
        println!("q8_0_i8 nrows={nr2} ncols={ncols}: {ms:.3} ms ({mib:.1} MiB -> {:.1} GB/s)", mib / ms * 1000.0);
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let s = ((h & 0x8000) << 16) as i32;
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as u32;
    if e == 0 { return (s as f32) * (m as f32) * (2f32).powi(-24); }
    let f = (1.0 + m as f32 / 1024.0) * (2f32).powi(e - 15);
    if s != 0 { -f } else { f }
}
