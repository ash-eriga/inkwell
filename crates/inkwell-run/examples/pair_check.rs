// 2 gemvs on 2 streams vs serial on 1 stream — does pairing overlap?
fn main() {
    let repo = inkwell_st::StRepo::open("/home/ash/models/inkling").unwrap();
    let base = std::env::var("W").unwrap_or("model.llm.layers.0.attn.wk_dv".into());
    let base2 = std::env::var("W2").unwrap_or("model.llm.layers.0.attn.wv_dv".into());
    let load = |b: &str| {
        let tr = repo.tensor(&format!("{b}.trellis")).expect("trellis");
        let (k, n) = ((tr.shape[0] * 16) as usize, (tr.shape[1] * 16) as usize);
        let k16 = tr.shape[2] as usize;
        let (ka, halfk) = (k16 / 16, k16 % 16 != 0);
        let mut trel = inkwell_cuda::DevBuf::new_bytes(tr.nbytes() as usize);
        trel.write_bytes(repo.data(&format!("{b}.trellis")).unwrap());
        let su = repo.tensor(&format!("{b}.suh")).unwrap();
        let mut suh = inkwell_cuda::DevBuf::new_bytes(su.nbytes() as usize);
        suh.write_bytes(repo.data(&format!("{b}.suh")).unwrap());
        let sv = repo.tensor(&format!("{b}.svh")).unwrap();
        let mut svh = inkwell_cuda::DevBuf::new_bytes(sv.nbytes() as usize);
        svh.write_bytes(repo.data(&format!("{b}.svh")).unwrap());
        eprintln!("{b}: k={k} n={n} ka={ka} halfk={halfk}");
        (trel, suh, svh, k, n, ka, halfk)
    };
    let (t1, s1, v1, k, n, ka, hk) = load(&base);
    let (t2, s2, v2, k2, n2, ka2, hk2) = load(&base2);
    let mut x = inkwell_cuda::DevBuf::new(k);
    let xv: Vec<f32> = (0..k).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    x.write(&xv);
    let mut x2 = inkwell_cuda::DevBuf::new(k2);
    x2.write(&xv[..k2.min(xv.len())].to_vec());
    let mut y = inkwell_cuda::DevBuf::new(n + n2);
    let mut scr_a = inkwell_cuda::DevBuf::new(k + 8 * n);
    let mut scr_b = inkwell_cuda::DevBuf::new(k2 + 8 * n2);
    let ev_pre = inkwell_cuda::new_event();
    let ev_post = inkwell_cuda::new_event();
    let side = inkwell_cuda::side_stream_raw();

    // serial baseline
    let t0 = std::time::Instant::now();
    unsafe {
        for _ in 0..40 {
            inkwell_cuda::exl3_gemv_async(t1.dev_ptr(), s1.dev_ptr(), v1.dev_ptr(),
                x.dev_ptr().cast(), y.dev_ptr().cast(), k, n, ka, hk, scr_a.dev_ptr().cast());
            inkwell_cuda::exl3_gemv_async(t2.dev_ptr(), s2.dev_ptr(), v2.dev_ptr(),
                x2.dev_ptr().cast(), y.dev_ptr().cast::<f32>().add(n), k2, n2, ka2, hk2, scr_b.dev_ptr().cast());
        }
        inkwell_cuda::DevBuf::sync();
    }
    let t_ser = t0.elapsed().as_secs_f64() * 1e3 / 40.0;

    // paired: g1 main, g2 side, joined via events (default stream semantics =
    // the "main" these _async helpers use IS stream 0 when INK_STREAM unset)
    let t0 = std::time::Instant::now();
    unsafe {
        for _ in 0..40 {
            let s0 = std::ptr::null_mut();
            inkwell_cuda::event_record(ev_pre, s0);
            inkwell_cuda::stream_wait(side, ev_pre);
            inkwell_cuda::exl3_gemv_async(t1.dev_ptr(), s1.dev_ptr(), v1.dev_ptr(),
                x.dev_ptr().cast(), y.dev_ptr().cast(), k, n, ka, hk, scr_a.dev_ptr().cast());
            inkwell_cuda::exl3_gemv_async_side(t2.dev_ptr(), s2.dev_ptr(), v2.dev_ptr(),
                x2.dev_ptr().cast(), y.dev_ptr().cast::<f32>().add(n), k2, n2, ka2, hk2, scr_b.dev_ptr().cast());
            inkwell_cuda::event_record(ev_post, side);
            inkwell_cuda::stream_wait(s0, ev_post);
        }
        inkwell_cuda::DevBuf::sync();
    }
    let t_par = t0.elapsed().as_secs_f64() * 1e3 / 40.0;
    println!("serial {t_ser:.3}ms | paired {t_par:.3}ms | {:.2}x", t_ser / t_par);
}
